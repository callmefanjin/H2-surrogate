"""V5 fixed-time learned map: exact V1-low/V2-map warm start and differentiable closure.

The initial function uses V1 below 1500 K and the original V2 blended map above.
All inherited training statistics are retained. No chemical solver fallback.
"""
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import cantera as ct
from .legacy_physics import FlowModel as Physics, SafePower
from .physics_numpy import Thermodynamics


class MLP(nn.Module):
    def __init__(self,widths):
        super().__init__(); layers=[]
        for a,b in zip([11]+widths,widths+[11]):
            layers.append(nn.Linear(a,b))
            if b!=11: layers.append(nn.ReLU())
        self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)


class CombinedFlow(nn.Module):
    def __init__(self,cfg,stats):
        super().__init__(); self.cfg=cfg; self.dt=cfg['dt']; self.lam=.1
        self.experts=nn.ModuleDict({k:MLP([256,128,128]) for k in ['v1_low','v2_low','v2_high']})
        self.stats=copy.deepcopy(stats)
        for k,e in stats.items():
            for name in ['x_mean','x_std','y_mean','y_std','target_scale']:
                self.register_buffer(k+'_'+name,torch.tensor(e[name],dtype=torch.float64))
        th=Thermodynamics(ct.Solution(cfg['mechanism'])); self.R=th.R
        for k,v in [('W',th.W),('H',th.h),('O',th.o),('a_low',th.low),('a_high',th.high),('mid',th.mid)]:
            self.register_buffer(k,torch.tensor(v,dtype=torch.float64))
    energy_cv=Physics.energy_cv
    invariants=Physics.invariants
    def project(self,Y,inv):
        out=Physics.project(self,Y,inv)
        # The two dependent reservoir differences are nonnegative analytically.
        # Suppress only their negative floating-point residuals after subtraction;
        # independent water/radical channels and the old projects are unchanged.
        return torch.cat([torch.clamp(out[:,:2],min=0),out[:,2:]],1)
    close=Physics.close
    def transform(self,x):
        return torch.cat([x[...,:2],(SafePower.apply(x[...,2:],self.lam)-1)/self.lam],-1)
    def expert_gradient(self,z,k):
        y=self.experts[k](((z-getattr(self,k+'_x_mean'))/getattr(self,k+'_x_std')).float()).double()
        q=y*getattr(self,k+'_y_std')+getattr(self,k+'_y_mean')
        # Match the original V1 and V2 clipping limits exactly at initialization.
        limit=60 if k=='v1_low' else 25
        return getattr(self,k+'_target_scale')*torch.sinh(torch.clamp(q,-limit,limit))
    def gradient(self,x):
        z=self.transform(x);T=x[:,0];g=torch.zeros_like(x);lo=T<1500;hi=~lo
        if lo.any():g[lo]=self.expert_gradient(z[lo],'v1_low')
        if hi.any():
            zz=z[hi];w=torch.clamp((T[hi]-1450)/100,0,1)[:,None]
            gh=w*self.expert_gradient(zz,'v2_high')
            mask=(T[hi]<1550)
            if mask.any():gh[mask]=gh[mask]+(1-w[mask])*self.expert_gradient(zz[mask],'v2_low')
            g[hi]=gh
        return torch.cat([g[:,:-1],torch.zeros_like(g[:,-1:])],1)
    def scaled_target(self,x,g):
        lo=x[:,0]<1500;out=torch.zeros_like(g)
        for k,mask in [('v1_low',lo),('v2_high',~lo)]:
            if mask.any():
                q=torch.asinh(g[mask]/getattr(self,k+'_target_scale'))
                out[mask]=(q-getattr(self,k+'_y_mean'))/getattr(self,k+'_y_std')
        return out
    def proposal(self,x,g=None):
        if g is None:g=self.gradient(x)
        z=self.transform(x)+self.dt*g
        Y=torch.clamp(1+self.lam*z[:,2:],min=0)**(1/self.lam)
        return torch.cat([z[:,:2],Y[:,:-1],x[:,-1:]],1)
    def valid(self,out,inv,raw=None,closed=True):
        ok=torch.isfinite(out).all(1)&(out[:,0]>=300)&(out[:,0]<=5000)&(out[:,1]>0)&(out[:,2:].min(1).values>=0)&(out[:,2:].max(1).values<=1.1)
        if closed:
            ok=ok&(abs(self.energy_cv(out[:,0],out[:,2:])[0]-inv[0])<=torch.maximum(torch.ones_like(inv[0])*1e-3,abs(inv[0])*1e-9))
            if raw is not None:ok=ok&(abs(out[:,2:]-raw[:,2:]).sum(1)<=self.cfg.get('maximum_projection_L1',.02))
        return ok
    def step(self,x,inv=None,closed=True,strict=True):
        if inv is None:inv=self.invariants(x)
        raw=self.proposal(x);out=self.close(raw[:,2:],x[:,0],inv) if closed else raw
        if strict and not bool(self.valid(out,inv,raw,closed).all()):
            raise ValueError('Learned map rejected: bounds, nonfinite, infeasible u, or excessive projection')
        return out


def initial_model(cfg):
    a=torch.load('artifacts/model_V1_reference.pt',map_location='cpu',weights_only=False)
    b=torch.load('artifacts/model_V2_reference.pt',map_location='cpu',weights_only=False)
    e=a['models']['low'];stats={'v1_low':{k:np.asarray(e[k]).tolist() for k in ['x_mean','x_std','y_mean','y_std','target_scale']}}
    for part in ['low','high']:
        stats['v2_'+part]={'x_mean':b['stats'][part]['mean'],'x_std':b['stats'][part]['std'],'y_mean':[0.]*11,'y_std':[1.]*11,'target_scale':b['config']['target_scale']}
    m=CombinedFlow(cfg,stats);m.experts['v1_low'].load_state_dict(a['models']['low']['state_dict'])
    for part in ['low','high']:
        state={k.replace(part+'.net.','layers.'):v for k,v in b['state_dict'].items() if k.startswith(part+'.net.')}
        m.experts['v2_'+part].load_state_dict(state)
    return m


def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=CombinedFlow(ck['config'],ck['stats']);m.load_state_dict(ck['state_dict']);m.eval();return m,ck


def increment_loss(m,x,y):
    truth=(m.transform(y)-m.transform(x))/m.dt
    return torch.mean(abs(m.scaled_target(x,m.gradient(x))[:,:-1]-m.scaled_target(x,truth)[:,:-1]))


def state_loss(m,p,y):
    # Fixed physical units, not fitted to validation/test errors.
    t=torch.mean(abs(p[:,0]-y[:,0]))
    b=torch.mean(abs(m.transform(p)[:,2:10]-m.transform(y)[:,2:10]))/.01
    return t+b
