"""Self-written fixed-dt model and differentiable constant-u/rho closure."""
import json
import numpy as np
import torch
from torch import nn
import cantera as ct
from .physics_numpy import Thermodynamics,transform

class SafePower(torch.autograd.Function):
    @staticmethod
    def forward(ctx,y,lam):
        ctx.save_for_backward(y);ctx.lam=lam
        return torch.clamp(y,min=0)**lam
    @staticmethod
    def backward(ctx,grad):
        (y,)=ctx.saved_tensors;lam=ctx.lam
        derivative=torch.where(y>0,lam*torch.clamp(y,min=1e-40)**(lam-1),torch.zeros_like(y))
        return grad*derivative,None

class Network(nn.Module):
    def __init__(self,widths):
        super().__init__();layers=[]
        for a,b in zip([11]+widths,widths+[11]):
            layers.append(nn.Linear(a,b))
            if b!=11:layers.append(nn.ReLU())
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)

class FlowModel(nn.Module):
    def __init__(self,cfg,stats):
        super().__init__();self.cfg=cfg;self.dt=cfg['dt'];self.lam=cfg['boxcox_lambda']
        self.low=Network(cfg['hidden_layers']);self.high=Network(cfg['hidden_layers'])
        for name in ['low','high']:
            self.register_buffer(name+'_mean',torch.tensor(stats[name]['mean'],dtype=torch.float64));self.register_buffer(name+'_std',torch.tensor(stats[name]['std'],dtype=torch.float64))
        self.register_buffer('target_scale',torch.tensor(cfg['target_scale'],dtype=torch.float64))
        th=Thermodynamics(ct.Solution(cfg['mechanism']))
        for name,val in [('W',th.W),('H',th.h),('O',th.o),('a_low',th.low),('a_high',th.high),('mid',th.mid)]:self.register_buffer(name,torch.tensor(val,dtype=torch.float64))
        self.R=th.R
    def transform(self,x):return torch.cat([x[:,:2],(SafePower.apply(x[:,2:],self.lam)-1)/self.lam],1)
    def predict_target(self,x,name):
        z=self.transform(x);return getattr(self,name)(((z-getattr(self,name+'_mean'))/getattr(self,name+'_std')).float())
    def gradient(self,x):
        a,b=self.cfg['blend_temperature_K'];T=x[:,0];out=torch.zeros_like(x)
        low=T<b;high=T>a
        if low.any():
            w=torch.clamp((b-T[low])/(b-a),0,1)[:,None];p=self.predict_target(x[low],'low').double();out[low]=w*self.target_scale*torch.sinh(torch.clamp(p,-25,25))
        if high.any():
            w=torch.clamp((T[high]-a)/(b-a),0,1)[:,None];p=self.predict_target(x[high],'high').double();out[high]=out[high]+w*self.target_scale*torch.sinh(torch.clamp(p,-25,25))
        out=torch.cat([out[:,:-1],torch.zeros_like(out[:,-1:])],1)
        return out
    def energy_cv(self,T,Y):
        t=T[:,None];a=torch.where(t[:,:,None]<=self.mid[None,:,None],self.a_low[None,:,:],self.a_high[None,:,:])
        uRT=a[:,:,0]-1+a[:,:,1]*t/2+a[:,:,2]*t*t/3+a[:,:,3]*t**3/4+a[:,:,4]*t**4/5+a[:,:,5]/t
        cvR=a[:,:,0]-1+a[:,:,1]*t+a[:,:,2]*t*t+a[:,:,3]*t**3+a[:,:,4]*t**4
        return (Y*uRT*self.R/self.W*t).sum(1),(Y*cvR*self.R/self.W).sum(1)
    def invariants(self,x):
        Y=x[:,2:];T=x[:,0];u=self.energy_cv(T,Y)[0];rho=x[:,1]/(self.R*T*(Y/self.W).sum(1))
        return u,rho,Y@self.H,Y@self.O,Y[:,-1].clone()
    def project(self,Y,inv):
        u,rho,H,O,N2=inv;Y=torch.clamp(Y,min=0);rad=Y[:,3:8]
        h=rad@self.H[3:8];o=rad@self.O[3:8]
        scale=torch.minimum(torch.ones_like(H),torch.minimum(H/torch.clamp(h,min=1e-20),O/torch.clamp(o,min=1e-20)))
        rad=rad*scale[:,None];hr=torch.clamp(H-rad@self.H[3:8],min=0);orr=torch.clamp(O-rad@self.O[3:8],min=0)
        water=torch.minimum(Y[:,2],torch.minimum(hr/self.H[2],orr/self.O[2]))
        return torch.cat([(hr-water*self.H[2])[:,None],(orr-water*self.O[2])[:,None],water[:,None],rad,N2[:,None]],1)
    def close(self,Y,Tguess,inv):
        u0,rho,H,O,N2=inv;Y=self.project(Y,inv);T=Tguess
        for _ in range(5):
            u,cv=self.energy_cv(T,Y);T=torch.clamp(T-(u-u0)/cv,300,5000)
        P=rho*self.R*T*(Y/self.W).sum(1)
        return torch.cat([T[:,None],P[:,None],Y],1)
    def proposal(self,x,gradient=None):
        if gradient is None:gradient=self.gradient(x)
        z=self.transform(x)+self.dt*gradient
        Y=torch.clamp(1+self.lam*z[:,2:],min=0)**(1/self.lam)
        return torch.cat([z[:,:2],Y[:,:-1],x[:,-1:]],1)
    def step(self,x,inv=None,closed=True,strict=True):
        if inv is None:inv=self.invariants(x)
        raw=self.proposal(x)
        out=self.close(raw[:,2:],x[:,0],inv) if closed else raw
        if strict:
            finite=torch.isfinite(out).all(1)&(out[:,0]>=300)&(out[:,0]<=5000)&(out[:,1]>0)
            if closed:
                residual=torch.abs(self.energy_cv(out[:,0],out[:,2:])[0]-inv[0])
                finite=finite&(residual<=torch.maximum(torch.ones_like(residual)*1e-3,torch.abs(inv[0])*1e-9))
            if not bool(finite.all()):raise ValueError('Learned step rejected: nonfinite state, temperature bounds, or infeasible energy closure')
        return out


def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=FlowModel(ck['config'],ck['stats']);m.load_state_dict(ck['state_dict']);m.eval();return m,ck


def state_loss(model,pred,truth):
    z=model.transform(pred);zt=model.transform(truth)
    # Equal component weighting in a declared fixed physical scale; not relative to tiny Y.
    scale=torch.tensor([100.,10000.]+[.2]*8+[1.],dtype=torch.float64)
    return torch.mean(torch.abs((z[:,:-1]-zt[:,:-1])/scale[:-1]))
