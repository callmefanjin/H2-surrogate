"""Self-written thermochemistry helpers and fixed-step DLODE-style network.
The network estimates a *finite-dt transformed increment*, not an instantaneous RHS.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(x, lam=0.1):
    z = np.array(x, dtype=np.float64, copy=True)
    z[..., 2:] = (np.maximum(z[..., 2:], 0.0) ** lam - 1.0) / lam
    return z


def inverse_transform(z, lam=0.1):
    x = np.array(z, dtype=np.float64, copy=True)
    x[..., 2:] = np.maximum(1.0 + lam * x[..., 2:], 0.0) ** (1.0 / lam)
    return x


class MLP(nn.Module):
    def __init__(self, widths=(256,128,128)):
        super().__init__()
        layers=[]
        for a,b in zip([11]+list(widths),list(widths)+[11]):
            layers.append(nn.Linear(a,b))
            if b != 11: layers.append(nn.ReLU())
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)


class Thermodynamics:
    """NASA7 ideal-gas closure, validated against Cantera in test_pipeline.py."""
    def __init__(self, gas):
        self.names=gas.species_names
        self.W=gas.molecular_weights.copy()
        coeff=np.array([s.thermo.coeffs for s in gas.species()])
        if coeff.shape != (9,15): raise ValueError('NASA7 with 9 species required')
        self.mid=coeff[:,0]
        self.high=coeff[:,1:8]
        self.low=coeff[:,8:15]
        self.R=8314.46261815324
        self.h=np.array([gas.n_atoms(i,'H')*gas.atomic_weight('H')/self.W[i] for i in range(9)])
        self.o=np.array([gas.n_atoms(i,'O')*gas.atomic_weight('O')/self.W[i] for i in range(9)])

    def energy_cv(self,T,Y):
        T=np.asarray(T,dtype=float).reshape(-1,1)
        Y=np.asarray(Y,dtype=float).reshape(-1,9)
        a=np.where(T[:,:,None] <= self.mid[None,:,None],self.low[None,:,:],self.high[None,:,:])
        u_RT=(a[:,:,0]-1+a[:,:,1]*T/2+a[:,:,2]*T**2/3+a[:,:,3]*T**3/4+a[:,:,4]*T**4/5+a[:,:,5]/T)
        cv_R=(a[:,:,0]-1+a[:,:,1]*T+a[:,:,2]*T**2+a[:,:,3]*T**3+a[:,:,4]*T**4)
        u=(Y*u_RT*self.R/self.W*T).sum(axis=1)
        cv=(Y*cv_R*self.R/self.W).sum(axis=1)
        return u,cv

    def project(self,Y,H,O,N2):
        """Positive elemental closure using H2/O2 as residual reservoirs.
        This is an engineering enhancement, NOT a feature claimed by the paper.
        """
        Y=np.maximum(np.asarray(Y,dtype=float),0).copy()
        radicals=Y[:,3:8]
        h=radicals@self.h[3:8]; o=radicals@self.o[3:8]
        scale=np.minimum(1,np.minimum(H/np.maximum(h,1e-300),O/np.maximum(o,1e-300)))
        radicals*=scale[:,None]
        hr=np.maximum(H-radicals@self.h[3:8],0)
        ore=np.maximum(O-radicals@self.o[3:8],0)
        water=np.minimum(Y[:,2],np.minimum(hr/self.h[2],ore/self.o[2]))
        Y[:,2]=water
        Y[:,0]=np.maximum(hr-water*self.h[2],0)
        Y[:,1]=np.maximum(ore-water*self.o[2],0)
        Y[:,3:8]=radicals
        Y[:,8]=N2
        return Y

    def close(self,Y,Tguess,u0,rho,H,O,N2,strict=True):
        Y=self.project(Y,H,O,N2)
        T=np.asarray(Tguess,dtype=float).copy()
        for _ in range(10):
            u,cv=self.energy_cv(T,Y)
            change=(u-u0)/cv
            T=np.clip(T-change,300,5000)
            if np.max(np.abs(change))<1e-9: break
        residual=self.energy_cv(T,Y)[0]-u0
        valid=np.isfinite(T)&(np.abs(residual)<=np.maximum(1e-3,np.abs(u0)*1e-9))
        if strict and not valid.all():
            raise ValueError(f"No thermodynamically valid closure at 300–5000 K; max energy residual {np.max(np.abs(residual)):.6g} J/kg")
        P=rho*self.R*T*(Y/self.W).sum(axis=1)
        return np.column_stack((T,P,Y))
