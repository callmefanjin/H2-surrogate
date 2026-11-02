"""Command line for fresh H2/air prediction and optional reference comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import FrozenSurrogate, reference, SPECIES

def main(argv=None):
    p=argparse.ArgumentParser(description='Frozen H2-surrogate research surrogate; CPU, fixed dt=1e-7 s')
    p.add_argument('--temperature-K',type=float,required=True)
    p.add_argument('--pressure-atm',type=float,required=True)
    p.add_argument('--phi',type=float,required=True)
    p.add_argument('--duration-s',type=float,default=.002)
    p.add_argument('--output',type=Path,default=Path('trajectory.npz'))
    p.add_argument('--compare',action='store_true',help='Run independent Cantera reference')
    p.add_argument('--plot',type=Path,help='Optional PNG plot (requires matplotlib)')
    a=p.parse_args(argv)
    if a.output.suffix not in ('.csv','.npz'):p.error('--output must end in .csv or .npz')
    torch.set_num_threads(1)
    args=dict(T0_K=a.temperature_K,P0_atm=a.pressure_atm,phi=a.phi,duration_s=a.duration_s)
    try: result=FrozenSurrogate().predict(**args)
    except (ValueError,RuntimeError) as exc:p.error(str(exc))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    (result.save_csv if a.output.suffix=='.csv' else result.save_npz)(a.output)
    info=dict(result.metadata); ref=None
    if a.compare:
        ref=reference(**args); n=min(len(ref.states),len(result.states))
        error=np.abs(ref.states[:n]-result.states[:n])
        info['comparison']=dict(scope='full horizon' if ref.completed and result.completed else 'common accepted prefix ONLY',
            reference_completed=ref.completed,T_MAE_K=float(error[:,0].mean()),P_MAE_Pa=float(error[:,1].mean()),
            species_MAE=dict(zip(SPECIES,error[:,2:].mean(0).tolist())))
        ref.save_npz(a.output.with_suffix('.reference.npz'))
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(9,10))
        axes[0].plot(result.time_s*1000,result.temperature_K,label='H2-surrogate')
        axes[1].plot(result.time_s*1000,result.pressure_Pa/101325,label='H2-surrogate')
        for i,sp in enumerate(SPECIES):axes[2].plot(result.time_s*1000,result.mass_fractions[:,i],label=sp)
        if ref is not None:
            axes[0].plot(ref.time_s*1000,ref.temperature_K,'--',label='Cantera')
            axes[1].plot(ref.time_s*1000,ref.pressure_Pa/101325,'--',label='Cantera')
        for ax,label in zip(axes,['Temperature (K)','Pressure (atm)','Mass fraction']):
            ax.set(xlabel='Time (ms)',ylabel=label);ax.legend();ax.grid(alpha=.2)
        fig.suptitle('Research only — '+('complete' if result.completed else 'FAILED: accepted prefix only'))
        fig.tight_layout();a.plot.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.plot,dpi=150);plt.close(fig)
    print(json.dumps(info,indent=2,allow_nan=False))
    return 0 if result.completed and (ref is None or ref.completed) else 2

if __name__=='__main__':raise SystemExit(main())
