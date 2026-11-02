"""Validated fresh-mixture interface to the frozen H2-surrogate map."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numbers
import numpy as np
import cantera as ct
import torch
from ._frozen.model import CombinedFlow

VERSION = '1.0.0'
DT = 1e-7
SPECIES = ('H2','O2','H2O','H','O','OH','HO2','H2O2','N2')
ASSETS = Path(__file__).parent / 'assets'
WEIGHT_SHA256 = 'f4cb2ebab2721b5071cbebd5b0be2e64d555bb838e987a90e78f6989614e3d74'
MECHANISM_SHA256 = '7c3f1750b04ac1ce20ecca6914033f677fd3382c1497eb72b5aace7e150af8ca'

def _number(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'{name} must be finite and in [{lo}, {hi}]')
    return float(value)

def _steps(duration_s):
    duration_s = _number(duration_s, 'duration_s', DT, .002)
    steps = round(duration_s / DT)
    if not math.isclose(steps * DT, duration_s, rel_tol=0, abs_tol=1e-15):
        raise ValueError('duration_s must be an integer multiple of fixed dt=1e-7 s')
    return steps

def _verify_assets():
    for name, expected in [('h2_surrogate.pt', WEIGHT_SHA256), ('H2_Li.yaml', MECHANISM_SHA256)]:
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Bundled asset integrity check failed: {name}')

def fresh_state(T0_K, P0_atm, phi):
    """Fresh H2 / (O2 + 3.76 N2); returns [K, Pa, nine mass fractions]."""
    T = _number(T0_K, 'T0_K', 1100, 1400)
    P = _number(P0_atm, 'P0_atm', 1, 3)
    phi = _number(phi, 'phi', .7, 1.5)
    _verify_assets()
    gas = ct.Solution(str(ASSETS/'H2_Li.yaml'))
    if tuple(gas.species_names) != SPECIES or gas.n_reactions != 21:
        raise RuntimeError('Mechanism identity mismatch')
    gas.TP = T, P * ct.one_atm
    gas.set_equivalence_ratio(phi, 'H2', 'O2:1,N2:3.76')
    state = np.r_[gas.T, gas.P, gas.Y]
    if not np.isfinite(state).all() or (state[2:] < 0).any() or abs(state[2:].sum()-1)>1e-12:
        raise ValueError('Invalid generated fresh composition')
    return state

@dataclass(frozen=True)
class Trajectory:
    """Accepted prefix only. Check completed before treating as full-horizon output."""
    time_s: np.ndarray
    states: np.ndarray
    metadata: dict
    @property
    def completed(self): return self.metadata['completed']
    @property
    def temperature_K(self): return self.states[:, 0]
    @property
    def pressure_Pa(self): return self.states[:, 1]
    @property
    def mass_fractions(self): return self.states[:, 2:]
    def save_npz(self, path):
        np.savez_compressed(path, time_s=self.time_s, states=self.states,
                            species=np.array(SPECIES), metadata_json=json.dumps(self.metadata, allow_nan=False))
    def save_csv(self, path):
        path = Path(path)
        np.savetxt(path, np.column_stack((self.time_s,self.states)), delimiter=',',
                   header=','.join(('time_s','T_K','P_Pa')+tuple('Y_'+s for s in SPECIES)),comments='')
        path.with_suffix(path.suffix+'.metadata.json').write_text(json.dumps(self.metadata,indent=2,allow_nan=False)+'\n')

class FrozenSurrogate:
    """CPU-only trusted bundled checkpoint; no user checkpoint or alternate mechanism."""
    def __init__(self):
        _verify_assets()
        # This legacy file contains Python pickle metadata. Verify known bytes BEFORE
        # deserialization. Do not generalize this loader to untrusted checkpoint paths.
        ck = torch.load(ASSETS/'h2_surrogate.pt', map_location='cpu', weights_only=False)
        config = dict(ck['config']); config['mechanism'] = str(ASSETS/'H2_Li.yaml')
        self._model = CombinedFlow(config, ck['stats'])
        self._model.load_state_dict(ck['state_dict']); self._model.eval()
        if self._model.dt != DT: raise RuntimeError('Checkpoint time step mismatch')

    def predict(self, *, T0_K, P0_atm, phi, duration_s=.002):
        return self.predict_batch([dict(T0_K=T0_K,P0_atm=P0_atm,phi=phi)],duration_s=duration_s)[0]

    def predict_batch(self, cases, *, duration_s=.002):
        """List of fresh-mixture dictionaries; original batched inference semantics."""
        steps = _steps(duration_s)
        cases = list(cases)
        if not cases: raise ValueError('At least one initial condition is required')
        for case in cases:
            if not isinstance(case,dict) or set(case) != {'T0_K','P0_atm','phi'}:
                raise ValueError('Each case requires exactly T0_K, P0_atm, phi; arbitrary TPY is not validated')
        initial = np.array([fresh_state(**case) for case in cases])
        x = torch.as_tensor(initial.copy(), dtype=torch.float64)
        active = np.ones(len(cases),bool)
        buffers = np.empty((len(cases),steps+1,11),dtype=np.float64); buffers[:,0]=initial
        lengths = np.ones(len(cases),int); errors = [None]*len(cases)
        with torch.inference_mode():
            inv = tuple(v.clone() for v in self._model.invariants(x))
            for k in range(1, steps+1):
                ids=np.flatnonzero(active)
                if not len(ids): break
                xx=x[ids]; iv=tuple(v[ids] for v in inv)
                try:
                    raw=self._model.proposal(xx)
                    pred=self._model.close(raw[:,2:],xx[:,0],iv)
                    good=self._model.valid(pred,iv,raw,True).cpu().numpy()
                except (ValueError,FloatingPointError,OverflowError) as exc:
                    for j in ids:
                        active[j]=False; errors[j]={'step':k,'reason':str(exc),'type':type(exc).__name__}
                    continue
                accepted=ids[good]; x[accepted]=pred[good]
                buffers[accepted,k]=pred[good].cpu().numpy(); lengths[accepted]+=1
                for j in ids[~good]:
                    active[j]=False
                    errors[j]={'step':k,'reason':'Frozen bounds/finite/nonnegative/energy/projection guard rejected step','type':'GuardrailRejection'}
        results=[]
        for j,case in enumerate(cases):
            states=buffers[j,:lengths[j]].copy()
            meta=dict(version=VERSION,model='H2-surrogate',checkpoint_sha256=WEIGHT_SHA256,
                      mechanism_sha256=MECHANISM_SHA256,initial_condition={k:float(v) for k,v in case.items()},
                      dt_s=DT,requested_duration_s=float(duration_s),accepted_duration_s=float((lengths[j]-1)*DT),
                      completed=bool(lengths[j]==steps+1),error=errors[j],species=list(SPECIES),
                      batch_size=len(cases),device='cpu',research_only=True,reference_fallback=False)
            results.append(Trajectory(np.arange(lengths[j])*DT,states,meta))
        return results

def reference(*, T0_K, P0_atm, phi, duration_s=.002):
    """Independent Cantera CV adiabatic reference, never a surrogate fallback."""
    steps=_steps(duration_s); state=fresh_state(T0_K,P0_atm,phi)
    gas=ct.Solution(str(ASSETS/'H2_Li.yaml')); gas.TPY=state[0],state[1],state[2:]
    reactor=ct.IdealGasReactor(gas,energy='on',clone=True)
    net=ct.ReactorNet([reactor]);net.rtol=1e-11;net.atol=1e-19
    states=[state]; error=None
    for k in range(1,steps+1):
        try:
            net.advance(k*DT); current=np.r_[reactor.T,reactor.phase.P,reactor.phase.Y]
            if not np.isfinite(current).all():raise ValueError('Nonfinite reference')
            states.append(current)
        except (ct.CanteraError,ValueError) as exc:
            error={'step':k,'reason':str(exc)}; break
    return Trajectory(np.arange(len(states))*DT,np.array(states),dict(model='Cantera reference',completed=len(states)==steps+1,error=error,dt_s=DT,requested_duration_s=float(duration_s)))
