import torch
from h2_surrogate import FrozenSurrogate
# One thread avoids oversubscription for this small network. No speed claim.
torch.set_num_threads(1)
r = FrozenSurrogate().predict(T0_K=1250, P0_atm=2, phi=1, duration_s=2e-4)
r.save_npz('trajectory.npz')
r.save_csv('trajectory.csv')
print(r.metadata)
if not r.completed:
    raise RuntimeError('Accepted prefix saved; full horizon failed')
