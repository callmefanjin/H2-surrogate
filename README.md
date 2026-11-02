# H2-surrogate

Neural surrogate for hydrogen-air autoignition kinetics in a closed, adiabatic, constant-volume reactor. The model advances temperature, pressure and nine species mass fractions using a fixed time step of $10^{-7}$ s.

## Model

A neural network predicts the state increment in a transformed composition space. Each step applies a physical closure to recover temperature and pressure while maintaining the initial internal energy, density and elemental composition. Temperature-dependent networks represent the induction and hot reacting regimes.

$$
\mathbf{x}_{n+1}=\mathcal{C}\!\left[\mathcal{N}_{\theta}(\mathbf{x}_n;\Delta t)\right],
\qquad \mathbf{x}=[T,p,Y_1,\ldots,Y_9].
$$

Here, $\mathcal{N}_{\theta}$ is the learned one-step proposal and $\mathcal{C}$ is the physical closure. Predictions use the trained weights directly; Cantera provides an independent reference when requested.

The current interface accepts fresh H2/air mixtures at 1100–1400 K, 1–3 atm and equivalence ratios of 0.7–1.5, with trajectories up to 2 ms. The bundled mechanism has 9 species and 21 reactions.

## Ignition comparison

At 1250 K, 2 atm and $\phi=1$, the surrogate follows the Cantera temperature history through ignition. The ignition delays are 17.290 and 17.228 microseconds, respectively.

![Temperature history at 1250 K, 2 atm and equivalence ratio 1](figures/typical_ignition.png)

At fixed 2 atm and $\phi=1$, a scan of 21 initial temperatures from 1100 to 1400 K gives a maximum absolute ignition-delay relative error of 0.64%. All 21 points are shown. Ignition delay is defined by the first crossing of $T=T_0+400$ K; the vertical axis is logarithmic.

![Ignition delay versus inverse initial temperature at 2 atm and equivalence ratio 1](figures/tau_vs_inverse_temperature.png)

These comparisons describe the selected conditions. Accuracy varies with operating conditions; the accepted input range is not a uniform accuracy guarantee.

## Use

Requires Python 3.10 or later with PyTorch, NumPy and Cantera. From the repository directory:

```sh
python -m pip install .
h2-surrogate --temperature-K 1250 --pressure-atm 2 --phi 1 --duration-s 0.0002 --output trajectory.csv --compare
```

The Python interface supports individual and batched predictions:

```python
from h2_surrogate import FrozenSurrogate

result = FrozenSurrogate().predict(
    T0_K=1250, P0_atm=2, phi=1, duration_s=0.0002
)
if not result.completed:
    raise RuntimeError(result.metadata["error"])
result.save_csv("trajectory.csv")
```

This release provides a pretrained inference model for the stated reactor and mechanism. CPU speedup and coupled-CFD performance are not established. See [license and provenance status](LICENSE_STATUS.md) for reuse information.
