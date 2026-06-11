# Anomaly Scenarios

An anomaly scenario in ADCD is defined by a `Scenario` object. This class packages the classical equation, physical parameters, variables, dynamic ranges, and experimental noise.

## Anatomy of a Scenario

Here is how a scenario is defined programmatically:

```python
from adcd.anomaly_scenarios import AnomalyScenario

scenario = AnomalyScenario(
    name="Yukawa Gravity",
    # 1. Base variables and their physical dimensions
    variables=["r"],
    variable_units={"r": "L"},  # L = Length
    
    # 2. Free physical constants in the baseline
    constants={"G": 1.0, "M": 1.0, "m": 1.0},
    constant_units={"G": "L**3*M**-1*T**-2", "M": "M", "m": "M"},
    
    # 3. The classical baseline expression
    expr_classical_str="-G * M * m / r",
    expr_classical_units="M*L**2*T**-2",  # Energy
    
    # 4. Target variable to calculate residual on
    target_var="E",
    target_units="M*L**2*T**-2",
    
    # 5. Generation range and dynamic scale
    gen_range={"r": (0.5, 5.0)},
    limit_variable="r",
    limit_direction="inf",  # Baseline is recovered when r -> infinity
    
    # 6. Mathematical family classification for verification
    true_class="exponential",
    
    # 7. True physical correction formula
    true_correction_lambda=lambda X, th: -th[0]*np.exp(-X["r"]/th[1])/X["r"]
)
```

## Built-In Scenarios

ADCD comes with 9 standard scenarios grouped into three tiers:

### Tier A: Textbook Physical Anomalies
1. **Relativistic Kinetic Energy**: Classical $p^2/2m$ corrected to relativistic kinematics.
2. **Yukawa Gravity**: Classical Newtonian gravity $-GMm/r$ corrected with a Yukawa shielding factor $e^{-r/\lambda}$.
3. **Anharmonic Spring**: Classical harmonic potential $\frac{1}{2}kx^2$ corrected with a quartic perturbation term.

### Tier B: Cross-Domain & Engineering Anomalies
4. **Screened Coulomb Potential**: Classical Coulomb potential corrected with exponential screening.
5. **Net Radiation Heat Transfer**: Classical Stefan-Boltzmann radiative transfer corrected for non-unity emissivities.
6. **Nonlinear Fluid Drag**: Linear Stokes drag corrected with a quadratic turbulent drag term.

### Tier C: Synthetic Blind Tests
7. **Mystery-A (tanh²)**: Synthetic correction featuring hyperbolic saturation.
8. **Mystery-B (sinc)**: Synthetic oscillatory decay.
9. **Mystery-C (log-quotient)**: Synthetic logarithmic correction.
