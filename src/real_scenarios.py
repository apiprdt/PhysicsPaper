"""
AnomalyScenario wrappers for real experimental data (ADCD v2.0).

These use the same AnomalyScenario dataclass as the synthetic benchmarks,
so the full ADCD pipeline works without modification.

The generate_data() method falls through to the generic fallback branch in
anomaly_scenarios.py (uniform random sampling), which is fine — the scenario
metadata (classical_expr, limit_variable, correction_class, etc.) is what
guides the proposer and physics gates.

For precision analysis, use src.real_data_loader directly.
"""

from src.anomaly_scenarios import AnomalyScenario


def get_real_scenarios():
    """Return 4 AnomalyScenario objects backed by real experimental physics."""
    return [

        # R1: Mercury perihelion precession (GR correction to Newton)
        AnomalyScenario(
            name="Real: Mercury Perihelion",
            tier="real_data",
            domain="gravity",
            classical_expr="0",                  # Newtonian 2-body: no precession
            classical_variables=["r", "v"],
            classical_constants={"G": 6.674e-11, "M": 1.989e30, "c": 2.998e8},
            correction_type="additive",
            # GR correction ∝ (v/c)² — polynomial in the relativistic parameter β
            correction_expr="theta_0 * (v / c)**2",
            correction_constants={"theta_0": 1.0},
            anomaly_regime="strong gravitational field near perihelion, v/c > 0.04",
            variables_with_units={"r": "m", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial",
        ),

        # R2: Hydrogen Lamb shift (QED correction to Dirac)
        AnomalyScenario(
            name="Real: Hydrogen Lamb Shift",
            tier="real_data",
            domain="quantum_electrodynamics",
            classical_expr="-13.6 / n**2",       # Dirac energy levels (eV)
            classical_variables=["n"],
            classical_constants={},
            correction_type="additive",
            # Leading QED Lamb shift ∝ 1/n³ (power law in quantum number)
            correction_expr="theta_0 / n**3",
            correction_constants={"theta_0": 1.0},
            anomaly_regime="low principal quantum number n ~ 2, QED regime",
            variables_with_units={"n": "dimensionless"},
            classical_limit_variable="n",
            classical_limit_direction="oo",       # Δ→0 as n→∞ (classical limit)
            correction_class="power_law",
        ),

        # R3: Blackbody radiation (Planck correction to Rayleigh-Jeans)
        AnomalyScenario(
            name="Real: Blackbody Radiation",
            tier="real_data",
            domain="quantum_optics",
            classical_expr="2 * k_B * T * f**2 / c**2",   # Rayleigh-Jeans
            classical_variables=["f", "T"],
            classical_constants={"k_B": 1.381e-23, "c": 2.998e8},
            correction_type="multiplicative",
            # Planck correction: exponential suppression at high f (UV regime)
            # Delta = exp(-theta_0 * f / T) / (1 - exp(-theta_0 * f / T)) * (theta_0 * f / T) - 1
            # Simplified first-order: ~ exp(-theta_0 * f / T)
            correction_expr="exp(-theta_0 * f / T)",
            correction_constants={"theta_0": 4.799e-11},   # h/k_B
            anomaly_regime="high frequency UV/visible regime, f > 1e13 Hz",
            variables_with_units={"f": "Hz", "T": "K"},
            classical_limit_variable="f",
            classical_limit_direction="0",        # Δ→0 as f→0 (RJ regime)
            correction_class="exponential",
        ),

        # R4: Muon g-2 (QED correction to Dirac magnetic moment)
        AnomalyScenario(
            name="Real: Muon g-2",
            tier="real_data",
            domain="particle_physics",
            classical_expr="0",                   # Dirac: g=2, so a_mu=0
            classical_variables=["alpha"],
            classical_constants={"pi": 3.14159265358979},
            correction_type="additive",
            # Schwinger term (leading order): a = alpha/(2*pi)
            correction_expr="theta_0 * alpha / pi",
            correction_constants={"theta_0": 0.5},
            anomaly_regime="QED loop corrections to magnetic moment, alpha > 0",
            variables_with_units={"alpha": "dimensionless"},
            classical_limit_variable="alpha",
            classical_limit_direction="0",        # Δ→0 as alpha→0 (free field)
            correction_class="polynomial",
        ),
    ]
