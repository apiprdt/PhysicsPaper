"""
AnomalyScenario wrappers for real experimental data (ADCD v2.0).

RealAnomalyScenario subclasses AnomalyScenario and overrides generate_data()
to call the actual physics loaders from adcd.real_data_loader instead of the
generic uniform-random fallback.  The rest of the pipeline (proposer, physics
gates, JAX optimizer) is unchanged.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from adcd.anomaly_scenarios import AnomalyScenario
from adcd.constants import G as G_CODATA, C as C_CODATA, M_SUN as M_SUN_CODATA, K_B as K_B_CODATA
from adcd.real_data_loader import (
    load_mercury_perihelion,
    load_hydrogen_lamb_shift,
    load_blackbody_radiation,
    load_muon_g2,
    load_binary_pulsar_decay,
    binary_pulsar_prefactor,
)

# Map scenario name → loader function
_LOADERS = {
    "Real: Mercury Perihelion":  load_mercury_perihelion,
    "Real: Hydrogen Lamb Shift": load_hydrogen_lamb_shift,
    "Real: Blackbody Radiation": load_blackbody_radiation,
    "Real: Muon g-2":            load_muon_g2,
    "Real: Binary Pulsar Decay": load_binary_pulsar_decay,
}


@dataclass
class RealAnomalyScenario(AnomalyScenario):
    """AnomalyScenario that delegates generate_data() to a real physics loader.

    All other pipeline behaviour (proposer context, ARC limit checking,
    JAX optimisation) is inherited unchanged from AnomalyScenario.
    """

    def generate_data(
        self,
        n_points: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Delegates to the matching real physics loader.

        n_points and noise_level are ignored — the real loaders fix their own
        sample sizes and noise magnitudes to match physical precision.
        seed is forwarded to the loader for reproducible noise draws.
        """
        loader = _LOADERS.get(self.name)
        if loader is None:
            # Fallback to generic uniform sampling for unknown names
            return super().generate_data(n_points=n_points,
                                         noise_level=noise_level, seed=seed)
        return loader(seed=seed)


# ─────────────────────────────────────────────────────────────────────────────

def get_real_scenarios():
    """Return RealAnomalyScenario objects backed by synthetic-real hybrid data."""
    return [

        # R1: Mercury perihelion precession (GR correction to Newton)
        RealAnomalyScenario(
            name="Real: Mercury Perihelion",
            tier="real_data",
            domain="gravity",
            classical_expr="0",
            classical_variables=["vc2", "r", "v"],
            classical_constants={"G": G_CODATA, "M": M_SUN_CODATA, "c": C_CODATA},
            correction_type="additive",
            correction_expr="theta_0 * vc2",
            correction_constants={"theta_0": 1.0},
            anomaly_regime="strong gravitational field near perihelion, vc2=(v/c)^2",
            variables_with_units={"vc2": "dimensionless", "r": "m", "v": "m/s"},
            classical_limit_variable="vc2",
            classical_limit_direction="0",
            correction_class="polynomial",
        ),

        # R2: Hydrogen Lamb shift (QED correction to Dirac energy levels)
        RealAnomalyScenario(
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
        RealAnomalyScenario(
            name="Real: Blackbody Radiation",
            tier="real_data",
            domain="quantum_optics",
            classical_expr="2 * k_B * T * f**2 / c**2",   # Rayleigh-Jeans law
            classical_variables=["f", "T"],
            classical_constants={"k_B": K_B_CODATA, "c": C_CODATA},
            correction_type="multiplicative",
            # Planck correction: exponential suppression at high f (UV regime)
            correction_expr="exp(-theta_0 * f / T)",
            correction_constants={"theta_0": 4.799e-11},   # h/k_B
            anomaly_regime="high frequency UV/visible regime, f > 1e13 Hz",
            variables_with_units={"f": "Hz", "T": "K"},
            classical_limit_variable="f",
            classical_limit_direction="0",        # Δ→0 as f→0 (RJ regime)
            correction_class="exponential",
        ),

        # R4: Muon g-2 (QED correction to Dirac magnetic moment)
        RealAnomalyScenario(
            name="Real: Muon g-2",
            tier="real_data",
            domain="particle_physics",
            classical_expr="0",                   # Dirac: g=2, so a_mu = 0
            classical_variables=["alpha"],
            classical_constants={"pi": 3.14159265358979},
            correction_type="additive",
            # Schwinger term (leading order): a_mu = alpha/(2*pi)
            correction_expr="theta_0 * alpha / pi",
            correction_constants={"theta_0": 0.5},
            anomaly_regime="QED loop corrections to magnetic moment, alpha > 0",
            variables_with_units={"alpha": "dimensionless"},
            classical_limit_variable="alpha",
            classical_limit_direction="0",        # Δ→0 as alpha→0 (free field)
            correction_class="polynomial",
        ),

        # R5: Binary pulsar orbital decay (GR gravitational-wave energy loss)
        # v2.1: Only P varies; M, e folded into constant prefactor in loader.
        RealAnomalyScenario(
            name="Real: Binary Pulsar Decay",
            tier="real_data",
            domain="gravity",
            classical_expr="0",
            classical_variables=["P"],
            classical_constants={"G": G_CODATA, "c": C_CODATA},
            correction_type="additive",
            correction_expr="theta_0 * P**(-5.0/3.0)",
            correction_constants={"theta_0": binary_pulsar_prefactor()},
            anomaly_regime="compact binary inspiral (Hulse-Taylor), secular period decay, P→∞ gives Δ→0",
            variables_with_units={"P": "s"},
            classical_limit_variable="P",
            classical_limit_direction="oo",
            correction_class="power_law",
        ),
    ]
