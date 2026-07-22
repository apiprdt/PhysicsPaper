"""
sparc_rar_scenario.py
=====================
RealAnomalyScenario subclass for the SPARC Radial Acceleration Relation (RAR).

Plugs directly into the ADCD CorrectionOrchestrator pipeline.

Baseline (classical model): Newtonian gravity  =>  g_obs = g_bar
Anomaly:                     g_obs >> g_bar systematically at low accelerations
Discovery target:            additive correction delta(g_bar) = g_obs - g_bar

The ADCD GrammarProposer will:
 1. Receive g_bar as a variable with acceleration dimension [m/s^2]
 2. Automatically build the dimensionless ratio  g_bar / theta_k  (theta_k ~ a0)
 3. Search for correction formulas that vanish at g_bar -> inf (Newtonian limit)
 4. Optimize theta_k freely -> discovers a0 from data without any prior
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from adcd.anomaly_scenarios import AnomalyScenario


@dataclass
class SPARCRARScenario(AnomalyScenario):
    """
    ADCD scenario for the SPARC galaxy radial acceleration relation.

    Overrides generate_data() to load real SPARC data from the v2 pipeline
    (SI units, proper error budget).
    """
    data_path: str = r"E:\InternalResearch\data\kepler\kepler_sparc_clean_v2.csv"
    split: str = "train"

    def generate_data(
        self,
        n_points: int = 200,   # ignored — uses full SPARC split
        noise_level: float = 0.0,  # ignored — real measurement errors baked in
        seed: int = 42,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Load real SPARC SI data.

        Returns
        -------
        X : {"g_bar": array of g_bar values in m/s^2}
        y_obs : g_obs values in m/s^2
        y_classical : g_bar values in m/s^2  (Newtonian baseline)
        residual : g_obs - g_bar  (the anomalous acceleration surplus)
        """
        df = pd.read_csv(self.data_path)
        df = df[df["split"] == self.split].copy()

        # Sort by g_bar for clean ordering (helps ARC analysis)
        df = df.sort_values("g_bar_SI").reset_index(drop=True)

        g_bar = df["g_bar_SI"].values.astype(np.float64)
        g_obs = df["g_obs_SI"].values.astype(np.float64)

        X = {"g_bar": g_bar}
        y_obs = g_obs
        y_classical = g_bar          # Newton: g_obs = g_bar
        residual = g_obs - g_bar     # always >= 0 for valid SPARC data

        return X, y_obs, y_classical, residual


def build_sparc_rar_scenario(
    data_path: str = r"E:\InternalResearch\data\kepler\kepler_sparc_clean_v2.csv",
    split: str = "train",
) -> SPARCRARScenario:
    """
    Construct a fully specified SPARCRARScenario ready for CorrectionOrchestrator.

    Physics framing
    ---------------
    - Variable: g_bar  [m/s^2]  — baryonic centripetal acceleration from rotation curves
    - Classical limit: g_bar -> infinity  =>  correction -> 0 (Newton dominates)
    - Anomaly regime: g_bar << a0 ~ 1.2e-10 m/s^2  =>  correction ~ sqrt(a0 * g_bar)
    - Correction class: power_law at small x  (MOND asymptote: delta ~ sqrt(a0 * g_bar))
    """
    return SPARCRARScenario(
        # Required AnomalyScenario fields
        name="Real: SPARC Galaxy RAR",
        tier="real_data",
        domain="gravity",

        # Classical (Newtonian) baseline
        classical_expr="g_bar",
        classical_variables=["g_bar"],
        classical_constants={},          # No fixed constants — a0 is a free parameter

        # Anomaly description
        correction_type="additive",
        correction_expr="theta_0 * sqrt(g_bar)",   # Placeholder; ADCD will discover the true form
        correction_constants={"theta_0": 1.0},      # ADCD optimizes this

        anomaly_regime="low baryonic acceleration g_bar << a0 ~ 1.2e-10 m/s^2 (MOND regime)",
        variables_with_units={"g_bar": "m/s^2"},

        # ARC asymptotic limit gate:
        # As g_bar -> infinity, correction -> 0 (Newtonian recovery)
        classical_limit_variable="g_bar",
        classical_limit_direction="oo",   # correction vanishes as g_bar -> inf

        correction_class="power_law",     # MOND deep regime: delta ~ sqrt(g_bar)

        # SPARCRARScenario-specific fields
        data_path=data_path,
        split=split,
    )
