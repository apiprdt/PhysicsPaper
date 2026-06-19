"""
SPARC stacked rotation-curve experiment — MOND interpolating function ν(x).

Labeling rules (non-negotiable):
  REAL    — parsed MassModels_Lelli2016c.mrt, ≥100 stacked points
  SIMULATED — synthetic Simple-MOND benchmark when download/parse fails
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from adcd.experiments.proposers import mond_correction_proposer
from adcd.experiments.sparc_data import G_DAGGER, SparcStackResult, load_sparc_stack
from adcd.iadcd_orchestrator import iADCDOrchestrator


@dataclass
class SparcDiscoveryResult:
    data_source: str
    n_galaxies: int
    n_points: int
    discovered_expr: str
    final_nmse: float
    g_dagger: float
    simple_mond_reference: str


SIMPLE_MOND_NU = "(1 + sqrt(1 + 4/x)) / 2"


def compare_to_simple_mond(x: np.ndarray, nu_obs: np.ndarray) -> float:
    """NMSE of Simple MOND vs stacked observations (reference baseline)."""
    nu_mond = (1.0 + np.sqrt(1.0 + 4.0 / x)) / 2.0
    return float(np.mean((nu_obs - nu_mond) ** 2) / max(np.var(nu_obs), 1e-15))


def run_sparc_discovery(
    cache_path: str = "data/sparc/MassModels_Lelli2016c.mrt",
    seed: int = 42,
    verbose: bool = True,
) -> SparcDiscoveryResult:
    stack: SparcStackResult = load_sparc_stack(cache_path=cache_path)

    X = {"x": stack.x}
    y_obs = stack.nu_obs
    y_classical = stack.nu_classical

    orchestrator = iADCDOrchestrator(
        max_rounds=1,
        convergence_nmse=1e-3,
        min_snr=0.5,
        verbose=verbose,
    )

    print("\nRunning iADCD on stacked (x, nu) dataset...")
    res = orchestrator.run_iterative_discovery(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable="x",
        limit_direction="oo",
        classical_expr="1.0",
        variables_with_units={"x": "dimensionless"},
        round_proposers=[mond_correction_proposer(variable="x", seed=seed)],
        seed=seed,
    )

    ref_nmse = compare_to_simple_mond(stack.x, stack.nu_obs)

    result = SparcDiscoveryResult(
        data_source=stack.data_source,
        n_galaxies=stack.n_galaxies,
        n_points=stack.n_points,
        discovered_expr=res.final_expr,
        final_nmse=float(res.final_nmse_full),
        g_dagger=G_DAGGER,
        simple_mond_reference=SIMPLE_MOND_NU,
    )

    print("\n=== SPARC MOND Discovery Results ===")
    print(f"Data source:     {result.data_source}")
    print(f"Galaxies/points: {result.n_galaxies} / {result.n_points}")
    print(f"Discovered ν(x): {result.discovered_expr}")
    print(f"ADCD NMSE:       {result.final_nmse:.5f}")
    print(f"Simple MOND NMSE (reference): {ref_nmse:.5f}")
    if result.data_source == "SIMULATED":
        print("WARNING: Simulated benchmark — do NOT claim real SPARC discovery.")

    return result


def save_sparc_json(result: SparcDiscoveryResult, path: str = "results/sparc_discovery.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run_sparc_discovery()
    save_sparc_json(r)
