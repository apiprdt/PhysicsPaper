"""
SPARC stacked rotation-curve experiment — MOND interpolating function ν(x).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adcd.experiments.mond_comparison import print_mond_comparison, score_mond_models, scores_to_dict
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
    mond_comparison: List[Dict[str, Any]] = field(default_factory=list)
    identifiability: Optional[Dict[str, Any]] = None


SIMPLE_MOND_NU = "(1 + sqrt(1 + 4/x)) / 2"


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

    if verbose:
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

    adcd_nmse = float(res.final_nmse_full)
    n_params = len(res.rounds[0].discovered_theta) if res.rounds else 2
    mond_scores = score_mond_models(stack.x, stack.nu_obs, adcd_nmse=adcd_nmse, adcd_n_params=n_params)

    ident: Optional[Dict[str, Any]] = None
    if res.rounds and res.rounds[0].adcd_result.search_result.identifiability_report:
        rep = res.rounds[0].adcd_result.search_result.identifiability_report
        ident = {
            "is_identifiable": rep.is_identifiable,
            "failure_mode": rep.failure_mode,
            "snr": rep.snr,
            "weight_ratio": rep.weight_ratio,
            "summary": rep.summary,
        }

    result = SparcDiscoveryResult(
        data_source=stack.data_source,
        n_galaxies=stack.n_galaxies,
        n_points=stack.n_points,
        discovered_expr=res.final_expr,
        final_nmse=adcd_nmse,
        g_dagger=G_DAGGER,
        simple_mond_reference=SIMPLE_MOND_NU,
        mond_comparison=scores_to_dict(mond_scores),
        identifiability=ident,
    )

    print("\n=== SPARC MOND Discovery Results ===")
    print(f"Data source:     {result.data_source}")
    print(f"Galaxies/points: {result.n_galaxies} / {result.n_points}")
    print(f"Discovered nu(x): {result.discovered_expr}")
    print(f"ADCD NMSE:       {result.final_nmse:.5f}")
    print_mond_comparison(mond_scores)
    if ident:
        print(f"\nIdentifiability: {ident['summary']}")
    if result.data_source == "SIMULATED":
        print("\nWARNING: Simulated benchmark — do NOT claim real SPARC discovery.")

    return result


def save_sparc_json(result: SparcDiscoveryResult, path: str = "results/sparc_discovery.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    if payload.get("identifiability") and payload["identifiability"].get("weight_ratio") == float("inf"):
        payload["identifiability"]["weight_ratio"] = None
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run_sparc_discovery()
    save_sparc_json(r)
