"""
Run all ADCD physics experiments with timing and JSON outputs.

Estimated wall time (Intel i5-class laptop, CPU JAX):
  Muon g-2 validation:  ~30–90 s
  SPARC discovery:      ~20–60 s (simulated) / ~60–120 s (real stack)
  Full suite:           ~2–4 min

Usage:
  python -m adcd.experiments.run_all
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from adcd.experiments.muon_g2_validation import (
    run_validation_demo,
    validate_iadcd_on_qed,
    validate_integrated_iadcd,
)
from adcd.experiments.sparc_stacking import run_sparc_discovery, save_sparc_json


@dataclass
class ExperimentTiming:
    name: str
    seconds: float
    success: bool
    notes: str = ""


def _timed(name: str, fn) -> tuple[Any, ExperimentTiming]:
    t0 = time.perf_counter()
    ok, notes = True, ""
    try:
        result = fn()
        if isinstance(result, bool):
            ok = result
            result = {"passed": result}
    except Exception as exc:
        ok = False
        notes = str(exc)
        result = {"error": str(exc)}
    elapsed = time.perf_counter() - t0
    return result, ExperimentTiming(name=name, seconds=elapsed, success=ok, notes=notes)


def run_all(output_dir: str = "results") -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: list[ExperimentTiming] = []
    payload: Dict[str, Any] = {"experiments": {}, "timings": []}

    print("=" * 60)
    print("ADCD Experiment Suite")
    print("Estimated total: 2–4 minutes on CPU")
    print("=" * 60)

    # --- Muon g-2 ---
    print("\n[1/3] Muon g-2 QED validation (SYNTHETIC)...")
    muon_result, t1 = _timed("muon_g2", lambda: validate_iadcd_on_qed(max_order=2, verbose=False))
    from adcd.experiments.muon_g2_validation import print_validation_report, save_validation_json
    print_validation_report(muon_result)
    save_validation_json(muon_result, str(out / "muon_g2_validation.json"))
    timings.append(t1)
    payload["experiments"]["muon_g2"] = asdict(muon_result)

    # --- Muon Tier C with OLS projection ---
    print("\n[2/3] Muon g-2 integrated + OLS projection (Tier C+)...")
    def _tier_c_ols():
        from adcd.experiments.proposers import perturbative_order_proposer
        from adcd.experiments.muon_g2_validation import (
            VARIABLE, QED_COEFFICIENTS, TOLERANCES, OrderRecovery, generate_muon_g2_data, _eval_pass,
        )
        from adcd.iadcd_orchestrator import iADCDOrchestrator

        X, y_obs, y_classical = generate_muon_g2_data(150, 0.0, 42, max_order=2)
        x = X[VARIABLE]
        round_proposers = [
            perturbative_order_proposer(k, VARIABLE, 42 + k) for k in (1, 2)
        ]
        orch = iADCDOrchestrator(
            max_rounds=2, convergence_nmse=1e-6, min_snr=0.01,
            verbose=False, subtraction_mode="ols_projection", projection_variable=VARIABLE,
        )
        res = orch.run_iterative_discovery(
            X=X, y_obs=y_obs, y_classical=y_classical,
            limit_variable=VARIABLE, limit_direction="0", classical_expr="0.0",
            variables_with_units={VARIABLE: "dimensionless"},
            round_proposers=round_proposers, seed=42,
        )
        recoveries = []
        for rnd in res.rounds:
            coeff = rnd.discovered_theta.get(f"theta_r{rnd.round_idx}_0")
            rel = abs(coeff - QED_COEFFICIENTS[rnd.round_idx]) / QED_COEFFICIENTS[rnd.round_idx] if coeff else None
            recoveries.append({
                "order": rnd.round_idx,
                "coefficient": coeff,
                "known": QED_COEFFICIENTS[rnd.round_idx],
                "error_pct": rel * 100 if rel is not None else None,
                "passed": _eval_pass(rnd.round_idx, rel),
            })
        return {
            "final_nmse": res.final_nmse_full,
            "final_expr": res.final_expr,
            "recoveries": recoveries,
            "all_passed": all(r["passed"] for r in recoveries),
        }

    tier_c, t2 = _timed("muon_g2_ols", _tier_c_ols)
    timings.append(t2)
    payload["experiments"]["muon_g2_ols_projection"] = tier_c
    if isinstance(tier_c, dict) and "recoveries" in tier_c:
        print(f"  Tier C+ OLS: all_passed={tier_c.get('all_passed')}  NMSE={tier_c.get('final_nmse', 0):.2e}")

    # --- SPARC ---
    print("\n[3/3] SPARC MOND discovery...")
    sparc_result, t3 = _timed("sparc", lambda: run_sparc_discovery(verbose=False))
    save_sparc_json(sparc_result, str(out / "sparc_discovery.json"))
    timings.append(t3)
    payload["experiments"]["sparc"] = asdict(sparc_result)

    payload["timings"] = [asdict(t) for t in timings]
    payload["total_seconds"] = sum(t.seconds for t in timings)
    (out / "experiment_suite.json").write_text(
        json.dumps(payload, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else (
            float(o) if isinstance(o, np.floating) else o
        )),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    for t in timings:
        status = "OK" if t.success else "FAIL"
        print(f"  {t.name:<20} {t.seconds:6.1f}s  [{status}]  {t.notes}")
    print(f"  {'TOTAL':<20} {payload['total_seconds']:6.1f}s")
    print(f"\nResults: {out.resolve()}")
    print("=" * 60)
    return payload


if __name__ == "__main__":
    run_all()
