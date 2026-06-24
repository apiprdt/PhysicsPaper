#!/usr/bin/env python3
"""
Generate the per-seed x per-noise structural recovery distribution for ADCD.

This is the anti-cherry-pick artifact: it documents the FULL distribution of
recovery rates across 16 random seeds and 4 noise levels, so that the paper /
README / PyPI page can headline the MEAN (80.4%) rather than the single best
seed (seed=42 = 94.4%).

Reads:   reproducibility_results.json   (per-trial entries tagged with `seed`)
Writes:  results/seed_distribution.json (full per-seed x per-noise breakdown)

Single-variable (SV) scenarios only. Multivariable (MV-*) trials are excluded
because they score class_match=False by construction and would deflate the
rate. This convention matches scripts/aggregate_seeds.py.

Usage:
    py -3.11 scripts/generate_seed_distribution.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reproducibility_results.json"
OUT_PATH = ROOT / "results" / "seed_distribution.json"
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]


def main() -> int:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run run_reproducibility.py first.",
              file=__import__("sys").stderr)
        return 1

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    # SV-only (exclude MV-*, matching aggregate_seeds.py).
    sv = [r for r in results if not r.get("scenario", "").startswith("MV-")]
    seeds = sorted({r["seed"] for r in sv})
    scenarios = sorted({r["scenario"] for r in sv})

    def seed_rates_all_noise():
        by = defaultdict(list)
        for r in sv:
            by[r["seed"]].append(1.0 if r["class_match"] else 0.0)
        return {s: float(np.mean(by[s])) for s in seeds}

    overall_rates = seed_rates_all_noise()
    rates_arr = np.array([overall_rates[s] for s in seeds])

    # Bootstrap 95% CI of the mean over per-seed rates (seeds are the
    # independent units, so we resample at the seed level).
    rng = np.random.default_rng(2026)
    n = len(rates_arr)
    idx = rng.integers(0, n, size=(10000, n))
    ci = np.quantile(rates_arr[idx].mean(axis=1), [0.025, 0.975])

    per_noise = {}
    for noise in NOISE_LEVELS:
        by = defaultdict(list)
        for r in sv:
            if abs(r["noise"] - noise) < 1e-9:
                by[r["seed"]].append(1.0 if r["class_match"] else 0.0)
        seed_pct = sorted(float(np.mean(by[s])) for s in by)
        s42 = by.get(42, [])
        per_noise[str(noise)] = {
            "mean": float(np.mean([np.mean(by[s]) for s in by])),
            "std_ddof0": float(np.std([np.mean(by[s]) for s in by], ddof=0)),
            "seed42": float(np.mean(s42)) if s42 else None,
            "per_seed_rates": [round(x, 4) for x in seed_pct],
        }

    out = {
        "_description": (
            "Per-seed x per-noise structural recovery distribution for ADCD "
            "(Mock Proposer, single-variable scenarios only). The headline "
            "number is the MEAN across 16 seeds, not seed=42. Multivariable "
            "(MV-*) trials are excluded (class_match=False by construction)."
        ),
        "method": "ADCD Mock Proposer",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "scenarios_sv": scenarios,
        "n_scenarios_sv": len(scenarios),
        "noise_levels": NOISE_LEVELS,
        "trials_per_seed_per_noise": len(scenarios),
        "trials_per_seed_total": len(scenarios) * len(NOISE_LEVELS),
        "overall": {
            "mean": float(rates_arr.mean()),
            "std_ddof0": float(rates_arr.std(ddof=0)),
            "std_ddof1": float(rates_arr.std(ddof=1)),
            "min": float(rates_arr.min()),
            "max": float(rates_arr.max()),
            "median": float(np.median(rates_arr)),
            "bootstrap_ci_95": [float(ci[0]), float(ci[1])],
            "seed42_rate": overall_rates.get(42),
            "note": (
                "seed=42 is the BEST of 16 seeds; the headline claim is the "
                "mean, not the peak."
            ),
        },
        "per_noise": per_noise,
        "per_seed_rates_overall": {str(s): round(overall_rates[s], 4) for s in seeds},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")

    # Console summary.
    print("\n=== ADCD recovery: mean +/- std across 16 seeds ===")
    print(f"Overall : {out['overall']['mean']*100:.1f}% +/- "
          f"{out['overall']['std_ddof0']*100:.1f}%   "
          f"(seed=42 = {out['overall']['seed42_rate']*100:.1f}%)")
    for noise in NOISE_LEVELS:
        d = per_noise[str(noise)]
        print(f"  {noise*100:>4.0f}% noise: {d['mean']*100:5.1f}% +/- "
              f"{d['std_ddof0']*100:4.1f}%   (seed=42 = "
              f"{d['seed42']*100 if d['seed42'] is not None else float('nan'):.1f}%)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
