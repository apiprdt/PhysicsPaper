#!/usr/bin/env python3
"""
T2.3b: Aggregate multi-seed ADCD benchmark into mean +/- std with a
95% bootstrap confidence interval, and emit a LaTeX table for the paper.

Reads reproducibility_results.json (per-trial entries tagged with `seed`).
Writes:
    multi_seed_summary.json      - aggregate stats + per-seed rates + CI
    paper/generated/tab_multi_seed.tex  - LaTeX table for the paper / appendix

Usage:
    py -3.11 scripts/aggregate_seeds.py
    py -3.11 scripts/aggregate_seeds.py --n-bootstrap 10000 --seed 2026
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reproducibility_results.json"
SUMMARY_PATH = ROOT / "multi_seed_summary.json"
TEX_PATH = ROOT / "paper" / "generated" / "tab_multi_seed.tex"

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]


def _seed_rates(results):
    """Map seed -> overall structural recovery rate (fraction in [0,1])."""
    rates = {}
    for r in results:
        rates.setdefault(r["seed"], []).append(1.0 if r["class_match"] else 0.0)
    return {s: float(np.mean(v)) for s, v in rates.items()}


def _rate_at_noise(results, noise):
    sub = [1.0 if r["class_match"] else 0.0
           for r in results if abs(r["noise"] - noise) < 1e-9]
    return sub


def bootstrap_ci(seed_rates, n_bootstrap=10000, alpha=0.05, rng=None):
    """95% bootstrap CI of the mean recovery rate.

    Resamples the per-seed rates (one number per seed) with replacement,
    which respects the independence structure of the experiment (seeds are
    the independent units, not individual trials).
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    arr = np.asarray(seed_rates, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _fmt_pct(x):
    return f"{100*x:.1f}"


def generate_latex(summary):
    """Emit a LaTeX table: per-seed recovery + aggregate mean/CI."""
    per_seed = summary["per_seed_rates"]
    n_scenarios = summary["n_trials_per_seed"] // len(NOISE_LEVELS)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Structural recovery rate per random seed and aggregate statistics "
        rf"(Mock Proposer, {n_scenarios} scenarios $\times$ {len(NOISE_LEVELS)} noise levels "
        r"= 36 trials per seed). The 95\% confidence interval is a bootstrap CI "
        r"(10{,}000 resamples of the per-seed rates).}",
        r"\label{tab:multi_seed}",
        r"\small",
        r"\begin{tabular}{c r}",
        r"\toprule",
        r"\textbf{Seed} & \textbf{Recovery rate} \\",
        r"\midrule",
    ]
    for s in sorted(per_seed):
        rate = per_seed[s]
        lines.append(rf"{s} & {_fmt_pct(rate)}\% \\")
    mean = summary["overall_mean"]
    std = summary["overall_std"]
    ci_lo, ci_hi = summary["bootstrap_ci_95"]
    lines += [
        r"\midrule",
        rf"\textbf{{Mean $\pm$ std}} & {_fmt_pct(mean)}\% $\pm$ {_fmt_pct(std)}\% \\",
        rf"\textbf{{95\% bootstrap CI}} & [{_fmt_pct(ci_lo)}\%, {_fmt_pct(ci_hi)}\%] \\",
        rf"\textbf{{Min / Max}} & {_fmt_pct(summary['min'])}\% / {_fmt_pct(summary['max'])}\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed ADCD results with bootstrap CI."
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--results", type=str, default=str(RESULTS_PATH))
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run run_reproducibility.py first.",
              file=sys.stderr)
        return 1

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print(f"ERROR: {results_path} is empty.", file=sys.stderr)
        return 1

    seeds_present = sorted(set(r["seed"] for r in results))
    seed_rates = _seed_rates(results)
    rate_values = list(seed_rates.values())

    rng = np.random.default_rng(args.seed)
    ci_lo, ci_hi = bootstrap_ci(rate_values, n_bootstrap=args.n_bootstrap, rng=rng)

    # Per-noise breakdown (pooled over seeds).
    per_noise = {}
    for noise in NOISE_LEVELS:
        r = _rate_at_noise(results, noise)
        per_noise[noise] = {
            "n": len(r),
            "rate": float(np.mean(r)) if r else float("nan"),
            "std": float(np.std(r)) if r else float("nan"),
        }

    summary = {
        "n_seeds": len(seeds_present),
        "seeds": seeds_present,
        "n_trials_per_seed": len(results) // len(seeds_present),
        "per_seed_rates": seed_rates,
        "overall_mean": float(np.mean(rate_values)),
        "overall_std": float(np.std(rate_values, ddof=0)),
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "bootstrap_n_resamples": args.n_bootstrap,
        "min": float(np.min(rate_values)),
        "max": float(np.max(rate_values)),
        "median": float(np.median(rate_values)),
        "per_noise": per_noise,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {SUMMARY_PATH}")

    # Console report
    print("\n" + "=" * 60)
    print("   MULTI-SEED AGGREGATE")
    print("=" * 60)
    print(f"Seeds ({len(seeds_present)}): {seeds_present}")
    print("\nPer-seed recovery rate:")
    for s in sorted(seed_rates):
        print(f"  seed {s:>3}: {100*seed_rates[s]:.1f}%")
    print()
    print(f"Mean +/- std        : {100*summary['overall_mean']:.1f}% "
          f"+/- {100*summary['overall_std']:.1f}%")
    print(f"95% bootstrap CI    : [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%]  "
          f"(n_boot={args.n_bootstrap})")
    print(f"Min / median / max  : {100*summary['min']:.1f}% / "
          f"{100*summary['median']:.1f}% / {100*summary['max']:.1f}%")
    print("\nPer-noise breakdown (pooled):")
    for noise in NOISE_LEVELS:
        d = per_noise[noise]
        print(f"  noise {noise*100:>4.0f}%: {100*d['rate']:.1f}% "
              f"+/- {100*d['std']:.1f}%  (n={d['n']})")

    # LaTeX table
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(generate_latex(summary), encoding="utf-8")
    print(f"\nWrote {TEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
