"""
Reproducibility Study: multi-seed ADCD benchmark with bootstrap CI.

Runs the 9-scenario primary benchmark across multiple random seeds to
estimate the mean structural recovery rate with a 95% bootstrap confidence
interval. Designed to be resumable: existing entries in the output JSON are
reused, so new seeds can be appended without recomputing old ones.

Cara jalankan:
    py -3.11 run_reproducibility.py                 # default 20 seeds
    py -3.11 run_reproducibility.py --seeds 0 7 21 42 99
    py -3.11 run_reproducibility.py --seeds 2 5 13  --append  # add new seeds

Output:
    reproducibility_results.json   - per-trial entries (one per seed x scenario x noise)
    multi_seed_summary.json        - aggregate mean/std/CI + per-seed rates
    Console                        - mean +/- std + bootstrap CI table

Bootstrap CI + per-seed table are computed by scripts/aggregate_seeds.py,
which this script invokes automatically when --aggregate is set (default on).
"""

import argparse
import json
import logging
import sys

import numpy as np

# Reuse existing run_scenario_benchmark
from run_correction_discovery import run_scenario_benchmark
from adcd.anomaly_scenarios import get_all_scenarios

logging.basicConfig(level=logging.WARNING)

# Canonical seed set: original 5 (legacy) + 15 new for 20-seed CI.
SEEDS_ORIGINAL = [0, 7, 21, 42, 99]
SEEDS_NEW = [2, 5, 13, 17, 23, 31, 37, 41, 53, 61, 67, 71, 79, 83, 89]
SEEDS_ALL = sorted(SEEDS_ORIGINAL + SEEDS_NEW)

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
OUTPUT_PATH = "reproducibility_results.json"


def _load_existing(path: str) -> list:
    """Load existing per-trial results, if any."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        logging.warning("Existing %s is corrupt; starting fresh.", path)
    return []


def _completed_seeds(results: list) -> set:
    """Return the set of seeds that have a full 36-trial complement."""
    counts = {}
    for r in results:
        s = r.get("seed")
        if s is not None:
            counts[s] = counts.get(s, 0) + 1
    return {s for s, c in counts.items() if c >= 36}


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed reproducibility study with bootstrap CI."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=SEEDS_ALL,
        help="Random seeds to run (default: 20 canonical seeds).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Reuse existing entries in the output JSON and only compute missing "
        "seeds (default: True; use --no-append to force a fresh run).",
    )
    parser.add_argument(
        "--no-append",
        dest="append",
        action="store_false",
        help="Ignore any existing results and recompute every requested seed.",
    )
    parser.add_argument(
        "--no-aggregate",
        dest="aggregate",
        action="store_false",
        default=True,
        help="Skip automatic invocation of scripts/aggregate_seeds.py after the run.",
    )
    args = parser.parse_args()

    seeds = sorted(set(int(s) for s in args.seeds))
    n_seeds = len(seeds)

    print("=" * 70)
    print(f"   REPRODUCIBILITY STUDY: {n_seeds} seeds x 9 scenarios x 4 noise levels")
    print("=" * 70)

    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]

    # Resumable: load existing results and skip seeds already complete.
    all_results = _load_existing(OUTPUT_PATH) if args.append else []
    done = _completed_seeds(all_results)
    if args.append and done:
        print(f"Reusing {len(done)} already-completed seeds from {OUTPUT_PATH}: "
              f"{sorted(done)}")

    pending = [s for s in seeds if s not in done]
    if not pending:
        print("All requested seeds already present in output JSON. Skipping run.")
    else:
        print(f"Seeds to compute: {pending}")

    total = len(pending) * len(scenarios) * len(NOISE_LEVELS)
    count = 0

    for seed in pending:
        for scenario in scenarios:
            for noise in NOISE_LEVELS:
                count += 1
                res = run_scenario_benchmark(
                    scenario, noise, max_iter=4, proposer_type="mock", seed=seed
                )
                res["seed"] = seed
                all_results.append(res)

                status = "OK" if res["class_match"] else "FAIL"
                print(f"[{count}/{total}] seed={seed} {status} "
                      f"{scenario.name} noise={noise*100:.0f}% "
                      f"NMSE={res['nmse_full']:.2e}")

        # Checkpoint after each seed so partial progress is never lost.
        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  -> checkpoint: {OUTPUT_PATH} "
              f"(now {len(all_results)} entries, seed {seed} complete)")

    # Keep only the requested seeds in the final file (drop legacy extras
    # that are no longer requested, e.g. when running a subset).
    requested = set(seeds)
    all_results = [r for r in all_results if r.get("seed") in requested]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH} ({len(all_results)} entries)")

    # Per-seed summary
    print("\n" + "=" * 70)
    print("   RINGKASAN STATISTIK (mean +/- std)")
    print("=" * 70)

    for noise in NOISE_LEVELS:
        subset = [r for r in all_results if abs(r["noise"] - noise) < 1e-9]
        rates = [1.0 if r["class_match"] else 0.0 for r in subset]
        nmses = [r["nmse_full"] for r in subset]
        print(f"Noise {noise*100:>4.0f}%: "
              f"ClassMatch = {np.mean(rates)*100:.1f}% +/- {np.std(rates)*100:.1f}%  |  "
              f"NMSE = {np.mean(nmses):.2e} +/- {np.std(nmses):.2e}")

    all_rates = [1.0 if r["class_match"] else 0.0 for r in all_results]
    print(f"\nOVERALL: {np.mean(all_rates)*100:.1f}% +/- {np.std(all_rates)*100:.1f}%")

    # Per-seed breakdown (full recovery rate per seed).
    print("\nPer-seed structural recovery rate:")
    for s in sorted(set(r["seed"] for r in all_results)):
        sub = [r for r in all_results if r["seed"] == s]
        m = sum(1 for r in sub if r["class_match"])
        print(f"  seed {s:>3}: {m}/{len(sub)} = {100*m/len(sub):.1f}%")

    # Bootstrap CI aggregation (delegated to a dedicated, importable script).
    if args.aggregate:
        from pathlib import Path
        agg_script = Path(__file__).resolve().parent / "scripts" / "aggregate_seeds.py"
        if agg_script.exists():
            print(f"\nAggregating bootstrap CI via {agg_script} ...")
            import subprocess
            try:
                subprocess.run(
                    [sys.executable, str(agg_script)], check=False
                )
            except Exception as e:  # pragma: no cover - defensive
                print(f"  (aggregation skipped: {e})")
        else:
            print("(scripts/aggregate_seeds.py not found; skipping aggregation.)")


if __name__ == "__main__":
    main()
