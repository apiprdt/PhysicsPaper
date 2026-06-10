#!/usr/bin/env python3
"""
LLM zero-shot / hybrid benchmark (separate from mock template-assisted results).

Requires GEMINI_API_KEY for hybrid/gemini proposers. Writes JSON to
results/llm_benchmark.json for README / paper tables.

Usage:
    python run_llm_benchmark.py --proposer hybrid
    python run_llm_benchmark.py --proposer gemini --scenarios "Relativistic KE"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import adcd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADCD LLM proposer benchmark")
    parser.add_argument(
        "--proposer",
        choices=("hybrid", "gemini"),
        default="hybrid",
        help="LLM proposer backend (not mock)",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenario names (default: first 3 textbook scenarios)",
    )
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/llm_benchmark.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "GEMINI_API_KEY not set. LLM benchmark skipped.\n"
            "Export the key and re-run, or use run_correction_discovery.py --proposer mock "
            "for template-assisted baseline.",
            file=sys.stderr,
        )
        return 1

    all_scenarios = adcd.get_all_scenarios()
    if args.scenarios.strip():
        names = {n.strip() for n in args.scenarios.split(",")}
        selected = [s for s in all_scenarios if s.name in names]
        missing = names - {s.name for s in selected}
        if missing:
            print(f"Unknown scenarios: {sorted(missing)}", file=sys.stderr)
            return 1
    else:
        selected = [s for s in all_scenarios if s.tier == "Textbook"][:3]

    rows = []
    for scenario in selected:
        print(f"\n=== {scenario.name} ({args.proposer}) ===")
        result = adcd.discover_correction(
            scenario,
            noise_level=args.noise,
            max_iterations=args.max_iter,
            proposer=args.proposer,
            correction_mode="auto",
            seed=args.seed,
            verbose=True,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "tier": scenario.tier,
                "proposer": args.proposer,
                "noise": args.noise,
                "best_expr": result.best_expr,
                "best_nmse_residual": result.best_nmse_residual,
                "correction_mode_detected": result.scenario.correction_type,
                "class_match": getattr(result.search_result, "class_match", None),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proposer": args.proposer,
        "note": "Zero-shot / hybrid LLM results — do not mix with mock template-assisted tables.",
        "results": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
