#!/usr/bin/env python3
"""
C3: Oracle ablation — inject ground-truth correction into proposer pool.

Tests whether gates + optimizer recover the true structural class when
the correct template family is guaranteed to appear in candidates.

Usage:
    py -3.11 run_oracle_ablation.py
"""

import json
import time
from typing import List

import sympy as sp

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.llm_proposer import CorrectionMockProposer, ProposalContext, BaseProposer
from run_ablation import build_orchestrator, ABLATION_NOISE


class OracleAugmentedProposer(BaseProposer):
    """Prepends ground-truth correction_expr to MockProposer output."""

    def __init__(self, oracle_expr: str, seed: int = 42):
        self.oracle_expr = oracle_expr
        self._inner = CorrectionMockProposer(seed=seed)

    def propose(self, context: ProposalContext) -> List[str]:
        cands = self._inner.propose(context)
        merged = [self.oracle_expr] + [c for c in cands if c != self.oracle_expr]
        return merged[: context.n_candidates]


def run_oracle_scenario(scenario, seed=42):
    orch = build_orchestrator(scenario, seed=seed)
    orch.proposer = OracleAugmentedProposer(scenario.correction_expr, seed=seed)

    t0 = time.time()
    search = orch.search_correction(scenario, noise_level=ABLATION_NOISE, seed=seed)
    elapsed = time.time() - t0
    ev = search.evaluation
    return {
        "scenario": scenario.name,
        "oracle_expr": scenario.correction_expr,
        "class_match": ev.class_match if ev else False,
        "discovered_expr": search.best_expr,
        "nmse_full": ev.nmse_full if ev else 1.0,
        "gate_survival_rate": (
            search.gate_stats.output_count / search.gate_stats.input_count
            if search.gate_stats and search.gate_stats.input_count
            else None
        ),
        "time_seconds": elapsed,
    }


def main():
    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]
    results = []
    print("=" * 70)
    print("ORACLE ABLATION @ 5% noise — ground truth injected into proposer")
    print("=" * 70)

    for sc in scenarios:
        res = run_oracle_scenario(sc)
        results.append(res)
        status = "OK" if res["class_match"] else "FAIL"
        print(f"  [{status}] {sc.name}: {res['discovered_expr'][:60]}")

    n_ok = sum(1 for r in results if r["class_match"])
    print(f"\nOracle recovery: {n_ok}/{len(results)} ({100*n_ok/len(results):.1f}%)")

    with open("oracle_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: oracle_ablation_results.json")


if __name__ == "__main__":
    main()
