"""
Ablation Study: Ukur kontribusi setiap Physics Gate di Stage 1 Pipeline.

Gates yang di-ablate:
  A. Full ADCD         — baseline (semua gates aktif)
  B. No ARC Gate       — matikan asymptotic consistency check (Gate 3)
  C. No AST Gate       — matikan complexity gate (Gate 1)
  D. No Dim Gate       — matikan dimensional suitability check (Gate 2)
  E. No Data Gate      — matikan coarse empirical evaluation (Gate 4)
  F. No Gates          — hanya JAX optimizer, tanpa screening apapun

Cara jalankan:
    py -3.11 run_ablation.py

Output:
    ablation_results.json  — raw results
    Console                — tabel kontribusi setiap gate
"""

import json
import time
import logging
import numpy as np
import sympy as sp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.anomaly_scenarios import get_all_scenarios, AnomalyScenario
from src.metrics import evaluate_correction, CorrectionEvaluation, classify_structure, bic_score
from src.llm_proposer import CorrectionMockProposer
from src.correction_orchestrator import CorrectionOrchestrator
from src.dimensional_checker import ASTValidator, DimensionalChecker
from src.arc_scorer import ARCScorer, AsymptoticRegime
from src.pipeline import Stage1Pipeline
from src.jax_optimizer import JAXOptimizer

logging.basicConfig(level=logging.WARNING)

# Hanya jalankan ablation pada 5% noise (cukup untuk mendeteksi perbedaan)
ABLATION_NOISE = 0.05

# Kondisi ablation
ABLATION_CONDITIONS = {
    "Full_ADCD":   {"use_arc": True,  "use_ast": True,  "use_dim": True,  "use_data": True},
    "No_ARC":      {"use_arc": False, "use_ast": True,  "use_dim": True,  "use_data": True},
    "No_AST":      {"use_arc": True,  "use_ast": False, "use_dim": True,  "use_data": True},
    "No_Dim":      {"use_arc": True,  "use_ast": True,  "use_dim": False, "use_data": True},
    "No_DataGate": {"use_arc": True,  "use_ast": True,  "use_dim": True,  "use_data": False},
    "No_Gates":    {"use_arc": False, "use_ast": False, "use_dim": False, "use_data": False},
}


class AblatedPipeline(Stage1Pipeline):
    """Pipeline dengan kemampuan mematikan gate secara individual untuk ablation."""

    def __init__(self, validator, checker, scorer,
                 use_arc=True, use_ast=True, use_dim=True, use_data=True):
        super().__init__(validator, checker, scorer)
        self.use_arc = use_arc
        self.use_ast = use_ast
        self.use_dim = use_dim
        self.use_data = use_data

    def execute(self, candidates, target_dimension_key, X=None, y_obs=None,
                beta=1.0, constants=None):
        """Override execute dengan kontrol ablation per gate."""
        from src.coarse_evaluator import CoarseEvaluator
        import sympy as sp
        import numpy as np
        from src.dimensional_checker import validate_transcendental_args

        screened_candidates = []
        evaluator = None
        if self.use_data and X is not None and y_obs is not None:
            evaluator = CoarseEvaluator(X, y_obs, constants=constants)

        for item in candidates:
            if isinstance(item, tuple):
                raw_cand, has_params = item
            else:
                raw_cand = item
                has_params = False

            try:
                expr = sp.sympify(raw_cand, locals=self.locals)
            except Exception:
                continue

            # Gate 1: AST Complexity
            if self.use_ast and not self.validator.verify(expr):
                continue

            # Gate 2: Dimensional Suitability Check
            if self.use_dim:
                is_dim_ok = True
                if target_dimension_key is not None and target_dimension_key != "dimensionless":
                    if has_params:
                        try:
                            self.checker._get_dim_vector(expr)
                            is_dim_ok = True
                        except TypeError:
                            is_dim_ok = False
                    else:
                        is_dim_ok = self.checker.verify(expr, target_dimension_key)
                else:
                    is_dim_ok = self.checker.verify(expr, target_dimension_key)
                
                if not is_dim_ok:
                    continue
                
                # Gate 2.5: Transcendental Argument Guardrail
                if not validate_transcendental_args(expr, self.checker):
                    continue

            # Gate 3: ARC Asymptotic
            arc_score = 1.0  # default jika di-disable
            if self.use_arc:
                try:
                    arc_score = float(self.scorer.score(expr, constants=constants))
                except Exception:
                    continue
                if arc_score <= 0.0:
                    continue

            # Gate 4: Data-driven coarse eval
            mse = 0.0
            nmse = 0.0
            if self.use_data and evaluator is not None:
                mse, nmse = evaluator.evaluate(expr, has_params=has_params)
                if np.isinf(mse):
                    continue

            combined_score = arc_score * float(np.exp(-beta * nmse))
            screened_candidates.append((raw_cand, combined_score, mse, arc_score))

        return sorted(screened_candidates, key=lambda x: x[1], reverse=True)


def build_orchestrator(scenario: AnomalyScenario, use_arc=True, use_ast=True,
                        use_dim=True, use_data=True, seed=42) -> CorrectionOrchestrator:
    """Build orchestrator dengan gate configuration yang ditentukan."""
    validator = ASTValidator()
    checker = DimensionalChecker()
    checker.registry["dimensionless"] = [0, 0, 0]

    # Populate registry variables from scenario units
    for var, unit in scenario.variables_with_units.items():
        if unit == "kg":
            checker.registry[var] = [1, 0, 0]
        elif unit == "m":
            checker.registry[var] = [0, 1, 0]
        elif unit == "s":
            checker.registry[var] = [0, 0, 1]
        elif unit == "m/s":
            checker.registry[var] = [0, 1, -1]
        elif unit == "N/m":
            checker.registry[var] = [1, 0, -2]
        elif unit == "C":
            checker.registry[var] = [0, 0, 0] # electrostatic simplified
        elif unit == "K":
            checker.registry[var] = [0, 0, 0] # dimensionless temperature
        elif unit == "mol":
            checker.registry[var] = [0, 0, 0]
        elif unit == "m^3":
            checker.registry[var] = [0, 3, 0]
        elif unit == "Hz":
            checker.registry[var] = [0, 0, -1]
        else:
            checker.registry[var] = [0, 0, 0]

    # Constants mapping
    for const in scenario.classical_constants:
        if const == "c":
            checker.registry[const] = [0, 1, -1]
        elif const == "G":
            checker.registry[const] = [-1, 3, -2]
        elif const == "k_e":
            checker.registry[const] = [1, 3, -2] # units of N*m^2/C^2
        elif const == "sigma":
            checker.registry[const] = [1, 0, -3] # units of W/(m^2*K^4) -> kg/s^3
        else:
            checker.registry[const] = [0, 0, 0]

    limit_var = sp.Symbol(scenario.classical_limit_variable)
    limit_target = sp.oo if scenario.classical_limit_direction == "oo" else 0
    regimes = [AsymptoticRegime(
        variable=limit_var,
        limit_target=limit_target,
        ground_truth_expr="0",
        weight=1.0
    )]
    scorer = ARCScorer(regimes=regimes)

    pipeline = AblatedPipeline(
        validator, checker, scorer,
        use_arc=use_arc, use_ast=use_ast,
        use_dim=use_dim, use_data=use_data
    )
    optimizer = JAXOptimizer()
    proposer = CorrectionMockProposer(seed=seed)

    return CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=4,
        top_k=8,
        convergence_nmse=1e-5,
        verbose=False
    )


def run_ablation_scenario(scenario: AnomalyScenario, condition_name: str,
                           gate_config: dict, noise: float = ABLATION_NOISE) -> dict:
    """Jalankan satu skenario dengan satu konfigurasi ablation."""
    orchestrator = build_orchestrator(scenario, **gate_config)

    X, y_obs, y_classical, _ = scenario.generate_data(noise_level=noise, seed=42)
    t0 = time.time()
    search_res = orchestrator.search_correction(scenario, noise_level=noise, seed=42)
    elapsed = time.time() - t0

    eval_res = search_res.evaluation
    if eval_res is None:
        return {
            "scenario": scenario.name,
            "condition": condition_name,
            "class_match": False,
            "nmse_full": 1.0,
            "time_seconds": elapsed,
        }

    return {
        "scenario": scenario.name,
        "condition": condition_name,
        "noise": noise,
        "class_match": eval_res.class_match,
        "nmse_full": eval_res.nmse_full,
        "discovered_class": eval_res.discovered_class,
        "time_seconds": elapsed,
    }


def main():
    print("=" * 70)
    print("     ABLATION STUDY: Kontribusi Physics Gates")
    print(f"     Noise level fixed: {ABLATION_NOISE*100:.0f}%")
    print("=" * 70)

    # Hanya 9 skenario utama
    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]
    all_results = []
    total = len(ABLATION_CONDITIONS) * len(scenarios)
    count = 0

    for condition_name, gate_config in ABLATION_CONDITIONS.items():
        print(f"\n--- Condition: {condition_name} ---")
        for scenario in scenarios:
            count += 1
            res = run_ablation_scenario(scenario, condition_name, gate_config)
            all_results.append(res)
            status = "OK" if res["class_match"] else "FAIL"
            print(f"  [{count}/{total}] {status} {scenario.name}")

    with open("ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: ablation_results.json ({len(all_results)} entries)")

    # Summary table
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Condition':<20} | {'Match':>8} | {'Rate':>8} | {'Delta vs Full':>14}")
    print("-" * 60)

    full_match = sum(1 for r in all_results
                     if r["condition"] == "Full_ADCD" and r["class_match"])

    for condition_name in ABLATION_CONDITIONS.keys():
        subset = [r for r in all_results if r["condition"] == condition_name]
        n_match = sum(1 for r in subset if r["class_match"])
        rate = n_match / len(subset) * 100
        delta = n_match - full_match
        sign = "+" if delta >= 0 else ""
        print(f"{condition_name:<20} | {n_match:>4}/{len(subset):<3} | {rate:>6.1f}% | {sign}{delta:>+4}")


if __name__ == "__main__":
    main()
