import time
import logging
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from adcd.llm_proposer import BaseProposer, ProposalContext
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer, OptimizationResult
from adcd.anomaly_scenarios import AnomalyScenario
from adcd.metrics import evaluate_correction, CorrectionEvaluation, bic_score

logger = logging.getLogger(__name__)

# Correction terms bypass dimensional checking — the JAX optimizer handles
# numerical fitting regardless of the physical dimensions of the additive residual.
# The ARC asymptotic limit gate + complexity gate + transcendental arg gate are
# sufficient to filter physically implausible candidates.

@dataclass
class CorrectionIterationResult:
    iteration: int
    n_proposed: int
    n_survived_stage1: int
    n_optimized_stage2: int
    best_expr: str
    best_nmse_residual: float
    best_nmse_full: float
    top_5: List[Tuple[str, float]]  # List of (expr, nmse_residual)
    time_seconds: float

@dataclass
class CorrectionSearchResult:
    best_expr: str
    best_nmse_residual: float
    best_nmse_full: float
    best_theta: Dict[str, float]
    history: List[CorrectionIterationResult]
    total_candidates_proposed: int
    total_candidates_survived_stage1: int
    total_time_seconds: float
    converged: bool
    evaluation: Optional[CorrectionEvaluation] = None

class CorrectionOrchestrator:
    def __init__(
        self,
        proposer: BaseProposer,
        pipeline: Stage1Pipeline,
        optimizer: JAXOptimizer,
        max_iterations: int = 5,
        top_k: int = 10,
        convergence_nmse: float = 1e-5,
        verbose: bool = True,
    ):
        self.proposer = proposer
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.convergence_nmse = convergence_nmse
        self.verbose = verbose

        # Ensure "dimensionless" is in the pipeline's checker registry
        if "dimensionless" not in self.pipeline.checker.registry:
            self.pipeline.checker.registry["dimensionless"] = [0, 0, 0]
            self.pipeline.locals["dimensionless"] = sp.Symbol("dimensionless")

    def _register_scenario_symbols(self, scenario: AnomalyScenario):
        """Register all scenario variables and constants in the pipeline checker
        so that sympify and dimensional analysis don't crash on unknown symbols."""
        for var in scenario.classical_variables:
            if var not in self.pipeline.checker.registry:
                self.pipeline.checker.registry[var] = [0, 0, 0]  # treat as dimensionless
            if var not in self.pipeline.locals:
                self.pipeline.locals[var] = sp.Symbol(var)
        for const in scenario.classical_constants:
            if const not in self.pipeline.checker.registry:
                self.pipeline.checker.registry[const] = [0, 0, 0]
            if const not in self.pipeline.locals:
                self.pipeline.locals[const] = sp.Symbol(const)

    def _substitute_thetas(self, expr_str: str, val: float = 1.0) -> str:
        """Substitute free parameter symbols (theta_N) with a default value for Stage 1 screening."""
        try:
            expr = sp.sympify(expr_str, locals=self.pipeline.locals)
            subs_dict = {s: val for s in expr.free_symbols if str(s).startswith("theta_")}
            if subs_dict:
                return str(expr.subs(subs_dict))
        except Exception:
            pass
        return expr_str

    def search_correction(
        self,
        scenario: AnomalyScenario,
        noise_level: float = 0.0,
        seed: int = 42
    ) -> CorrectionSearchResult:
        start_time = time.time()
        
        # Register all scenario symbols in the pipeline to prevent unknown-symbol crashes
        self._register_scenario_symbols(scenario)
        
        # 1. Generate anomalous data and compute residual
        X, y_obs, y_classical, residual = scenario.generate_data(noise_level=noise_level, seed=seed)
        
        # 1b. Analyze residual for physical feature prior weights
        from adcd.residual_analyzer import analyze_residual
        primary_var_name = scenario.classical_limit_variable
        if primary_var_name in X:
            res_feat = analyze_residual(X[primary_var_name], residual)
        else:
            res_feat = None
        
        # 2. Compute statistics of residual data for ProposalContext
        data_statistics = {}
        for var in scenario.classical_variables:
            if var in X:
                arr = X[var]
                data_statistics[var] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
        
        if scenario.correction_type == "multiplicative":
            target_dim_key = "dimensionless"
        else:
            target_dim_key = scenario.classical_expr
            
        # Global search state tracking
        best_expr: str = ""
        best_nmse_residual: float = float("inf")
        best_bic: float = float("inf")
        best_theta: Dict[str, float] = {}
        stuck_count: int = 0
        prev_best_nmse_res: float = float("inf")
        
        total_candidates_proposed = 0
        total_candidates_survived_stage1 = 0
        
        history: List[CorrectionIterationResult] = []
        previous_best: List[Tuple[str, float]] = []

        # Enforce that classical limits are checked
        classical_limit_cond = f"{scenario.classical_limit_variable} -> {scenario.classical_limit_direction}"

        for iteration in range(self.max_iterations):
            iter_start_time = time.time()
            
            # Step 1: Build ProposalContext with rich physics metadata
            context = ProposalContext(
                variable_names=scenario.classical_variables,
                target_name="residual",
                data_statistics=data_statistics,
                n_candidates=25,
                iteration=iteration,
                stuck_count=stuck_count,
                domain=scenario.domain,
                classical_expr=scenario.classical_expr,
                variables_with_units=scenario.variables_with_units,
                anomaly_description=f"Observed in {scenario.anomaly_regime}. Mode: {scenario.correction_type}",
                known_limits=[{
                    "variable": scenario.classical_limit_variable,
                    "limit": scenario.classical_limit_direction,
                    "expected": "0"
                }],
                classical_limit_condition=classical_limit_cond,
                max_nodes=10,
                structural_hints=[],
                previous_best=previous_best if previous_best else None,
                constants=scenario.classical_constants,
                residual_features=res_feat
            )

            
            # Step 2: Propose candidates
            proposed_candidates = self.proposer.propose(context)
            n_proposed = len(proposed_candidates)
            total_candidates_proposed += n_proposed

            # Step 3: Stage 1 Screening
            subbed_candidates = []
            orig_by_subbed = {}
            for cand in proposed_candidates:
                sub_expr = self._substitute_thetas(cand, 1.0)
                has_params = (sub_expr != cand)
                subbed_candidates.append((sub_expr, has_params))
                if sub_expr not in orig_by_subbed:
                    orig_by_subbed[sub_expr] = []
                orig_by_subbed[sub_expr].append(cand)

            # Execute screening against residual data!
            stage1_results = self.pipeline.execute(
                subbed_candidates,
                target_dim_key,
                X,
                residual,
                constants=scenario.classical_constants
            )
            
            seen_sub_exprs = set()
            reconstructed_results = []
            for sub_expr, combined_score, mse, arc_score in stage1_results:
                if sub_expr in orig_by_subbed:
                    if sub_expr not in seen_sub_exprs:
                        seen_sub_exprs.add(sub_expr)
                        orig_cand = orig_by_subbed[sub_expr][0]
                        reconstructed_results.append((orig_cand, combined_score, mse, arc_score))

            n_survived = len(reconstructed_results)
            total_candidates_survived_stage1 += n_survived

            if n_survived == 0:
                stuck_count += 1
                logger.warning(f"[Iter {iteration}] No survivors from Stage 1. stuck_count={stuck_count}")
                iter_time = time.time() - iter_start_time
                history.append(
                    CorrectionIterationResult(
                        iteration=iteration,
                        n_proposed=n_proposed,
                        n_survived_stage1=0,
                        n_optimized_stage2=0,
                        best_expr=best_expr,
                        best_nmse_residual=best_nmse_residual,
                        best_nmse_full=float("inf"),
                        top_5=[],
                        time_seconds=iter_time
                    )
                )
                continue

            # Step 4: Stage 2 Optimization
            top_k_candidates = reconstructed_results[:self.top_k]
            
            # Substitute physical constants into candidates before JAX optimizes them
            subbed_top_k = []
            for cand, score, mse, arc in top_k_candidates:
                try:
                    expr = sp.sympify(cand, locals=self.pipeline.locals)
                    subs_dict = {sp.Symbol(k): v for k, v in scenario.classical_constants.items()}
                    if subs_dict:
                        subbed_cand = str(expr.subs(subs_dict))
                    else:
                        subbed_cand = cand
                except Exception:
                    subbed_cand = cand
                subbed_top_k.append((subbed_cand, score, mse, arc))

            stage2_results = self.optimizer.optimize_batch(
                subbed_top_k,
                X,
                residual,
                scenario.classical_variables
            )

            # Step 5: Update global best using BIC reranking
            stage2_results_with_bic = []
            if stage2_results:
                for expr_str, stage2_combined, opt_nmse, arc_score, opt_result in stage2_results:
                    n_params = len([k for k in opt_result.theta.keys() if k.startswith("theta_")])
                    n_points = len(residual)
                    b_score = bic_score(opt_nmse, n_params, n_points)
                    stage2_results_with_bic.append((expr_str, stage2_combined, opt_nmse, arc_score, opt_result, b_score))
                
                # Sort by BIC ascending
                stage2_results_with_bic = sorted(stage2_results_with_bic, key=lambda x: x[5])
                best_iter_cand = stage2_results_with_bic[0]
                iter_best_expr, _, iter_best_nmse, _, iter_opt_res, iter_best_bic = best_iter_cand

                if iter_best_bic < best_bic:
                    best_bic = iter_best_bic
                    best_nmse_residual = iter_best_nmse
                    best_expr = iter_best_expr
                    best_theta = iter_opt_res.theta

            # Step 6: Explore/Exploit stuck_count
            if best_nmse_residual < prev_best_nmse_res * 0.99:
                stuck_count = 0
            else:
                stuck_count += 1
            prev_best_nmse_res = best_nmse_residual

            # Feedback loop
            if stage2_results_with_bic:
                iter_feedback = [(r[0], r[5]) for r in stage2_results_with_bic if np.isfinite(r[2])]
                previous_best.extend(iter_feedback)
                previous_best = sorted(previous_best, key=lambda x: x[1])[:20]

            # Build full reconstruction NMSE
            if best_expr:
                temp_eval = evaluate_correction(best_expr, scenario, X, y_obs, y_classical, best_theta)
                best_nmse_full = temp_eval.nmse_full
            else:
                best_nmse_full = float("inf")

            iter_time = time.time() - iter_start_time
            top_5 = [(r[0], r[2]) for r in stage2_results_with_bic[:5]]
            
            iter_res = CorrectionIterationResult(
                iteration=iteration,
                n_proposed=n_proposed,
                n_survived_stage1=n_survived,
                n_optimized_stage2=len(stage2_results),
                best_expr=best_expr,
                best_nmse_residual=best_nmse_residual,
                best_nmse_full=best_nmse_full,
                top_5=top_5,
                time_seconds=iter_time
            )
            history.append(iter_res)

            if self.verbose:
                print(f"[Iter {iteration}/{self.max_iterations}] Proposed: {n_proposed} | Stage 1: {n_survived} | Residual NMSE: {best_nmse_residual:.6f} | BIC: {best_bic:.2f} | Full NMSE: {best_nmse_full:.6f} | stuck: {stuck_count}")

            # Early convergence check
            if best_nmse_residual < self.convergence_nmse:
                best_n_params = len([k for k in best_theta.keys() if k.startswith("theta_")])
                if best_n_params <= 1:
                    if self.verbose:
                        print(f"[CONVERGED] Converged at iteration {iteration} (Residual NMSE={best_nmse_residual:.2e} < {self.convergence_nmse:.2e} with simple model)")
                    break

        total_time = time.time() - start_time
        converged = best_nmse_residual < self.convergence_nmse

        # Compute final validation metrics
        final_evaluation = evaluate_correction(best_expr, scenario, X, y_obs, y_classical, best_theta)

        return CorrectionSearchResult(
            best_expr=best_expr,
            best_nmse_residual=best_nmse_residual,
            best_nmse_full=final_evaluation.nmse_full,
            best_theta=best_theta,
            history=history,
            total_candidates_proposed=total_candidates_proposed,
            total_candidates_survived_stage1=total_candidates_survived_stage1,
            total_time_seconds=total_time,
            converged=converged,
            evaluation=final_evaluation
        )


def main_cli():
    import argparse
    import sys
    import os
    from adcd.anomaly_scenarios import get_all_scenarios
    from adcd.llm_proposer import CorrectionMockProposer, CorrectionGeminiProposer, HybridCorrectionProposer
    from adcd.dimensional_checker import ASTValidator, DimensionalChecker
    from adcd.arc_scorer import ARCScorer
    from adcd.pipeline import Stage1Pipeline
    from adcd.jax_optimizer import JAXOptimizer

    parser = argparse.ArgumentParser(description="ADCD: Anomaly-Driven Correction Discovery CLI")
    parser.add_argument("--scenario", type=str, help="Name of the scenario to run")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level (default: 0.0)")
    parser.add_argument("--max-iter", type=int, default=5, help="Maximum number of search iterations (default: 5)")
    parser.add_argument("--proposer", type=str, choices=["mock", "gemini", "hybrid"], default="mock", help="Proposer type (default: mock)")
    parser.add_argument("--list", action="store_true", help="List all available scenarios and exit")
    parser.add_argument("--api-key", type=str, default=None, help="API key for LLM proposer (if using gemini or hybrid)")

    args = parser.parse_args()

    scenarios = get_all_scenarios()

    if args.list:
        print("Available Scenarios:")
        for idx, sc in enumerate(scenarios, 1):
            print(f"{idx}. {sc.name} ({sc.domain}, tier: {sc.tier})")
        sys.exit(0)

    if not args.scenario:
        print("Error: --scenario or --list must be provided.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Find the scenario
    selected_scenario = None
    for sc in scenarios:
        if sc.name.lower() == args.scenario.lower():
            selected_scenario = sc
            break

    if not selected_scenario:
        print(f"Error: Scenario '{args.scenario}' not found.", file=sys.stderr)
        print("Use --list to see all available scenarios.", file=sys.stderr)
        sys.exit(1)

    print(f"Running ADCD for Scenario: {selected_scenario.name}")
    print(f"Domain: {selected_scenario.domain}")
    print(f"Classical Expression: {selected_scenario.classical_expr}")
    print(f"Asymptotic Regime: {selected_scenario.classical_limit_variable} -> {selected_scenario.classical_limit_direction}")
    print(f"Noise Level: {args.noise}")
    print(f"Proposer: {args.proposer}")
    print(f"Max Iterations: {args.max_iter}\n")

    # Set up proposer
    if args.proposer == "mock":
        proposer = CorrectionMockProposer()
    elif args.proposer == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: API key must be provided via --api-key or GEMINI_API_KEY environment variable.", file=sys.stderr)
            sys.exit(1)
        proposer = CorrectionGeminiProposer(api_key=api_key)
    elif args.proposer == "hybrid":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: API key must be provided via --api-key or GEMINI_API_KEY environment variable.", file=sys.stderr)
            sys.exit(1)
        proposer = HybridCorrectionProposer(api_key=api_key)

    import sympy as sp
    from adcd.arc_scorer import ARCScorer, AsymptoticRegime

    # Set up pipeline & optimizer
    validator = ASTValidator()
    checker = DimensionalChecker()

    limit_var = sp.Symbol(selected_scenario.classical_limit_variable)
    limit_target = sp.oo if selected_scenario.classical_limit_direction == "oo" else 0
    regimes = [
        AsymptoticRegime(
            variable=limit_var,
            limit_target=limit_target,
            ground_truth_expr="0",
            weight=1.0
        )
    ]
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer()

    orchestrator = CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=args.max_iter,
        verbose=True
    )

    result = orchestrator.search_correction(selected_scenario, noise_level=args.noise)

    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Best Discovered Correction: {result.best_expr}")
    print(f"Residual NMSE: {result.best_nmse_residual:.6e}")
    print(f"Full Model NMSE: {result.best_nmse_full:.6e}")
    print(f"Optimized Parameters (theta):")
    if result.best_theta:
        for k, v in result.best_theta.items():
            print(f"  {k}: {v:.6f}")
    else:
        print("  None")
    print(f"Total time: {result.total_time_seconds:.2f} seconds")
    print(f"Converged: {result.converged}")
    print("="*40)


if __name__ == "__main__":
    main_cli()


