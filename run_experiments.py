import os
import time
import json
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any

from adcd.feynman_dataset import get_all_problems, FeynmanProblem
from adcd.llm_proposer import MockProposer, GeminiProposer, ProposalContext
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer
from adcd.orchestrator import SROrchestrator, SearchResult

def get_target_name(problem_name: str) -> str:
    mapping = {
        "Kinetic Energy": "E",
        "Gravitational Force": "F",
        "Coulomb's Law": "F",
        "Spring Potential": "E",
        "Simple Pendulum Period": "T",
        "Linear Motion": "x",
        "Gravitational Potential Energy": "E",
        "Relativistic Kinetic Energy": "E",
        "Ideal Gas Law": "P",
        "Projectile Range": "R",
        # New equations (Phase 2)
        "Ohm's Law": "V",
        "Power Dissipation": "P",
        "Capacitor Energy": "E",
        "Wave Speed": "v",
        "Doppler Effect": "f_obs",
        "Stefan-Boltzmann": "P_rad",
        "Wien's Law": "lam_max",
        "Lorentz Force": "F",
        "Buoyancy Force": "F",
        "Lens Equation": "f_lens",
    }
    return mapping.get(problem_name, "y")

def setup_orchestrator(problem: FeynmanProblem, proposer_type: str, api_key: Optional[str] = None) -> SROrchestrator:
    # 1. Build the Asymptotic regimes
    regimes = []
    for r_dict in problem.regimes:
        limit_val = sp.oo if r_dict["limit"] == "oo" else r_dict["limit"]
        regimes.append(
            AsymptoticRegime(
                variable=r_dict["variable"],
                limit_target=limit_val,
                ground_truth_expr=r_dict["expected"],
                weight=1.0
            )
        )
        
    validator = ASTValidator(max_depth=6, max_tokens=20)
    
    # Configure Dimensional Checker registry (inject theta variables dynamically as dimensionless)
    checker = DimensionalChecker()
    for i in range(20):
        checker.registry[f"theta_{i}"] = [0, 0, 0]
    # Inject problem variables and constants to avoid clashes with SymPy builtins (like N, E, etc.)
    for var in problem.variables:
        if problem.target_dimension is None:
            checker.registry[var] = [0, 0, 0]
        elif var not in checker.registry:
            checker.registry[var] = [0, 0, 0]
    for const in problem.constants:
        if problem.target_dimension is None:
            checker.registry[const] = [0, 0, 0]
        elif const not in checker.registry:
            checker.registry[const] = [0, 0, 0]
    checker.locals = {s: sp.Symbol(s) for s in checker.registry}
    
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer(n_restarts=5)
    
    if proposer_type == "gemini":
        if not api_key:
            raise ValueError("api_key must be provided for gemini proposer")
        # Use gemini-2.5-flash for cost-effective execution
        proposer = GeminiProposer(api_key=api_key, model_name="gemini-2.5-flash")
    else:
        proposer = MockProposer(seed=42)
        
    orchestrator = SROrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=5,  # 5 iterations max per problem
        top_k=5,
        convergence_nmse=1e-5,
        verbose=True
    )
    return orchestrator

def run_problem(problem: FeynmanProblem, proposer_type: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    print(f"\n" + "="*60)
    print(f" Running '{problem.name}' using {proposer_type.upper()}...")
    print("="*60)
    
    # Generate data
    X, y_obs = problem.generate_data(n_points=100, seed=42)
    
    orchestrator = setup_orchestrator(problem, proposer_type, api_key)
    target_name = get_target_name(problem.name)
    
    # Run the search
    res = orchestrator.search(
        variable_names=problem.variables,
        target_name=target_name,
        target_dimension=problem.target_dimension,
        X=X,
        y_obs=y_obs,
        data_vars=list(X.keys()),
        constants=problem.constants,
        seed=42,
        domain=problem.domain,
        classical_expr=problem.classical_expr,
        variables_with_units=problem.variables_with_units,
        anomaly_description=problem.anomaly_description,
        known_limits=problem.known_limits,
        classical_limit_condition=problem.classical_limit_condition,
        structural_hints=problem.structural_hints
    )
    
    return {
        "name": problem.name,
        "true_eq": problem.equation,
        "discovered_eq": res.best_expr,
        "discovered_theta": {k: float(v) for k, v in res.best_theta.items()},
        "nmse": float(res.best_nmse),
        "converged": bool(res.converged),
        "iterations": len(res.history),
        "total_proposed": res.total_candidates_proposed,
        "total_survived_stage1": res.total_candidates_survived_stage1,
        "time_seconds": float(res.total_time_seconds)
    }

def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "="*100)
    print(f"{'Problem Name':<30} | {'Status':<10} | {'Final NMSE':<12} | {'Iters':<6} | {'Time (s)':<8} | {'Discovered Equation'}")
    print("-"*100)
    for r in results:
        status = "CONVERGED" if r["converged"] else "FAILED"
        nmse_str = f"{r['nmse']:.2e}"
        time_str = f"{r['time_seconds']:.2f}"
        print(f"{r['name']:<30} | {status:<10} | {nmse_str:<12} | {r['iterations']:<6} | {time_str:<8} | {r['discovered_eq']}")
    print("="*100 + "\n")

def generate_markdown_report(results: List[Dict[str, Any]], proposer_type: str) -> str:
    success_count = sum(1 for r in results if r["converged"])
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    avg_time = np.mean([r["time_seconds"] for r in results])
    
    md = f"# Experiment Report: Closed-Loop Physics Discovery ({proposer_type.upper()})\n\n"
    md += f"- **Proposer Backend**: {proposer_type.upper()}\n"
    md += f"- **Success Rate**: {success_count}/{total_count} ({success_rate:.1f}%)\n"
    md += f"- **Average Discovery Time**: {avg_time:.2f} seconds per equation\n\n"
    
    md += "## Performance Benchmark Table\n\n"
    md += "| Problem Name | True Equation | Discovered Equation | Final NMSE | Status | Iterations | Time (s) |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in results:
        status = "✅ CONVERGED" if r["converged"] else "❌ FAILED"
        nmse_str = f"{r['nmse']:.2e}"
        time_str = f"{r['time_seconds']:.2f}"
        md += f"| {r['name']} | `{r['true_eq']}` | `{r['discovered_eq']}` | {nmse_str} | {status} | {r['iterations']} | {time_str} |\n"
        
    md += "\n## Detailed Discovered Parameters\n\n"
    for r in results:
        md += f"### {r['name']}\n"
        md += f"- **Discovered Equation Structure**: `{r['discovered_eq']}`\n"
        if r["discovered_theta"]:
            md += "- **Optimized Parameters (θ)**:\n"
            for k, v in r["discovered_theta"].items():
                md += f"  - `{k}`: {v:.6e}\n"
        else:
            md += "- **Optimized Parameters (θ)**: None\n"
        md += f"- **Final Normalized MSE**: {r['nmse']:.6e}\n"
        md += f"- **Time Taken**: {r['time_seconds']:.2f}s\n\n"
        
    return md

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    proposer_type = "gemini" if api_key else "mock"
    
    print(f"Starting experiments with Proposer Type: {proposer_type.upper()}")
    if proposer_type == "mock":
        print("Note: GEMINI_API_KEY environment variable not set. Running with MockProposer baseline.")
        
    problems = get_all_problems()
    results = []
    
    for problem in problems:
        try:
            res_dict = run_problem(problem, proposer_type, api_key)
            results.append(res_dict)
        except Exception as e:
            print(f"Error running {problem.name}: {e}")
            results.append({
                "name": problem.name,
                "true_eq": problem.equation,
                "discovered_eq": "ERROR",
                "discovered_theta": {},
                "nmse": float("inf"),
                "converged": False,
                "iterations": 0,
                "total_proposed": 0,
                "total_survived_stage1": 0,
                "time_seconds": 0.0
            })
            
    # Print console summary
    print_summary_table(results)
    
    # Save raw JSON results
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Generate and save Markdown report
    report_md = generate_markdown_report(results, proposer_type)
    with open("experiment_results.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"Saved raw JSON to experiment_results.json")
    print(f"Saved Markdown report to experiment_results.md")

if __name__ == "__main__":
    main()
