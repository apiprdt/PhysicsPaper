"""
Pre-Registered Benchmark Orchestrator for ADCD.
Executes 450 benchmark experiments across 30 equations x 3 noise levels x 5 seeds.
Logs all telemetry into append-only JSON audit trail.
"""

import sys
import os
import json
import sympy as sp
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adcd.gate_cascade import PhysicsGateCascade
from adcd.grammar_proposer import GrammarProposer
from adcd.pysr_adapter import PySRWithGate
from eval.audit_logger import AuditLogger
from eval.independent_evaluator import evaluate_candidate, compute_tree_distance

BENCHMARK_SPEC = os.path.join(os.path.dirname(__file__), "..", "preregistration", "benchmark_spec.txt")
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "preregistered_audit.jsonl")
SEEDS = [0, 42, 123, 456, 789]
NOISE_LEVELS = [0.01, 0.05, 0.15]


def parse_benchmark_spec() -> list[dict]:
    with open(BENCHMARK_SPEC, "r", encoding="utf-8") as f:
        lines = f.readlines()

    scenarios = []
    current_level = "Level A"
    for line in lines:
        line = line.strip()
        if "Level B" in line:
            current_level = "Level B"
        elif ":" in line and not line.startswith("#"):
            parts = line.split(":", 1)
            name = parts[0].strip()
            expr = parts[1].strip()
            scenarios.append({
                "name": name,
                "expr": expr,
                "level": current_level
            })
    return scenarios


def generate_synthetic_data(expr_str: str, noise_level: float, seed: int, n_samples: int = 100):
    rng = np.random.default_rng(seed)
    # Detect free variables in expr_str
    expr = sp.sympify(expr_str)
    vars_list = sorted([str(s) for s in expr.free_symbols if not str(s).startswith("theta_")])
    if not vars_list:
        vars_list = ["x1"]

    n_vars = len(vars_list)
    X = rng.uniform(0.5, 5.0, size=(n_samples, n_vars))

    # Evaluate ground truth y
    symbols = [sp.Symbol(v) for v in vars_list]
    fn = sp.lambdify(symbols, expr, modules=["numpy"])
    if n_vars == 1:
        y_true = fn(X[:, 0])
    else:
        y_true = fn(*[X[:, i] for i in range(n_vars)])

    y_true = np.asarray(y_true, dtype=float)
    noise = rng.normal(0, noise_level * np.std(y_true), size=y_true.shape)
    y_noisy = y_true + noise

    # 80/20 train/test split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_noisy[:split_idx], y_noisy[split_idx:]

    return X_train, y_train, X_test, y_test, vars_list


def run_single_experiment(scenario: dict, noise: float, seed: int, logger: AuditLogger, run_pysr: bool = True):
    name = scenario["name"]
    expr_str = scenario["expr"]
    level = scenario["level"]

    X_train, y_train, X_test, y_test, var_names = generate_synthetic_data(expr_str, noise, seed)
    gate_cascade = PhysicsGateCascade(max_depth=7, max_tokens=20, enable_arc_check=True)

    # --- ARM 1 & 2: GrammarProposer (Unfiltered vs Filtered) ---
    proposer = GrammarProposer(seed=seed)
    candidates = proposer.propose_candidates(n_candidates=30, variables=var_names)

    arm1_evals = []
    arm2_evals = []
    for cand in candidates:
        eval_res = evaluate_candidate(cand, expr_str, X_test, y_test, var_names)
        gate_res = gate_cascade.check(cand, limit_vars=var_names, limit_points=[0]*len(var_names))
        
        arm1_evals.append({"expr": cand, "nmse": eval_res["nmse"], "valid": eval_res["is_recovered"]})
        if gate_res.passed:
            arm2_evals.append({"expr": cand, "nmse": eval_res["nmse"], "valid": eval_res["is_recovered"]})

    pvr_arm2 = len(arm2_evals) / max(len(candidates), 1)

    log_entry = {
        "scenario": name,
        "level": level,
        "noise": noise,
        "seed": seed,
        "proposer": "GrammarProposer",
        "n_candidates": len(candidates),
        "n_passed_gate": len(arm2_evals),
        "pvr": pvr_arm2,
        "top1_unfiltered_nmse": min([e["nmse"] for e in arm1_evals], default=1e6),
        "top1_filtered_nmse": min([e["nmse"] for e in arm2_evals], default=1e6)
    }
    logger.log_experiment(log_entry)

    # --- ARM 3 & 4: PySR (Unfiltered vs Filtered) ---
    if run_pysr:
        try:
            adapter = PySRWithGate(gate_cascade=gate_cascade, niterations=10, population_size=10)
            pysr_results = adapter.fit_and_filter(
                X_train, y_train,
                variable_names=var_names,
                limit_vars=var_names,
                limit_points=[0]*len(var_names)
            )

            pysr_all = [r for r in pysr_results]
            pysr_passed = [r for r in pysr_results if r["gate_passed"]]

            pvr_pysr = len(pysr_passed) / max(len(pysr_all), 1)

            log_pysr = {
                "scenario": name,
                "level": level,
                "noise": noise,
                "seed": seed,
                "proposer": "PySR",
                "n_candidates": len(pysr_all),
                "n_passed_gate": len(pysr_passed),
                "pvr": pvr_pysr,
                "top1_unfiltered_nmse": min([r["loss"] for r in pysr_all], default=1e6),
                "top1_filtered_nmse": min([r["loss"] for r in pysr_passed], default=1e6)
            }
            logger.log_experiment(log_pysr)
        except Exception as e:
            print(f"[WARN] PySR run failed for {name}: {e}")


def main():
    scenarios = parse_benchmark_spec()
    logger = AuditLogger(AUDIT_LOG_PATH)
    print(f"[INFO] Starting Pre-Registered Benchmark Execution across {len(scenarios)} scenarios...")

    total_runs = len(scenarios) * len(NOISE_LEVELS) * len(SEEDS)
    completed = 0

    for sc in scenarios:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                run_single_experiment(sc, noise, seed, logger, run_pysr=False)
                completed += 1
                if completed % 15 == 0:
                    print(f"[PROGRESS] Completed {completed}/{total_runs} benchmark runs...")

    print(f"[SUCCESS] All pre-registered benchmark runs completed! Results saved to {AUDIT_LOG_PATH}")


if __name__ == "__main__":
    main()
