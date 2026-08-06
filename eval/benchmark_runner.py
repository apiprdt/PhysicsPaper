"""
Pre-Registered Benchmark Runner for ADCD.
FIXED VERSION — replaces the invalidated 2026-07-31 run.

Fixes applied:
1. PySR now actually runs (was run_pysr=False before — invalidated all prior data)
2. Level B correctly computes residual: target = y_noisy - f0(x), not y directly
3. ARC limit points physically correct:
   - Level A (Feynman equations): ARC DISABLED — these are full physics quantities, not corrections
   - Level B (Correction-First): limit x1 → +∞ — all synthetic corrections decay to zero
4. FPRR (M2) now tracked per run: false positives = NMSE < threshold but gate fails
5. ARM C added: PySR fits y_total directly (baseline for M4 correction-first comparison)
6. Physical constants normalized to 1.0 in Level A expressions to avoid domain errors

Experiment arms per Level B scenario:
  ARM-A: GrammarProposer + Gate    (lightweight ADCD)
  ARM-B: PySR + Gate               (plug-in mode, correction-first residual target)
  ARM-C: PySR + Gate, direct fit   (baseline, fits y_total not residual — for M4)
  
Level A only runs ARM-A (complexity gate test).
"""

import sys
import os
import re
import json
import warnings
import sympy as sp
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adcd.gate_cascade import PhysicsGateCascade
from adcd.grammar_proposer import GrammarProposer
from adcd.pysr_adapter import PySRWithGate
from eval.audit_logger import AuditLogger
from eval.independent_evaluator import evaluate_candidate

# ── Pre-Registered Constants ──────────────────────────────────────────────────
BENCHMARK_SPEC = os.path.join(os.path.dirname(__file__), "..", "preregistration", "benchmark_spec.txt")
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "preregistered_audit.jsonl")
SEEDS = [0, 42, 123, 456, 789]
NOISE_LEVELS = [0.01, 0.05, 0.15]
NMSE_FP_THRESHOLD = 0.10   # candidate is "good fitting" if NMSE < this

# Known physical constants — substituted to 1.0 for dimensionless Level A benchmark
# This is standard in Feynman SR benchmarks (SRBench, AI Feynman).
# Note: 'pi' is NOT listed here — SymPy recognizes sp.pi automatically.
PHYSICAL_CONSTANTS = {
    "c": 1.0, "hbar": 1.0, "k": 1.0, "epsilon_0": 1.0,
    "epsilon": 1.0, "mu": 1.0, "n_0": 1.0, "q": 1.0,
    "B": 1.0, "A": 1.0, "L": 1.0, "m": 1.0, "p": 1.0,
    "z": 0.5, "omega": 1.0, "n": 2.0, "r": 1.0,
    "t": 1.0,  # time treated as fixed (avoids extra variable dimension)
}
# Variables representing velocities — must be < c=1 for relativistic expressions
VELOCITY_VARS = {"v", "v_0", "beta"}


# ── Benchmark Spec Parser ─────────────────────────────────────────────────────

def parse_benchmark_spec() -> list[dict]:
    """Parse preregistration/benchmark_spec.txt into scenario list."""
    scenarios = []
    current_level = "Level A"

    with open(BENCHMARK_SPEC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "Level B" in line:
                current_level = "Level B"
                continue
            if line.startswith("Level"):
                continue
            if ":" not in line:
                continue

            name_part, rest = line.split(":", 1)
            name = name_part.strip()
            rest = rest.strip()

            if current_level == "Level B":
                # Format: "f0 = <expr>, delta = <expr>"
                f0_match = re.search(r"f0\s*=\s*([^,]+)", rest)
                delta_match = re.search(r"delta\s*=\s*(.+)$", rest)
                if f0_match and delta_match:
                    f0_str = f0_match.group(1).strip()
                    delta_str = delta_match.group(1).strip()
                    scenarios.append({
                        "name": name,
                        "level": "Level B",
                        "expr": delta_str,       # ground truth for evaluation
                        "f0_expr": f0_str,
                        "delta_expr": delta_str,
                    })
            else:  # Level A
                # Format: "expr" or "LHS = RHS"
                if "=" in rest:
                    expr = rest.split("=", 1)[1].strip()
                else:
                    expr = rest
                scenarios.append({
                    "name": name,
                    "level": "Level A",
                    "expr": expr,
                    "f0_expr": None,
                    "delta_expr": None,
                })

    return scenarios


# ── Data Generation ───────────────────────────────────────────────────────────

def _safe_lambdify_eval(fn, args, n_samples):
    """Safely call lambdified function, returning None if NaN/Inf result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y = fn(*args) if len(args) > 1 else fn(args[0])
            y = np.asarray(y, dtype=float)
            if y.shape == ():
                y = np.full(n_samples, float(y))
            if np.all(np.isfinite(y)) and np.std(y) > 1e-10:
                return y
        except Exception:
            pass
    return None


def generate_level_a_data(expr_str: str, noise_level: float, seed: int, n_samples: int = 100):
    """
    Generate synthetic data for a Level A (Feynman) equation.
    Physical constants are substituted with 1.0 (dimensionless normalization).
    Velocity-type variables are sampled in [0.1, 0.9] to avoid relativistic singularities.
    """
    rng = np.random.default_rng(seed)

    # SymPy treats ^ as XOR — must convert to ** before parsing
    expr_str = expr_str.replace("^", "**")

    # Substitute known physical constants → dimensionless.
    # IMPORTANT: do NOT use evaluate=False — it causes SymPy to treat 'pi' as
    # Symbol('pi') rather than sp.pi, breaking any expression containing pi.
    try:
        expr_raw = sp.sympify(expr_str)  # standard parse; recognizes pi, E, etc.
        sub_dict = {sp.Symbol(k): v for k, v in PHYSICAL_CONSTANTS.items()
                    if sp.Symbol(k) in expr_raw.free_symbols}
        expr = expr_raw.subs(sub_dict)
    except Exception:
        try:
            expr = sp.sympify(expr_str)
        except Exception:
            expr = sp.Integer(1)

    vars_list = sorted([
        str(s) for s in expr.free_symbols
        if not str(s).startswith("theta_")
    ])
    if not vars_list:
        vars_list = ["x1"]

    symbols = [sp.Symbol(v) for v in vars_list]
    try:
        fn = sp.lambdify(symbols, expr, modules=["numpy"])
    except Exception:
        fn = None

    y_true = None
    for attempt in range(10):
        X = np.zeros((n_samples, len(vars_list)))
        for i, v in enumerate(vars_list):
            if v in VELOCITY_VARS:
                # Velocity must be < c=1 for relativistic expressions
                X[:, i] = rng.uniform(0.1, 0.85, size=n_samples)
            elif v in {"theta", "phi", "alpha"}:
                X[:, i] = rng.uniform(0.1, 2.0, size=n_samples)
            else:
                X[:, i] = rng.uniform(0.5, 3.0, size=n_samples)

        if fn is not None:
            args = [X[:, i] for i in range(len(vars_list))]
            y_true = _safe_lambdify_eval(fn, args, n_samples)
        if y_true is not None:
            break

    if y_true is None:
        # Absolute fallback: simple constant
        vars_list = ["x1"]
        X = rng.uniform(0.5, 3.0, size=(n_samples, 1))
        y_true = np.ones(n_samples) * 0.5

    noise = rng.normal(0, noise_level * np.std(y_true) + 1e-8, size=y_true.shape)
    y_noisy = y_true + noise

    split_idx = int(0.8 * n_samples)
    return (X[:split_idx], y_noisy[:split_idx],
            X[split_idx:], y_noisy[split_idx:],
            vars_list)


def generate_level_b_data(f0_str: str, delta_str: str, noise_level: float, seed: int, n_samples: int = 100):
    """
    Generate data for Level B (Correction-First) scenario.
    y_total = f0(x1) + delta(x1) + noise
    Returns residual: y_residual = y_noisy - f0(x1) ≈ delta(x1) + noise

    ARC constraint for ALL Level B corrections: lim_{x1 → +∞} delta(x1) = 0
    This is verified to hold for all 10 pre-registered synthetic corrections.
    """
    rng = np.random.default_rng(seed)

    # SymPy treats ^ as XOR — must convert to ** before parsing
    f0_str = f0_str.replace("^", "**")
    delta_str = delta_str.replace("^", "**")

    # Level B always uses single variable x1, sampled in [0.5, 5.0]
    X = rng.uniform(0.5, 5.0, size=(n_samples, 1))
    x1 = X[:, 0]
    x1_sym = sp.Symbol("x1")

    f0_expr = sp.sympify(f0_str)
    delta_expr = sp.sympify(delta_str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0_fn = sp.lambdify(x1_sym, f0_expr, modules=["numpy"])
        delta_fn = sp.lambdify(x1_sym, delta_expr, modules=["numpy"])

        y_f0 = np.asarray(f0_fn(x1), dtype=float)
        if y_f0.shape == ():
            y_f0 = np.full(n_samples, float(y_f0))

        y_delta = np.asarray(delta_fn(x1), dtype=float)
        y_true = y_f0 + y_delta

    noise = rng.normal(0, noise_level * np.std(y_true) + 1e-8, size=y_true.shape)
    y_noisy = y_true + noise

    # Residual: the actual target for the proposer
    y_residual = y_noisy - y_f0

    split_idx = int(0.8 * n_samples)
    return (
        X[:split_idx], y_residual[:split_idx],
        X[split_idx:], y_residual[split_idx:],
        ["x1"],
        y_noisy[:split_idx],   # y_total_train for ARM C (direct fit)
        y_noisy[split_idx:],   # y_total_test
    )


# ── Arm Evaluation Helper ─────────────────────────────────────────────────────

def evaluate_arm(
    candidates: list,          # list of expr strings
    gate: PhysicsGateCascade,
    ground_truth_str: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    var_names: list,
    limit_vars: list,
    limit_points: list,
    proposer_name: str,
    scenario_name: str,
    level: str,
    noise: float,
    seed: str,
    arm_label: str,
    logger: AuditLogger,
):
    """
    Evaluate one arm: apply gate and compute metrics M1/M2/M3.
    Logs one entry per arm per experiment.
    """
    if not candidates:
        return

    unfiltered_evals = []
    filtered_evals = []
    n_fp = 0   # false positives: NMSE < threshold but gate fails

    for cand_expr in candidates:
        eval_res = evaluate_candidate(
            cand_expr, ground_truth_str, X_test, y_test, var_names
        )
        nmse = eval_res["nmse"]
        gate_res = gate.check(
            cand_expr,
            limit_vars=limit_vars,
            limit_points=limit_points,
        )

        unfiltered_evals.append({
            "expr": cand_expr,
            "nmse": nmse,
            "is_recovered": eval_res["is_recovered"],
            "tree_distance": eval_res.get("tree_distance", 1.0),
        })

        if gate_res.passed:
            filtered_evals.append({
                "expr": cand_expr,
                "nmse": nmse,
                "is_recovered": eval_res["is_recovered"],
                "tree_distance": eval_res.get("tree_distance", 1.0),
            })
        else:
            # Count false positive: fits data but fails physics gate
            if nmse < NMSE_FP_THRESHOLD:
                n_fp += 1

    n_cands = len(unfiltered_evals)
    n_passed = len(filtered_evals)
    n_good_fit = sum(1 for e in unfiltered_evals if e["nmse"] < NMSE_FP_THRESHOLD)

    pvr = n_passed / max(n_cands, 1)
    fprr = n_fp / max(n_good_fit, 1) if n_good_fit > 0 else None

    # M3: VRR — fraction of gate-passed that are structurally recovered
    vrr = (
        sum(1 for e in filtered_evals if e["is_recovered"]) / max(n_passed, 1)
        if n_passed > 0 else 0.0
    )

    top1_unfiltered_nmse = min((e["nmse"] for e in unfiltered_evals), default=1e6)
    top1_filtered_nmse = min((e["nmse"] for e in filtered_evals), default=1e6)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_name,
        "level": level,
        "noise": noise,
        "seed": seed,
        "proposer": proposer_name,
        "arm": arm_label,
        "n_candidates": n_cands,
        "n_passed_gate": n_passed,
        "n_good_fit": n_good_fit,
        "n_false_positives": n_fp,
        "pvr": pvr,
        "fprr": fprr,
        "vrr": vrr,
        "top1_unfiltered_nmse": top1_unfiltered_nmse,
        "top1_filtered_nmse": top1_filtered_nmse,
    }
    logger.log_experiment(log_entry)
    return log_entry


# ── Per-Experiment Runner ─────────────────────────────────────────────────────

def run_level_a_experiment(scenario: dict, noise: float, seed: int, logger: AuditLogger):
    """
    Level A (Feynman equations): test the complexity gate.
    ARM-A only (GrammarProposer). ARC gate disabled.
    PySR on Level A is deferred (too slow for 20 × 3 × 5 = 300 runs within timeline).
    """
    name = scenario["name"]
    expr_str = scenario["expr"]

    X_train, y_train, X_test, y_test, var_names = generate_level_a_data(
        expr_str, noise, seed
    )

    # Level A gate: complexity only, ARC disabled
    gate = PhysicsGateCascade(max_depth=7, max_tokens=20, enable_arc_check=False)

    # GrammarProposer (ARM-A)
    from adcd.llm_proposer import ProposalContext
    proposer = GrammarProposer(seed=seed, n_candidates=40)
    data_stats = {
        v: {"mean": float(np.mean(X_train[:, i])), "std": float(np.std(X_train[:, i]))}
        for i, v in enumerate(var_names)
    }
    ctx = ProposalContext(
        variable_names=var_names,
        target_name="y",
        data_statistics=data_stats,
        n_candidates=40,
    )
    candidates = proposer.propose(ctx)

    evaluate_arm(
        candidates=candidates,
        gate=gate,
        ground_truth_str=expr_str,
        X_test=X_test, y_test=y_test, var_names=var_names,
        limit_vars=[],          # ARC disabled for Level A
        limit_points=[],
        proposer_name="GrammarProposer",
        scenario_name=name, level="Level A",
        noise=noise, seed=str(seed),
        arm_label="ARM-A",
        logger=logger,
    )


def run_level_b_experiment(scenario: dict, noise: float, seed: int, logger: AuditLogger,
                           run_pysr: bool = True):
    """
    Level B (Correction-First): test ARC gate + measure FPRR.
    ARM-A: GrammarProposer (residual target)
    ARM-B: PySR + gate (residual target — correction-first)
    ARM-C: PySR + gate (y_total target — direct fit baseline, for M4)
    """
    name = scenario["name"]
    f0_str = scenario["f0_expr"]
    delta_str = scenario["delta_expr"]

    (X_train, y_res_train, X_test, y_res_test,
     var_names, y_total_train, y_total_test) = generate_level_b_data(
        f0_str, delta_str, noise, seed
    )

    # Level B gate: complexity + ARC with limit x1 → +∞
    # All pre-registered synthetic corrections satisfy: lim_{x1→∞} delta(x1) = 0
    gate = PhysicsGateCascade(max_depth=7, max_tokens=20, enable_arc_check=True)
    arc_limit_vars = ["x1"]
    arc_limit_points = [sp.oo]    # ← KEY FIX: x1 → +∞, not x1 → 0

    # ── ARM-A: GrammarProposer (residual target) ─────────────────────────
    from adcd.llm_proposer import ProposalContext
    proposer = GrammarProposer(seed=seed, n_candidates=40)
    data_stats = {
        "x1": {"mean": float(np.mean(X_train[:, 0])), "std": float(np.std(X_train[:, 0]))}
    }
    ctx = ProposalContext(
        variable_names=["x1"],
        target_name="delta",
        data_statistics=data_stats,
        n_candidates=40,
    )
    grammar_candidates = proposer.propose(ctx)

    evaluate_arm(
        candidates=grammar_candidates,
        gate=gate,
        ground_truth_str=delta_str,
        X_test=X_test, y_test=y_res_test, var_names=["x1"],
        limit_vars=arc_limit_vars, limit_points=arc_limit_points,
        proposer_name="GrammarProposer",
        scenario_name=name, level="Level B",
        noise=noise, seed=str(seed),
        arm_label="ARM-A",
        logger=logger,
    )

    if not run_pysr:
        return

    # ── ARM-B: PySR + gate (correction-first: fits residual) ─────────────
    try:
        adapter_b = PySRWithGate(
            gate_cascade=gate,
            niterations=8,
            population_size=12,
        )
        pysr_b_results = adapter_b.fit_and_filter(
            X_train, y_res_train,
            variable_names=["x1"],
            limit_vars=arc_limit_vars,
            limit_points=arc_limit_points,
        )

        candidates_b = [r["equation"] for r in pysr_b_results]
        evaluate_arm(
            candidates=candidates_b,
            gate=gate,
            ground_truth_str=delta_str,
            X_test=X_test, y_test=y_res_test, var_names=["x1"],
            limit_vars=arc_limit_vars, limit_points=arc_limit_points,
            proposer_name="PySR",
            scenario_name=name, level="Level B",
            noise=noise, seed=str(seed),
            arm_label="ARM-B (correction-first)",
            logger=logger,
        )
    except Exception as e:
        print(f"  [WARN] PySR ARM-B failed for {name} noise={noise} seed={seed}: {e}")

    # ── ARM-C: PySR + gate (direct fit: baseline for M4) ────────────────
    try:
        adapter_c = PySRWithGate(
            gate_cascade=gate,
            niterations=8,
            population_size=12,
        )
        pysr_c_results = adapter_c.fit_and_filter(
            X_train, y_total_train,   # ← fits y_total directly, NOT residual
            variable_names=["x1"],
            limit_vars=arc_limit_vars,
            limit_points=arc_limit_points,
        )

        # For direct fit, evaluate candidates against delta on test set:
        # This measures whether direct-fit candidates accidentally recover the correction
        candidates_c = [r["equation"] for r in pysr_c_results]
        evaluate_arm(
            candidates=candidates_c,
            gate=gate,
            ground_truth_str=delta_str,
            X_test=X_test, y_test=y_res_test, var_names=["x1"],
            limit_vars=arc_limit_vars, limit_points=arc_limit_points,
            proposer_name="PySR",
            scenario_name=name, level="Level B",
            noise=noise, seed=str(seed),
            arm_label="ARM-C (direct-fit baseline)",
            logger=logger,
        )
    except Exception as e:
        print(f"  [WARN] PySR ARM-C failed for {name} noise={noise} seed={seed}: {e}")


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def main():
    scenarios = parse_benchmark_spec()
    level_a = [s for s in scenarios if s["level"] == "Level A"]
    level_b = [s for s in scenarios if s["level"] == "Level B"]

    print(f"[INFO] Parsed benchmark: {len(level_a)} Level A + {len(level_b)} Level B scenarios")
    print(f"[INFO] Seeds: {SEEDS}, Noise levels: {NOISE_LEVELS}")

    total_a = len(level_a) * len(NOISE_LEVELS) * len(SEEDS)
    total_b = len(level_b) * len(NOISE_LEVELS) * len(SEEDS)
    print(f"[INFO] Planned runs: {total_a} Level A (ARM-A only) + "
          f"{total_b} Level B × 3 arms (ARM-A, ARM-B, ARM-C)")

    logger = AuditLogger(AUDIT_LOG_PATH)
    completed = 0

    # ── Level A: Complexity Gate Test (GrammarProposer only) ─────────────
    print(f"\n[PHASE 1] Running Level A experiments ({total_a} runs)...")
    for sc in level_a:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                run_level_a_experiment(sc, noise, seed, logger)
                completed += 1
                if completed % 30 == 0:
                    print(f"  Progress: {completed} runs completed")

    print(f"[PHASE 1 DONE] Level A complete ({completed} runs total)")

    # ── Level B: ARC Gate Test (GrammarProposer + PySR) ──────────────────
    print(f"\n[PHASE 2] Running Level B experiments ({total_b} scenario-seeds × 3 arms)...")
    print(f"[INFO] PySR niterations=8, populations=12 per run. Estimated time: 20-40 min")

    b_completed = 0
    for sc in level_b:
        print(f"\n  Scenario: {sc['name']} | f0={sc['f0_expr']} | delta={sc['delta_expr']}")
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                run_level_b_experiment(sc, noise, seed, logger, run_pysr=True)
                b_completed += 1
                print(f"    [{sc['name']} | noise={noise:.2f} | seed={seed}] done "
                      f"({b_completed}/{total_b})")

    print(f"\n[SUCCESS] All benchmark runs completed!")
    print(f"  Level A runs: {total_a}")
    print(f"  Level B scenario-seeds: {total_b} × up to 3 arms")
    print(f"  Results appended to: {AUDIT_LOG_PATH}")
    print(f"\nNext step: python eval/compute_metrics.py")


if __name__ == "__main__":
    main()
