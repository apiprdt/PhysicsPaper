
import sys
sys.path.insert(0, 'src')
import numpy as np
import sympy as sp
import json
from datetime import datetime, timezone
from adcd.arc_gate import ARCGate
from adcd.constrained_pysr import ADCDConstrainedPySR

SEEDS = [0, 42, 123]
N = 200
NOISE = 0.02
TIMEOUT = 180
TIMESTAMP = datetime.now(timezone.utc).isoformat()

print("=" * 65)
print("ADCD REDESIGN: 3-WAY COMPARISON")
print("=" * 65)
print(f"Timestamp : {TIMESTAMP}")
print(f"Seeds     : {SEEDS}")
print(f"N samples : {N}, noise={NOISE}")
print(f"Timeout   : {TIMEOUT}s per seed")
print()

# ====================================================================
# DATA GENERATION
# ====================================================================
def generate_data(seed):
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.05, 0.85, N)
    f0 = 0.5 * v**2
    y_true = 1.0 / np.sqrt(1.0 - v**2) - 1.0
    delta_true = y_true - f0
    sigma = NOISE * np.std(delta_true)
    y_obs_noisy = y_true + rng.normal(0, sigma, N)
    delta_noisy = y_obs_noisy - f0
    return v.reshape(-1,1), y_obs_noisy, f0, delta_noisy, delta_true

# ====================================================================
# ARC GATE (shared)
# ====================================================================
arc = ARCGate(target_limit=0.0, tolerance=1e-3)

def verify_candidates(df, var_name, limit_sym):
    passed, rejected = [], []
    for _, row in df.iterrows():
        expr = str(row.get("sympy_format", row.get("equation", "")))
        loss = float(row.get("loss", 999))
        arc_res = arc.check(expr.replace("^","**"), [var_name], [limit_sym])
        entry = {"expr": expr, "nmse": loss, "arc_passed": arc_res.passed,
                 "arc_limit": arc_res.limit_value, "reason": arc_res.rejection_reason}
        (passed if arc_res.passed else rejected).append(entry)
    return passed, rejected

def has_sqrt(expr_str):
    return "sqrt" in expr_str.lower()

# ====================================================================
# PYSR IMPORT
# ====================================================================
try:
    from pysr import PySRRegressor
    PYSR_OK = True
except ImportError:
    PYSR_OK = False
    print("ERROR: PySR not available")
    sys.exit(1)

# ====================================================================
# ARM A: PySR SOLO (no ADCD at all)
# ====================================================================
print("=" * 65)
print("ARM A: PySR SOLO (no ADCD, tabula rasa)")
print("  PySR searches for y_obs directly (not correction-first)")
print("=" * 65)

arm_a_results = []
for seed in SEEDS:
    X, y_obs, f0, delta_noisy, delta_true = generate_data(seed)
    pysr = PySRRegressor(
        niterations=80,
        populations=25,
        maxsize=28,
        binary_operators=["+","-","*","/"],
        unary_operators=["sqrt","square","abs","exp"],
        extra_sympy_mappings={"square": lambda x: x**2},
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        timeout_in_seconds=TIMEOUT,
        verbosity=0,
        temp_equation_file=True,
    )
    print(f"  [seed={seed}] fitting on y_obs directly...", flush=True)
    pysr.fit(X, y_obs, variable_names=["v"])
    passed, rejected = verify_candidates(pysr.equations_, "v", sp.Integer(0))
    best_p = min(passed, key=lambda x:x["nmse"]) if passed else None
    print(f"  [seed={seed}] {len(passed)+len(rejected)} candidates: "
          f"{len(passed)} ARC-passed ({len(passed)/(len(passed)+len(rejected)+1e-9):.1%})")
    if best_p:
        has_s = has_sqrt(best_p["expr"])
        print(f"             best: {best_p['expr']}  NMSE={best_p['nmse']:.4f}  sqrt={has_s}")
    arm_a_results.append({
        "seed": seed, "n_cands": len(passed)+len(rejected),
        "n_arc_passed": len(passed),
        "best_expr": best_p["expr"] if best_p else "NULL",
        "best_nmse": best_p["nmse"] if best_p else 999,
        "best_has_sqrt": has_sqrt(best_p["expr"]) if best_p else False,
    })

# ====================================================================
# ARM B: PySR + ADCD POST-FILTER (correction-first but no ARC in data)
# ====================================================================
print()
print("=" * 65)
print("ARM B: PySR + ADCD POST-FILTER")
print("  Correction-first: fits delta=y_obs-f0")
print("  ARC applied AFTER PySR (original approach)")
print("=" * 65)

arm_b_results = []
for seed in SEEDS:
    X, y_obs, f0, delta_noisy, delta_true = generate_data(seed)
    # v normalized to beta = v/max(v) ≈ v/0.85
    scale = float(np.max(X))
    X_norm = X / scale
    pysr = PySRRegressor(
        niterations=80,
        populations=25,
        maxsize=28,
        binary_operators=["+","-","*","/"],
        unary_operators=["sqrt","square","abs"],
        extra_sympy_mappings={"square": lambda x: x**2},
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        timeout_in_seconds=TIMEOUT,
        verbosity=0,
        temp_equation_file=True,
    )
    print(f"  [seed={seed}] fitting on delta (correction-first, norm)...", flush=True)
    pysr.fit(X_norm, delta_noisy, variable_names=["v_b"])
    passed, rejected = verify_candidates(pysr.equations_, "v_b", sp.Integer(0))
    best_p = min(passed, key=lambda x:x["nmse"]) if passed else None
    n_total = len(passed) + len(rejected)
    print(f"  [seed={seed}] {n_total} candidates: "
          f"{len(passed)} ARC-passed ({len(passed)/(n_total+1e-9):.1%})")
    if best_p:
        has_s = has_sqrt(best_p["expr"])
        print(f"             best: {best_p['expr']}  NMSE={best_p['nmse']:.4f}  sqrt={has_s}")
    arm_b_results.append({
        "seed": seed, "n_cands": n_total,
        "n_arc_passed": len(passed),
        "best_expr": best_p["expr"] if best_p else "NULL",
        "best_nmse": best_p["nmse"] if best_p else 999,
        "best_has_sqrt": has_sqrt(best_p["expr"]) if best_p else False,
    })

# ====================================================================
# ARM C: ADCD INTEGRATED (redesign — all constraints in search)
# ====================================================================
print()
print("=" * 65)
print("ARM C: ADCD INTEGRATED (redesign)")
print("  All 5 ADCD components built into PySR search")
print("=" * 65)

arm_c_results = []
for seed in SEEDS:
    X, y_obs, f0, delta_noisy, delta_true = generate_data(seed)
    model = ADCDConstrainedPySR(
        physics_domain="mechanics",
        arc_n_anchors=15,
        arc_anchor_weight=5.0,
        arc_tolerance=1e-3,
        normalize_variables=True,
        char_scales={"v": 1.0},  # v/c, c=1 in natural units
        niterations=80,
        maxsize=28,
        populations=25,
        random_state=seed,
        deterministic=True,
        timeout_in_seconds=TIMEOUT,
    )
    print(f"  [seed={seed}] ADCDConstrainedPySR fitting...", flush=True)
    result = model.fit(X, y_obs, f0, ["v"],
                       limit_var_idx=0, limit_point=0.0)
    best_has_s = has_sqrt(result.best_expr)
    arm_c_results.append({
        "seed": seed,
        "n_cands": result.n_total_candidates,
        "n_arc_passed": result.n_arc_verified,
        "arc_pass_rate": result.arc_pass_rate,
        "best_expr": result.best_expr,
        "best_nmse": result.best_nmse,
        "best_has_sqrt": best_has_s,
    })

# ====================================================================
# COMPARISON TABLE
# ====================================================================
print()
print("=" * 65)
print("COMPARISON RESULTS")
print("=" * 65)
print(f"{'Metric':<30} {'ARM A':>10} {'ARM B':>10} {'ARM C':>10}")
print(f"{'':30} {'Solo':>10} {'PostFilter':>10} {'Integrated':>10}")
print("-" * 65)

def avg(lst, key):
    vals = [r[key] for r in lst if key in r]
    return sum(vals)/len(vals) if vals else 0

avg_arc_a = avg(arm_a_results, "n_arc_passed") / max(avg(arm_a_results, "n_cands"), 1)
avg_arc_b = avg(arm_b_results, "n_arc_passed") / max(avg(arm_b_results, "n_cands"), 1)
avg_arc_c = avg(arm_c_results, "arc_pass_rate")

sqrt_a = sum(1 for r in arm_a_results if r["best_has_sqrt"]) / len(arm_a_results)
sqrt_b = sum(1 for r in arm_b_results if r["best_has_sqrt"]) / len(arm_b_results)
sqrt_c = sum(1 for r in arm_c_results if r["best_has_sqrt"]) / len(arm_c_results)

null_a = sum(1 for r in arm_a_results if r["best_expr"] == "NULL")
null_b = sum(1 for r in arm_b_results if r["best_expr"] == "NULL")
null_c = sum(1 for r in arm_c_results if r["best_expr"] == "NULL")

print(f"{'ARC pass rate (avg)':<30} {avg_arc_a:>10.1%} {avg_arc_b:>10.1%} {avg_arc_c:>10.1%}")
print(f"{'Seeds with sqrt in best':<30} {sqrt_a:>10.1%} {sqrt_b:>10.1%} {sqrt_c:>10.1%}")
print(f"{'Null results (no ARC pass)':<30} {null_a:>10d} {null_b:>10d} {null_c:>10d}")
print()
print("Per-seed details:")
for i, seed in enumerate(SEEDS):
    ra = arm_a_results[i]; rb = arm_b_results[i]; rc = arm_c_results[i]
    print(f"  seed={seed}:")
    print(f"    A: {ra['n_arc_passed']}/{ra['n_cands']} passed  best={ra['best_expr'][:50]}")
    print(f"    B: {rb['n_arc_passed']}/{rb['n_cands']} passed  best={rb['best_expr'][:50]}")
    print(f"    C: {rc['n_arc_passed']}/{rc['n_cands']} passed  best={rc['best_expr'][:50]}")
    print()

# Save audit
import os; os.makedirs("results", exist_ok=True)
audit = {
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "arm_a": arm_a_results, "arm_b": arm_b_results, "arm_c": arm_c_results,
    "summary": {
        "arc_pass_rate": {"A": avg_arc_a, "B": avg_arc_b, "C": avg_arc_c},
        "sqrt_recovery_rate": {"A": sqrt_a, "B": sqrt_b, "C": sqrt_c},
        "null_results": {"A": null_a, "B": null_b, "C": null_c},
    }
}
with open("results/rediscovery_comparison.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(audit) + "\n")
print(f"Audit saved: results/rediscovery_comparison.jsonl")
print("Status: COMPLETE")
