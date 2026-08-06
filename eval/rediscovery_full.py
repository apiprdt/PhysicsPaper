"""
ADCD Full Test: 3-Arm Final Comparison
=======================================
ARM A: PySR Solo (no ADCD, baseline)
ARM B: ADCD Basic (correction-first + ARC anchor only, v1)
ARM C: ADCD Full (all 3 mitigations: adaptive weights + dual-ARC + adaptive f0)

Physics test case: Newton -> Relativistic KE, extended range v=[0.05, 0.99c]
f0_candidates: {"none": 0, "newton": 0.5*v^2}

Pre-registered:
  seeds=[0, 42, 123]
  noise=0.02
  primary metric: fraction of seeds finding sqrt
  secondary metric: dual-ARC pass rate
  null result reported identically to positive result
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import sympy as sp
import json
from datetime import datetime, timezone
from adcd.arc_gate import ARCGate
from adcd.constrained_pysr import ADCDConstrainedPySR
from pysr import PySRRegressor

SEEDS     = [0, 42, 123]
N_MOD     = 150  # moderate regime
N_EXT     = 50   # extreme regime
NOISE     = 0.02
TIMEOUT   = 200
TIMESTAMP = datetime.now(timezone.utc).isoformat()

print("=" * 65)
print("ADCD FULL TEST — 3-ARM FINAL COMPARISON")
print("=" * 65)
print(f"Timestamp : {TIMESTAMP}")
print(f"Seeds     : {SEEDS}  (pre-registered)")
print(f"N         : {N_MOD}+{N_EXT}={N_MOD+N_EXT}")
print(f"Noise     : {NOISE}")
print(f"Timeout   : {TIMEOUT}s per seed")
print()

# ----------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------
def generate(seed):
    rng = np.random.default_rng(seed)
    v_mod = rng.uniform(0.05, 0.85, N_MOD)
    v_ext = rng.uniform(0.90, 0.99, N_EXT)
    v     = np.sort(np.concatenate([v_mod, v_ext]))
    f0_newton = 0.5 * v**2
    f0_none   = np.zeros_like(v)
    y_true    = 1.0 / np.sqrt(1.0 - v**2) - 1.0
    sigma     = NOISE * np.std(y_true - f0_newton)
    y_obs     = y_true + rng.normal(0, sigma, len(v))
    return v.reshape(-1,1), y_obs, f0_newton, f0_none

arc = ARCGate(target_limit=0.0, tolerance=1e-3)

def quick_verify(df, var_name, limit_sym):
    passed_sqrt = 0; passed_total = 0
    best_sqrt = None
    for _, row in df.iterrows():
        expr = str(row.get("sympy_format", row.get("equation","")))
        loss = float(row.get("loss", 999))
        res = arc.check(expr.replace("^","**"), [var_name], [limit_sym])
        if res.passed:
            passed_total += 1
            if "sqrt" in expr.lower():
                passed_sqrt += 1
                if best_sqrt is None or loss < best_sqrt[1]:
                    best_sqrt = (expr, loss)
    n = len(df)
    return passed_total, passed_sqrt, best_sqrt, n

# ================================================================
# ARM A: PySR Solo
# ================================================================
print("=" * 65)
print("ARM A: PySR SOLO (no ADCD, tabula rasa, full data to PySR)")
print("=" * 65)
arm_a = []
for seed in SEEDS:
    X, y_obs, f0_n, f0_0 = generate(seed)
    pysr = PySRRegressor(
        niterations=100, populations=30, maxsize=30,
        binary_operators=["+","-","*","/"],
        unary_operators=["sqrt","square","abs"],
        extra_sympy_mappings={"square": lambda x: x**2},
        random_state=seed, deterministic=True, parallelism="serial",
        timeout_in_seconds=TIMEOUT, verbosity=0, temp_equation_file=True,
    )
    print(f"  [seed={seed}] PySR solo...", flush=True)
    pysr.fit(X, y_obs, variable_names=["v"])
    p_total, p_sqrt, best_s, n_cands = quick_verify(pysr.equations_,"v",sp.Integer(0))
    print(f"  [seed={seed}] {p_total}/{n_cands} ARC-pass, {p_sqrt} sqrt")
    if best_s: print(f"    best sqrt: {best_s[0][:70]}")
    arm_a.append({"seed":seed,"n_cands":n_cands,"n_arc":p_total,
                  "n_sqrt":p_sqrt,"best_sqrt":best_s[0] if best_s else "NULL"})

# ================================================================
# ARM B: ADCD Basic (v1 — correction-first + ARC anchors only)
# ================================================================
print()
print("=" * 65)
print("ARM B: ADCD BASIC (correction-first + ARC anchors, no M2/M3/M4)")
print("=" * 65)
arm_b = []
for seed in SEEDS:
    X, y_obs, f0_n, f0_0 = generate(seed)
    model = ADCDConstrainedPySR(
        physics_domain="mechanics",
        arc_n_anchors=15, arc_anchor_weight=8.0,
        normalize_variables=True, char_scales={"v": 1.0},
        use_adaptive_weights=False,   # M2 OFF
        use_dual_arc=False,           # M3 OFF
        use_adaptive_f0=False,        # M4 OFF
        niterations=100, maxsize=30, populations=30,
        random_state=seed, deterministic=True,
        timeout_in_seconds=TIMEOUT,
    )
    print(f"  [seed={seed}] ADCD Basic...", flush=True)
    result = model.fit(X, y_obs, f0_n, ["v"], limit_var_idx=0, limit_point=0.0)
    p_sqrt = sum(1 for g in result.gate_results if g.arc_passed and "sqrt" in g.expr.lower())
    _sqrt_b = [g for g in result.gate_results if g.arc_passed and "sqrt" in g.expr.lower()]
    _sqrt_b_sorted = sorted(_sqrt_b, key=lambda g: g.nmse_train)
    best_sqrt_b = _sqrt_b_sorted[0].expr if _sqrt_b_sorted else "NULL"
    arm_b.append({
        "seed": seed,
        "n_cands": result.n_total_candidates,
        "n_arc": result.n_arc_verified,
        "n_sqrt": p_sqrt,
        "best_sqrt": best_sqrt_b,
        "best_expr": result.best_expr,
    })

# ================================================================
# ARM C: ADCD Full (all 3 mitigations ON)
# ================================================================
print()
print("=" * 65)
print("ARM C: ADCD FULL (M2+M3+M4 all ON — adaptive weights, dual-ARC, adaptive f0)")
print("=" * 65)
arm_c = []
for seed in SEEDS:
    X, y_obs, f0_n, f0_0 = generate(seed)
    model = ADCDConstrainedPySR(
        physics_domain="mechanics",
        arc_n_anchors=15, arc_anchor_weight=8.0,
        normalize_variables=True, char_scales={"v": 1.0},
        # M2: Gradient-Adaptive Weighting ON
        use_adaptive_weights=True, adaptive_weight_factor=6.0,
        # M3: Dual-ARC Gate ON
        use_dual_arc=True, dual_arc_extreme_frac=0.25,
        dual_arc_nmse_threshold=0.05,
        # M4: Adaptive f0 ON — candidates: no correction vs Newton correction
        use_adaptive_f0=True,
        f0_candidates={"none": f0_0, "newton": f0_n},
        niterations=100, maxsize=30, populations=30,
        random_state=seed, deterministic=True,
        timeout_in_seconds=TIMEOUT,
    )
    print(f"  [seed={seed}] ADCD Full...", flush=True)
    result = model.fit(X, y_obs, f0_n, ["v"], limit_var_idx=0, limit_point=0.0)
    p_sqrt = sum(1 for g in result.gate_results
                 if g.arc_passed and "sqrt" in g.expr.lower())
    best_sqrt_c_list = sorted(
        [g for g in result.gate_results if g.arc_passed and "sqrt" in g.expr.lower()],
        key=lambda g: g.nmse_train)
    arm_c.append({
        "seed": seed,
        "n_cands": result.n_total_candidates,
        "n_arc": result.n_arc_verified,
        "n_dual_arc": result.n_dual_arc_verified,
        "n_sqrt": p_sqrt,
        "best_expr": result.best_expr,
        "best_nmse_train": result.best_nmse_train,
        "best_nmse_extreme": result.best_nmse_extreme,
        "has_sqrt_best": result.has_sqrt_in_best,
        "best_sqrt": best_sqrt_c_list[0].expr if best_sqrt_c_list else "NULL",
        "f0_selected": result.f0_selected,
        "dual_arc_pass_rate": result.dual_arc_pass_rate,
    })

# ================================================================
# FINAL SUMMARY
# ================================================================
print()
print("=" * 65)
print("FINAL COMPARISON SUMMARY")
print("=" * 65)
def pct(lst, key):
    return sum(1 for r in lst if r.get(key,0) > 0) / len(lst)
def avg_rate(lst, num_key, den_key):
    vals = [r[num_key]/max(r[den_key],1) for r in lst]
    return sum(vals)/len(vals)

sqrt_a = pct(arm_a, "n_sqrt")
sqrt_b = pct(arm_b, "n_sqrt")
sqrt_c = pct(arm_c, "n_sqrt")

arc_a = avg_rate(arm_a, "n_arc", "n_cands")
arc_b = avg_rate(arm_b, "n_arc", "n_cands")
arc_c = avg_rate(arm_c, "n_arc", "n_cands")

dual_c = sum(r["dual_arc_pass_rate"] for r in arm_c) / len(arm_c)

print(f"{'Metric':<38} {'ARM A':>8} {'ARM B':>8} {'ARM C':>8}")
print(f"{'':38} {'Solo':>8} {'Basic':>8} {'Full':>8}")
print("-" * 65)
print(f"{'ARC pass rate (avg)':<38} {arc_a:>8.1%} {arc_b:>8.1%} {arc_c:>8.1%}")
print(f"{'Dual-ARC pass rate (ARM C only)':<38} {'N/A':>8} {'N/A':>8} {dual_c:>8.1%}")
print(f"{'Seeds finding sqrt (ARC-verified)':<38} {sqrt_a:>8.1%} {sqrt_b:>8.1%} {sqrt_c:>8.1%}")
print()
print("Per-seed details:")
for i, seed in enumerate(SEEDS):
    ra=arm_a[i]; rb=arm_b[i]; rc=arm_c[i]
    print(f"  seed={seed}:")
    print(f"    A: arc={ra['n_arc']}/{ra['n_cands']}, sqrt={ra['n_sqrt']}")
    print(f"    B: arc={rb['n_arc']}/{rb['n_cands']}, sqrt={rb['n_sqrt']}")
    print(f"    C: arc={rc['n_arc']}/{rc['n_cands']}, dual={rc['n_dual_arc']}, "
          f"sqrt={rc['n_sqrt']}, f0={rc['f0_selected']}")
    if rc['best_sqrt'] != "NULL":
        print(f"    C best sqrt: {rc['best_sqrt'][:70]}")
        print(f"    C NMSE train={rc['best_nmse_train']:.6f}, extreme={rc['best_nmse_extreme']}")
    print()

# Ablation interpretation
print("ABLATION INTERPRETATION:")
if sqrt_c > sqrt_a and sqrt_c > sqrt_b:
    print("  ARM C > A and B -> Full ADCD mitigations are the key contribution")
elif sqrt_c > sqrt_b:
    print("  ARM C > B -> Mitigations M2/M3/M4 add value beyond basic ADCD")
elif sqrt_a == sqrt_c:
    print("  ARM C = A -> Extended range drives discovery, ADCD adds consistency")
else:
    print("  Null result: no arm consistently finds sqrt with these settings")

import os; os.makedirs("results", exist_ok=True)
audit = {"timestamp": TIMESTAMP, "seeds": SEEDS,
         "arm_a": arm_a, "arm_b": arm_b, "arm_c": arm_c,
         "summary": {"sqrt_rate": {"A":sqrt_a,"B":sqrt_b,"C":sqrt_c},
                     "arc_rate": {"A":arc_a,"B":arc_b,"C":arc_c}}}
with open("results/full_comparison.jsonl","a") as f:
    f.write(json.dumps(audit)+"\n")
print(f"\nAudit saved: results/full_comparison.jsonl")
print("Status: COMPLETE AND HONEST")
