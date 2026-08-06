"""
Ablation: PySR Solo vs ADCD Integrated, keduanya pada extended range
Tujuan: isolasi kontribusi ADCD vs kontribusi data range semata
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

SEEDS = [0, 42, 123]
TIMEOUT = 200
TIMESTAMP = datetime.now(timezone.utc).isoformat()

print("=" * 65)
print("ABLATION: PySR Solo vs ADCD Integrated (same extended range)")
print("Tujuan: isolasi kontribusi ADCD vs data range saja")
print("=" * 65)

def generate_extended(seed, noise=0.02):
    rng = np.random.default_rng(seed)
    v_mod = rng.uniform(0.05, 0.85, 150)
    v_rel = rng.uniform(0.90, 0.99, 50)
    v = np.sort(np.concatenate([v_mod, v_rel]))
    f0 = 0.5 * v**2
    y_true = 1.0 / np.sqrt(1.0 - v**2) - 1.0
    delta_true = y_true - f0
    sigma = noise * np.std(delta_true)
    y_obs = y_true + rng.normal(0, sigma, len(v))
    return v.reshape(-1,1), y_obs, f0, delta_true

arc = ARCGate(target_limit=0.0, tolerance=1e-3)

def count_sqrt(candidates_df, var_name, limit_sym):
    arc_pass, arc_sqrt = 0, 0
    best_sqrt = None
    for _, row in candidates_df.iterrows():
        expr = str(row.get("sympy_format", row.get("equation","")))
        loss = float(row.get("loss", 999))
        arc_res = arc.check(expr.replace("^","**"), [var_name], [limit_sym])
        if arc_res.passed:
            arc_pass += 1
            if "sqrt" in expr.lower():
                arc_sqrt += 1
                if best_sqrt is None or loss < best_sqrt[1]:
                    best_sqrt = (expr, loss)
    return arc_pass, arc_sqrt, best_sqrt

# ----------------------------------------------------------------
# ARM A: PySR Solo, extended range, no ADCD at all
# ----------------------------------------------------------------
print("\nARM A: PySR SOLO on extended range (no ADCD)")
arm_a = []
for seed in SEEDS:
    X, y_obs, f0, delta_true = generate_extended(seed)
    pysr = PySRRegressor(
        niterations=100, populations=30, maxsize=30,
        binary_operators=["+","-","*","/"],
        unary_operators=["sqrt","square","abs"],
        extra_sympy_mappings={"square": lambda x: x**2},
        random_state=seed, deterministic=True, parallelism="serial",
        timeout_in_seconds=TIMEOUT, verbosity=0, temp_equation_file=True,
    )
    print(f"  [seed={seed}] fitting y_obs directly (no correction)...", flush=True)
    pysr.fit(X, y_obs, variable_names=["v"])
    n_arc, n_sqrt, best_s = count_sqrt(pysr.equations_, "v", sp.Integer(0))
    n_total = len(pysr.equations_)
    print(f"  [seed={seed}]: {n_arc}/{n_total} ARC-pass, {n_sqrt} sqrt-candidates")
    if best_s:
        print(f"    best sqrt: {best_s[0][:70]}  NMSE={best_s[1]:.6f}")
    arm_a.append({"seed":seed,"n_total":n_total,"n_arc":n_arc,"n_sqrt":n_sqrt,
                  "best_sqrt": best_s[0] if best_s else "NULL"})

# ----------------------------------------------------------------
# ARM B-ext: Correction-First only, NO ARC anchor, extended range
# ----------------------------------------------------------------
print("\nARM B-ext: Correction-First ONLY (no ARC anchor, no operator prior)")
arm_b = []
for seed in SEEDS:
    X, y_obs, f0, delta_true = generate_extended(seed)
    scale = float(np.max(X))
    X_norm = X / scale
    delta_noisy = y_obs - f0
    pysr = PySRRegressor(
        niterations=100, populations=30, maxsize=30,
        binary_operators=["+","-","*","/"],
        unary_operators=["sqrt","square","abs"],
        extra_sympy_mappings={"square": lambda x: x**2},
        random_state=seed, deterministic=True, parallelism="serial",
        timeout_in_seconds=TIMEOUT, verbosity=0, temp_equation_file=True,
    )
    print(f"  [seed={seed}] fitting delta (correction-first, no anchor)...", flush=True)
    pysr.fit(X_norm, delta_noisy, variable_names=["v_b"])
    n_arc, n_sqrt, best_s = count_sqrt(pysr.equations_, "v_b", sp.Integer(0))
    n_total = len(pysr.equations_)
    print(f"  [seed={seed}]: {n_arc}/{n_total} ARC-pass, {n_sqrt} sqrt-candidates")
    if best_s:
        print(f"    best sqrt: {best_s[0][:70]}  NMSE={best_s[1]:.6f}")
    arm_b.append({"seed":seed,"n_total":n_total,"n_arc":n_arc,"n_sqrt":n_sqrt,
                  "best_sqrt": best_s[0] if best_s else "NULL"})

# ----------------------------------------------------------------
# ABLATION SUMMARY
# ----------------------------------------------------------------
print("\n" + "=" * 65)
print("ABLATION SUMMARY (all on extended v range 0.05-0.99c)")
print("=" * 65)
print(f"{'':40} {'ARM A':>8} {'ARM B':>8} {'ARM C':>8}")
print(f"{'Setup':40} {'Solo':>8} {'CorrOnly':>8} {'Full ADCD':>8}")
print("-" * 65)

def rate(lst, key):
    return sum(1 for r in lst if r[key] > 0) / len(lst)

# ARM C results from previous run (recorded)
arm_c_sqrt = [0, 3, 1]  # sqrt candidates per seed from extended test
arm_c_arc  = [81.0, 92.9, 66.7]

sqrt_a = rate(arm_a, "n_sqrt")
sqrt_b = rate(arm_b, "n_sqrt")
sqrt_c = sum(1 for x in arm_c_sqrt if x > 0) / len(arm_c_sqrt)

arc_a = sum(r["n_arc"]/max(r["n_total"],1) for r in arm_a) / len(arm_a)
arc_b = sum(r["n_arc"]/max(r["n_total"],1) for r in arm_b) / len(arm_b)
arc_c = sum(arm_c_arc) / len(arm_c_arc) / 100

print(f"{'Seeds finding sqrt':<40} {sqrt_a:>8.1%} {sqrt_b:>8.1%} {sqrt_c:>8.1%}")
print(f"{'ARC pass rate (avg)':<40} {arc_a:>8.1%} {arc_b:>8.1%} {arc_c:>8.1%}")
print()
print("Per-seed (sqrt_candidates):")
for i, seed in enumerate(SEEDS):
    n_c = arm_c_sqrt[i]
    print(f"  seed={seed}: A={arm_a[i]['n_sqrt']}  B={arm_b[i]['n_sqrt']}  C={n_c}")
print()
print("Interpretation:")
print("  If A finds sqrt: extended range alone sufficient (ADCD not needed)")
print("  If only C finds sqrt: ADCD integration is the key contribution")
print("  If B finds sqrt: correction-first alone sufficient")

import os; os.makedirs("results", exist_ok=True)
with open("results/ablation_extended.jsonl","a") as f:
    f.write(json.dumps({"timestamp":TIMESTAMP,"arm_a":arm_a,"arm_b":arm_b,
                        "arm_c_sqrt":arm_c_sqrt,"arm_c_arc":arm_c_arc})+"\n")
print(f"\nAudit: results/ablation_extended.jsonl")
