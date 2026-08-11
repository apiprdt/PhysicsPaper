
import sys
sys.path.insert(0, 'src')
import numpy as np
import sympy as sp
import json
from datetime import datetime, timezone
from adcd.arc_gate import ARCGate
from adcd.constrained_pysr import ADCDConstrainedPySR

SEEDS = [0, 42, 123]
TIMEOUT = 200
TIMESTAMP = datetime.now(timezone.utc).isoformat()

print("=" * 65)
print("EXTENDED RANGE TEST: v up to 0.99c")
print("=" * 65)
print(f"Timestamp : {TIMESTAMP}")
print(f"Seeds     : {SEEDS}")
print()
print("Physics motivation:")
print("  At v=0.99c:")
v_test = 0.99
delta_true_test = 1/np.sqrt(1-v_test**2) - 1 - v_test**2/2
taylor4_test = 3/8 * v_test**4
print(f"    True delta = {delta_true_test:.4f}")
print(f"    Taylor 4th = {taylor4_test:.4f}")
print(f"    Ratio = {delta_true_test/taylor4_test:.1f}x  (polynomial misses by {delta_true_test/taylor4_test:.1f}x!)")
print()
print("  -> PySR cannot fit full range with polynomial, MUST use sqrt")
print()

def generate_extended(seed, noise=0.02):
    rng = np.random.default_rng(seed)
    # Extended range: moderate + highly relativistic
    v_mod = rng.uniform(0.05, 0.85, 150)
    v_rel = rng.uniform(0.90, 0.99, 50)
    v = np.concatenate([v_mod, v_rel])
    rng.shuffle(v)
    f0 = 0.5 * v**2
    y_true = 1.0 / np.sqrt(1.0 - v**2) - 1.0
    delta_true = y_true - f0
    sigma = noise * np.std(delta_true)
    y_obs_noisy = y_true + rng.normal(0, sigma, len(v))
    return v.reshape(-1,1), y_obs_noisy, f0, delta_true

arc = ARCGate(target_limit=0.0, tolerance=1e-3)
results = []

for seed in SEEDS:
    print(f"{'='*65}")
    print(f"[seed={seed}] ARM C: ADCD Integrated, extended range")
    X, y_obs, f0, delta_true = generate_extended(seed)
    print(f"  v range: [{X.min():.3f}, {X.max():.3f}]c, n={len(X)}")
    
    model = ADCDConstrainedPySR(
        physics_domain="mechanics",
        arc_n_anchors=15,
        arc_anchor_weight=5.0,
        arc_tolerance=1e-3,
        normalize_variables=True,
        char_scales={"v": 1.0},
        niterations=100,
        maxsize=30,
        populations=30,
        random_state=seed,
        deterministic=True,
        timeout_in_seconds=TIMEOUT,
    )
    
    result = model.fit(X, y_obs, f0, ["v"],
                       limit_var_idx=0, limit_point=0.0)
    
    # Structural check: does best expression contain sqrt?
    best = result.best_expr
    has_sqrt = "sqrt" in best.lower()
    
    # Check all ARC-verified candidates for sqrt
    arc_verified = [g for g in result.gate_results if g.arc_passed]
    sqrt_candidates = [g for g in arc_verified if "sqrt" in g.expr.lower()]
    
    print(f"\n  STRUCTURAL ANALYSIS:")
    print(f"    ARC verified: {result.n_arc_verified}/{result.n_total_candidates} ({result.arc_pass_rate:.1%})")
    print(f"    Candidates containing sqrt: {len(sqrt_candidates)}/{result.n_arc_verified}")
    print(f"    Best ARC-verified: {best}")
    print(f"    Contains sqrt: {has_sqrt}")
    print(f"    NMSE: {result.best_nmse:.8f}")
    
    if sqrt_candidates:
        best_sqrt = min(sqrt_candidates, key=lambda x: x.nmse)
        print(f"\n  BEST SQRT CANDIDATE:")
        print(f"    Expr: {best_sqrt.expr}")
        print(f"    NMSE: {best_sqrt.nmse:.8f}")
        
        # Evaluate structural similarity to 1/sqrt(1-v^2) - 1 - v^2/2
        v_test_pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
        delta_ref = 1/np.sqrt(1 - v_test_pts**2) - 1 - v_test_pts**2/2
        print(f"\n  Extrapolation check (true vs best sqrt candidate):")
        try:
            v_sym = sp.Symbol("v_b")
            expr_sym = sp.sympify(best_sqrt.expr.replace("^","**"))
            delta_pred = np.array([float(expr_sym.subs(v_sym, vv).evalf()) 
                                   for vv in v_test_pts])
            print(f"    v:     {v_test_pts}")
            print(f"    True:  {delta_ref.round(4)}")
            print(f"    Pred:  {delta_pred.round(4)}")
            rel_err = np.abs((delta_pred - delta_ref)/(delta_ref+1e-10))
            print(f"    RelErr:{rel_err.round(3)}")
        except Exception as e:
            print(f"    (eval error: {e})")
    else:
        print(f"\n  No ARC-verified candidates contain sqrt")
    
    results.append({
        "seed": seed,
        "n_total": result.n_total_candidates,
        "n_arc": result.n_arc_verified,
        "arc_rate": result.arc_pass_rate,
        "best_expr": result.best_expr,
        "best_nmse": result.best_nmse,
        "has_sqrt_best": has_sqrt,
        "n_sqrt_candidates": len(sqrt_candidates),
        "best_sqrt_expr": sqrt_candidates[0].expr if sqrt_candidates else "NULL",
        "best_sqrt_nmse": sqrt_candidates[0].nmse if sqrt_candidates else 999,
    })
    print()

# Summary
print("=" * 65)
print("EXTENDED RANGE SUMMARY")
print("=" * 65)
avg_arc = sum(r["arc_rate"] for r in results) / len(results)
n_sqrt = sum(1 for r in results if r["n_sqrt_candidates"] > 0)
print(f"ARC pass rate (avg): {avg_arc:.1%}")
print(f"Seeds finding sqrt:  {n_sqrt}/{len(SEEDS)}")
print()
for r in results:
    print(f"  seed={r['seed']}: arc={r['arc_rate']:.1%}, "
          f"sqrt_candidates={r['n_sqrt_candidates']}, "
          f"best_nmse={r['best_nmse']:.6f}")
    print(f"    best: {r['best_expr'][:70]}")

if n_sqrt > 0:
    print()
    print("REDISCOVERY RESULT: sqrt-containing expressions found!")
    print("Compare with earlier (no extended range): 0 seeds found sqrt")
else:
    print()
    print("NULL RESULT: Extended range still insufficient for sqrt discovery")
    print("Recommendation: increase niterations or add explicit sqrt seeding")

import os; os.makedirs("results", exist_ok=True)
with open("results/rediscovery_extended.jsonl", "a") as f:
    f.write(json.dumps({"timestamp": TIMESTAMP, "results": results}) + "\n")
print(f"\nAudit saved: results/rediscovery_extended.jsonl")
