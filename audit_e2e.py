import sys
sys.path.insert(0, "e:/ADCD/PhysicsPaper/src")
import numpy as np, sympy as sp

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import _run_search
from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig, JuliaEngineData

# ─── AUDIT 1: _run_search ranked tuple format ───────────────────────────────
print("=" * 60)
print("AUDIT 1: _run_search Julia – theta_dict construction")
td = next(s for s in get_all_scenarios() if "Time Dilation" in s.name)
td.engine = "julia"
ranked, space, _ = _run_search(td, exclude_primitives=None, seed=42, n_candidates=20)
print("  space_size:", space, "  ranked:", len(ranked))
for i, (expr_str, nmse, bic, theta_fit) in enumerate(ranked[:3]):
    print(f"  Rank {i+1}: nmse={nmse:.5f} | theta_fit={theta_fit}")

# ─── AUDIT 2: CandidateResult.expr_str ──────────────────────────────────────
print()
print("=" * 60)
print("AUDIT 2: CandidateResult.expr_str (sympy string from Julia)")
td2 = next(s for s in get_all_scenarios() if "Time Dilation" in s.name)
X, y_obs, y_cls, _ = td2.generate_data(n_points=100, noise_level=0.01, seed=42)
cfg = JuliaEngineConfig(
    domain=td2.domain, target_dim="dimensionless",
    input_vars=td2.classical_variables, known_constants=td2.classical_constants,
    bic_threshold=6.0, nmse_coarse=1.0, nmse_fine=0.1,
    n_restarts=5, max_proposals=20, groups=None, excluded_primitives=[]
)
dat = JuliaEngineData(y_classical=y_cls, y_obs=y_obs, vars=X)
res = ADCDJuliaEngine().run(cfg, dat)
for c in res.identifiable[:3]:
    print(f"  description = {c.description}")
    print(f"  expr_str    = {c.expr_str}")
    print(f"  theta       = {c.theta}")

# ─── AUDIT 3: sympy round-trip ───────────────────────────────────────────────
print()
print("=" * 60)
print("AUDIT 3: sympy round-trip – theta substitution correctness")
if res.identifiable:
    best = min(res.identifiable, key=lambda c: c.nmse)
    print(f"  Using best candidate (nmse={best.nmse:.6f}): {best.expr_str}")
    expr_s = sp.sympify(best.expr_str)
    subs_d = {sp.Symbol(f"theta_{i}"): v for i, v in enumerate(best.theta)}
    expr_sub = expr_s.subs(subs_d)
    beta, c_SI = 0.5, 3e8
    val = float(expr_sub.subs({sp.Symbol("v"): beta*c_SI, sp.Symbol("c"): c_SI}))
    expected = 1.0 / np.sqrt(1.0 - beta**2) - 1.0
    print(f"  eval(beta=0.5) = {val:.8f}")
    print(f"  expected gamma-1 = {expected:.8f}")
    print(f"  error = {abs(val-expected):.2e}")
    print(f"  PASS: {abs(val - expected) < 0.05}")
else:
    print("  No identifiable candidates – check engine.")

# ─── AUDIT 4: expr_str double-counted thetas? ────────────────────────────────
print()
print("=" * 60)
print("AUDIT 4: theta index uniqueness – no double-mapping")
for c in res.identifiable[:3]:
    syms = sp.sympify(c.expr_str).free_symbols
    theta_syms = sorted([str(s) for s in syms if str(s).startswith("theta_")])
    n_theta_in_expr = len(theta_syms)
    n_theta_in_list = len(c.theta)
    ok = n_theta_in_expr == n_theta_in_list
    print(f"  {c.description}: thetas_in_expr={theta_syms}  thetas_in_list={n_theta_in_list}  MATCH={ok}")