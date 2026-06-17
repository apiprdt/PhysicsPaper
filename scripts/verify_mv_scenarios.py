"""Verify MV scenarios: theta-dimensionless and per-variable ARC limits."""

import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adcd.anomaly_scenarios import get_mv_scenarios
from adcd.sequential_arc import SequentialARCChecker


def main() -> int:
    checker = SequentialARCChecker(seed=42)
    failures = 0

    for sc in get_mv_scenarios():
        expr = sp.sympify(sc.correction_expr)
        theta_syms = [s for s in expr.free_symbols if str(s).startswith("theta")]
        if not theta_syms:
            print(f"FAIL {sc.name}: no theta symbols")
            failures += 1
            continue

        limit_vars = [v.strip() for v in sc.classical_limit_variable.split(",")]
        limit_dirs = [d.strip() for d in sc.classical_limit_direction.split(",")]
        if len(limit_dirs) < len(limit_vars):
            limit_dirs.extend([limit_dirs[-1]] * (len(limit_vars) - len(limit_dirs)))

        subs_theta = {s: 1.0 for s in theta_syms}
        arc_ok = True
        for var, direction in zip(limit_vars, limit_dirs):
            lim_val = sp.oo if direction == "oo" else 0
            subs = dict(subs_theta)
            for other_var in sc.classical_variables:
                if other_var != var:
                    subs[sp.Symbol(other_var)] = 1.0
            for const_name, const_val in sc.classical_constants.items():
                subs[sp.Symbol(const_name)] = const_val
            lim_result = sp.limit(expr.subs(subs), sp.Symbol(var), lim_val)
            if lim_result != 0:
                print(f"ARC FAIL {sc.name}: var={var} limit={lim_result}")
                arc_ok = False
                failures += 1

        seq = checker.check(
            expr,
            limit_vars=limit_vars,
            limit_dirs=limit_dirs,
            constants=sc.classical_constants,
        )
        if arc_ok and seq.passes:
            print(f"OK {sc.name}: theta-dimensionless OK, ARC per-variable OK")
        elif not seq.passes:
            print(f"FAIL {sc.name}: sequential ARC failed at {seq.failing_var}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
