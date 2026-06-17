"""Sequential (independent) ARC checker for multivariable corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import sympy as sp


@dataclass
class SequentialARCResult:
    passes: bool
    vanishing_at: List[str] = field(default_factory=list)
    failing_var: Optional[str] = None


class SequentialARCChecker:
    """
    Check ARC compliance by testing each limit variable independently.

    Unlike simultaneous pre-filtering, other variables stay at midpoint
    while one limit variable is pushed to its extreme.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def check(
        self,
        expr: sp.Expr,
        limit_vars: List[str],
        limit_dirs: List[str],
        constants: Dict[str, float],
        tol: float = 0.05,
        n_samples: int = 10,
    ) -> SequentialARCResult:
        vanishing_at: List[str] = []

        for limit_var, limit_dir in zip(limit_vars, limit_dirs):
            var_passes = False

            for _ in range(n_samples):
                subs: Dict[sp.Basic, float] = {}

                for sym in expr.free_symbols:
                    name = str(sym)
                    if name.startswith("theta"):
                        subs[sym] = float(self._rng.uniform(0.1, 5.0))
                    elif name in constants:
                        subs[sym] = constants[name]
                    elif name == limit_var:
                        subs[sym] = 1e-7 if limit_dir == "0" else 1e8
                    else:
                        subs[sym] = 1.0

                try:
                    val = float(complex(expr.subs(subs)).real)
                    if np.isfinite(val) and abs(val) < tol:
                        var_passes = True
                        break
                except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                    continue

            if not var_passes:
                return SequentialARCResult(
                    passes=False,
                    vanishing_at=vanishing_at,
                    failing_var=limit_var,
                )

            arrow = "∞" if limit_dir == "oo" else limit_dir
            vanishing_at.append(f"{limit_var}→{arrow}")

        return SequentialARCResult(passes=True, vanishing_at=vanishing_at)
