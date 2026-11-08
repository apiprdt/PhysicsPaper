"""
scale_invariant_screening.py
============================
Scale-invariant screening for ADCD Stage 1 candidate evaluation.

Removes hardcoded dependence on theta=1.0 by:
1. Normalizing each physical variable by its median: x_hat = x / median(|x|).
2. Sweeping a discrete log-grid of dimensionless scales s in [10^-2, 10^2].
3. Solving the optimal amplitude parameter theta_0 in closed form via Weighted Least Squares (WLS).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)

# Canonical discrete log-scale sweep
DEFAULT_SCALE_GRID = np.logspace(-2.0, 2.0, 9)  # 0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0


class ScaleInvariantScreener:
    """
    Evaluates and ranks candidate mathematical structures invariant to unit scaling.
    """
    def __init__(
        self,
        X: Dict[str, np.ndarray],
        y_obs: np.ndarray,
        weights: Optional[np.ndarray] = None,
        scale_grid: Optional[np.ndarray] = None,
    ):
        if not X:
            raise ValueError("Dataset X cannot be empty.")
        self.X_raw = {k: np.asarray(v, dtype=np.float64) for k, v in X.items()}
        self.y_obs = np.asarray(y_obs, dtype=np.float64)
        self.n_points = len(self.y_obs)

        # Variance of target for NMSE denominator
        self.y_var = float(np.var(self.y_obs))
        if self.y_var < 1e-14:
            self.y_var = 1.0

        # Weights for WLS
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64)
        else:
            self.weights = np.ones(self.n_points, dtype=np.float64)

        # Scale sweep grid
        self.scale_grid = np.asarray(scale_grid if scale_grid is not None else DEFAULT_SCALE_GRID, dtype=np.float64)

        # 1. Normalize each variable by its median magnitude
        self.medians: Dict[str, float] = {}
        self.X_norm: Dict[str, np.ndarray] = {}
        for k, v in self.X_raw.items():
            med = float(np.median(np.abs(v)))
            if med < 1e-12:
                med = 1.0
            self.medians[k] = med
            self.X_norm[k] = v / med

    def _extract_primary_var(self, sym_expr: sp.Expr) -> Optional[str]:
        """Identify the primary physical variable in the expression."""
        free_names = [s.name for s in sym_expr.free_symbols if not s.name.startswith("theta_")]
        for k in self.X_raw.keys():
            if k in free_names:
                return k
        return free_names[0] if free_names else None

    def evaluate_candidate(self, expr_str: str) -> Dict[str, Any]:
        """
        Evaluate a candidate expression over the scale grid and solve amplitude in closed form.
        """
        try:
            sym_expr = sp.sympify(expr_str)
        except Exception as e:
            logger.debug(f"Failed to parse expression '{expr_str}': {e}")
            return {
                "expr": expr_str,
                "nmse": float("inf"),
                "rss": float("inf"),
                "theta_0": 0.0,
                "theta_1": 1.0,
                "scale_idx": -1,
            }

        var_name = self._extract_primary_var(sym_expr)
        if not var_name or var_name not in self.X_norm:
            return {
                "expr": expr_str,
                "nmse": float("inf"),
                "rss": float("inf"),
                "theta_0": 0.0,
                "theta_1": 1.0,
                "scale_idx": -1,
            }

        x_hat = self.X_norm[var_name]
        med_val = self.medians[var_name]

        # Identify free parameters
        free_thetas = sorted([s.name for s in sym_expr.free_symbols if s.name.startswith("theta_")])
        has_inner_scale = len(free_thetas) >= 2 or any(s == "theta_1" for s in free_thetas)

        # Form a pure unitless evaluation function D(u)
        # Substitute theta_0 -> 1.0 and (var_name/theta_1) -> u
        u_sym = sp.Symbol("u", real=True)
        # Substitute r/theta_1 -> u
        sub_dict = {}
        for s in sym_expr.free_symbols:
            if s.name == "theta_0":
                sub_dict[s] = 1.0

        expr_unit = sym_expr.subs(sub_dict)
        if has_inner_scale:
            theta_1_sym = sp.Symbol("theta_1", real=True)
            var_sym = sp.Symbol(var_name, real=True)
            expr_unit = expr_unit.subs(var_sym / theta_1_sym, u_sym)
            expr_unit = expr_unit.subs(var_sym, u_sym)
        else:
            var_sym = sp.Symbol(var_name, real=True)
            expr_unit = expr_unit.subs(var_sym, u_sym)

        try:
            f_u = sp.lambdify(u_sym, expr_unit, modules=["numpy"])
        except Exception:
            return {
                "expr": expr_str,
                "nmse": float("inf"),
                "rss": float("inf"),
                "theta_0": 0.0,
                "theta_1": 1.0,
                "scale_idx": -1,
            }

        best_rss = float("inf")
        best_A = 0.0
        best_scale = 1.0
        best_idx = 0

        # Sweep log grid of dimensionless scales s
        scales_to_sweep = self.scale_grid if has_inner_scale else [1.0]

        for idx, s in enumerate(scales_to_sweep):
            u_vals = s * x_hat
            try:
                f_eval = np.asarray(f_u(u_vals), dtype=np.float64)
            except Exception:
                continue

            if not np.all(np.isfinite(f_eval)):
                continue

            # Closed-form Weighted Least Squares for amplitude A:
            # A* = sum(w * f * y) / sum(w * f^2)
            denom = float(np.sum(self.weights * f_eval ** 2))
            if denom < 1e-14:
                continue

            num = float(np.sum(self.weights * f_eval * self.y_obs))
            A_star = num / denom

            pred = A_star * f_eval
            rss = float(np.sum(self.weights * (self.y_obs - pred) ** 2))

            if rss < best_rss:
                best_rss = rss
                best_A = A_star
                best_scale = s
                best_idx = idx

        if not np.isfinite(best_rss):
            nmse = float("inf")
            theta_1_phys = med_val
        else:
            nmse = float(best_rss / (self.n_points * self.y_var))
            # Recover physical scale theta_1:
            # Since u = s * x_hat = s * (x / med) = x / (med / s)
            # Therefore theta_1 = med / s
            theta_1_phys = float(med_val / best_scale)

        return {
            "expr": expr_str,
            "nmse": nmse,
            "rss": best_rss,
            "theta_0": float(best_A),
            "theta_1": theta_1_phys,
            "scale_idx": best_idx,
        }

    def rank_candidates(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """
        Rank candidate formulas by scale-invariant WLS fit error.
        """
        results = [self.evaluate_candidate(cand) for cand in candidates]
        results.sort(key=lambda r: r["rss"])
        return results
