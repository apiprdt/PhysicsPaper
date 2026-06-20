"""MOND interpolating-function comparison for SPARC stacking experiments.

Provides both zero-parameter (canonical) and two-parameter (fitted) variants
of each MOND interpolating function for fair comparison against the ADCD-discovered
2-parameter form ν(x) = θ₀(√(1 + θ₁/x) − 1) + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from adcd.metrics import bic_score


# ---------------------------------------------------------------------------
# NMSE helper
# ---------------------------------------------------------------------------

def _nmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_obs - y_pred) ** 2) / max(np.var(y_obs), 1e-15))


# ---------------------------------------------------------------------------
# Canonical (zero-free-parameter) formula strings
# ---------------------------------------------------------------------------

SIMPLE_MOND_FORMULA = "(1 + sqrt(1 + 4/x)) / 2"
STANDARD_MOND_FORMULA = "1 / sqrt(1 - exp(-sqrt(x)))"
RAR_FORMULA = "1 / (1 - exp(-sqrt(x)))"


# ---------------------------------------------------------------------------
# Zero-parameter evaluators (original — unchanged)
# ---------------------------------------------------------------------------

def nu_simple_mond(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 + np.sqrt(1.0 + 4.0 / x)) / 2.0


def nu_standard_mond(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / np.sqrt(1.0 - np.exp(-np.sqrt(x)))


def nu_rar(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


# ---------------------------------------------------------------------------
# Two-parameter fitted variants
#
# Each model re-parameterises the canonical form by adding a rescaling of
# the internal acceleration scale (θ₀) and an overall amplitude (θ₁) so that
# every model has exactly 2 free parameters — matching the ADCD form.
#
#   Simple MOND  :  ν = θ₁ · (1 + √(1 + 4/(θ₀·x))) / 2
#   Standard MOND:  ν = θ₁ / √(1 − exp(−√(θ₀·x)))
#   RAR McGaugh  :  ν = θ₁ / (1 − exp(−√(θ₀·x)))
# ---------------------------------------------------------------------------

def nu_simple_mond_2param(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """ν = θ₁ · (1 + √(1 + 4/(θ₀·x))) / 2"""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.sqrt(1.0 + 4.0 / (theta0 * x))
        return theta1 * (1.0 + inner) / 2.0


def nu_standard_mond_2param(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """ν = θ₁ / √(1 − exp(−√(θ₀·x)))"""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        arg = -np.sqrt(theta0 * x)
        arg = np.clip(arg, -500, 500)  # guard against overflow
        return theta1 / np.sqrt(1.0 - np.exp(arg))


def nu_rar_2param(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """ν = θ₁ / (1 − exp(−√(θ₀·x)))"""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        arg = -np.sqrt(theta0 * x)
        arg = np.clip(arg, -500, 500)
        return theta1 / (1.0 - np.exp(arg))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MondModelScore:
    name: str
    formula: str
    nmse: float
    bic: float
    n_params: int
    fitted_theta: Optional[Dict[str, float]] = field(default=None)


@dataclass
class FittedBaselineResult:
    """Container for a single 2-param fitted baseline result."""
    name: str
    formula: str
    theta0: float
    theta1: float
    nmse: float
    bic: float


# ---------------------------------------------------------------------------
# Zero-parameter scoring (original — unchanged)
# ---------------------------------------------------------------------------

def score_mond_models(
    x: np.ndarray,
    nu_obs: np.ndarray,
    adcd_nmse: float | None = None,
    adcd_n_params: int = 2,
) -> List[MondModelScore]:
    """Compare fixed MOND variants (+ optional ADCD) on stacked ν(x) data."""
    x = np.asarray(x, dtype=float)
    nu_obs = np.asarray(nu_obs, dtype=float)
    n = len(nu_obs)

    scores: List[MondModelScore] = []
    for name, formula, pred in [
        ("Simple MOND", SIMPLE_MOND_FORMULA, nu_simple_mond(x)),
        ("Standard MOND", STANDARD_MOND_FORMULA, nu_standard_mond(x)),
        ("RAR (McGaugh)", RAR_FORMULA, nu_rar(x)),
    ]:
        nmse_val = _nmse(nu_obs, pred)
        scores.append(MondModelScore(
            name=name, formula=formula, nmse=nmse_val,
            bic=bic_score(nmse_val, 0, n), n_params=0,
        ))

    if adcd_nmse is not None:
        scores.append(MondModelScore(
            name="ADCD discovered",
            formula="(from iADCD search)",
            nmse=adcd_nmse,
            bic=bic_score(adcd_nmse, adcd_n_params, n),
            n_params=adcd_n_params,
        ))

    scores.sort(key=lambda s: s.bic)
    return scores


# ---------------------------------------------------------------------------
# Two-parameter fitting via L-BFGS-B
# ---------------------------------------------------------------------------

def _fit_2param(
    fn,
    x: np.ndarray,
    nu_obs: np.ndarray,
    name: str,
    formula: str,
    n_restarts: int = 20,
    seed: int = 42,
) -> FittedBaselineResult:
    """Fit a 2-param ν model with multiple random restarts (L-BFGS-B)."""
    rng = np.random.default_rng(seed)
    n = len(nu_obs)

    def objective(log_params: np.ndarray) -> float:
        theta0 = np.exp(log_params[0])  # log-parametrise for positivity
        theta1 = np.exp(log_params[1])
        pred = fn(x, theta0, theta1)
        cost = float(np.mean((nu_obs - pred) ** 2))
        # Return a large finite value on numerical blow-up so L-BFGS-B stays stable
        if not np.isfinite(cost):
            return 1e15
        return cost

    best_cost = np.inf
    best_params = np.array([0.0, np.log(1.0)])  # default: θ₀=1, θ₁=1

    for _ in range(n_restarts):
        # Random initialisation in log-space: θ₀ ∈ [0.01, 100], θ₁ ∈ [0.1, 10]
        x0 = np.array([
            rng.uniform(np.log(0.01), np.log(100.0)),
            rng.uniform(np.log(0.1), np.log(10.0)),
        ])
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           options={"maxiter": 5000, "ftol": 1e-15})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x.copy()
        except Exception:
            continue

    theta0 = float(np.exp(best_params[0]))
    theta1 = float(np.exp(best_params[1]))
    pred = fn(x, theta0, theta1)
    nmse_val = _nmse(nu_obs, pred)
    bic_val = bic_score(nmse_val, 2, n)

    return FittedBaselineResult(
        name=name, formula=formula,
        theta0=theta0, theta1=theta1,
        nmse=nmse_val, bic=bic_val,
    )


def score_fitted_baselines(
    x: np.ndarray,
    nu_obs: np.ndarray,
    adcd_nmse: float | None = None,
    adcd_n_params: int = 2,
    n_restarts: int = 20,
    seed: int = 42,
) -> List[MondModelScore]:
    """Fit each baseline with exactly 2 free parameters (fair comparison).

    Uses L-BFGS-B with multiple random restarts and log-parameterisation
    to ensure θ₀, θ₁ > 0.
    """
    x = np.asarray(x, dtype=float)
    nu_obs = np.asarray(nu_obs, dtype=float)
    n = len(nu_obs)

    models = [
        ("RAR (McGaugh, 2-param)", "θ₁ / (1 - exp(-√(θ₀·x)))", nu_rar_2param),
        ("Simple MOND (2-param)", "θ₁·(1 + √(1 + 4/(θ₀·x))) / 2", nu_simple_mond_2param),
        ("Standard MOND (2-param)", "θ₁ / √(1 - exp(-√(θ₀·x)))", nu_standard_mond_2param),
    ]

    scores: List[MondModelScore] = []

    for name, formula, fn in models:
        fitted = _fit_2param(fn, x, nu_obs, name, formula,
                             n_restarts=n_restarts, seed=seed)
        print(f"  Fitted {name}: θ₀={fitted.theta0:.4f}, θ₁={fitted.theta1:.4f}, "
              f"NMSE={fitted.nmse:.5f}, BIC={fitted.bic:.2f}")
        scores.append(MondModelScore(
            name=fitted.name,
            formula=fitted.formula,
            nmse=fitted.nmse,
            bic=fitted.bic,
            n_params=2,
            fitted_theta={"theta_0": fitted.theta0, "theta_1": fitted.theta1},
        ))

    if adcd_nmse is not None:
        scores.append(MondModelScore(
            name="ADCD discovered",
            formula="(from iADCD search)",
            nmse=adcd_nmse,
            bic=bic_score(adcd_nmse, adcd_n_params, n),
            n_params=adcd_n_params,
        ))

    scores.sort(key=lambda s: s.bic)
    return scores


# ---------------------------------------------------------------------------
# Pretty-printing (supports fitted_theta column)
# ---------------------------------------------------------------------------

def print_mond_comparison(scores: List[MondModelScore]) -> None:
    has_fitted = any(s.fitted_theta for s in scores)
    if has_fitted:
        print("\n=== MOND Model Comparison (lower BIC = better) ===")
        print(f"{'Rank':>4} | {'Model':<24} | {'NMSE':>10} | {'BIC':>10} | {'k':>3} | {'θ₀':>8} | {'θ₁':>8}")
        for i, s in enumerate(scores, 1):
            t0 = f"{s.fitted_theta['theta_0']:.4f}" if s.fitted_theta else "—"
            t1 = f"{s.fitted_theta['theta_1']:.4f}" if s.fitted_theta else "—"
            print(f"{i:>4} | {s.name:<24} | {s.nmse:>10.5f} | {s.bic:>10.2f} | {s.n_params:>3} | {t0:>8} | {t1:>8}")
    else:
        print("\n=== MOND Model Comparison (lower BIC = better) ===")
        print(f"{'Rank':>4} | {'Model':<18} | {'NMSE':>10} | {'BIC':>10} | {'k':>3}")
        for i, s in enumerate(scores, 1):
            print(f"{i:>4} | {s.name:<18} | {s.nmse:>10.5f} | {s.bic:>10.2f} | {s.n_params:>3}")


def scores_to_dict(scores: List[MondModelScore]) -> List[Dict]:
    return [
        {"name": s.name, "formula": s.formula, "nmse": s.nmse, "bic": s.bic,
         "n_params": s.n_params,
         **({"fitted_theta": s.fitted_theta} if s.fitted_theta else {})}
        for s in scores
    ]
