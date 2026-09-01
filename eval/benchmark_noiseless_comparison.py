#!/usr/bin/env python3
"""
benchmark_pysr_comparison.py
==============================================================================
Comparative Benchmark Harness: ADCD vs PySR (Noise Robustness & Extrapolation).

Evaluates structural recovery rate and extrapolation performance across
systematic noise sweeps under identical data, search targets, and feature sets.
Supports ADCD (guided / blind) and PySR (default / dimensionally constrained).
==============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.mode_detection import detect_correction_mode
from adcd.metrics import classify_structure, extended_bic_score
from adcd.run_adcd_v3_validation_blind import (
    DOMAIN_RESTRICTIONS,
    ScenarioThresholdConfig,
    run_scenario_protocol,
)

try:
    import pandas as pd
except ImportError:
    pd = None


# ==============================================================================
# Configuration
# ==============================================================================

NOISE_SWEEP: List[float] = [0.0]
DEFAULT_SEEDS: List[int] = list(range(42, 62))

LOCKED_SCENARIOS: List[str] = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]

DEFAULT_CLEAN_DOMAINS: Dict[str, float] = {
    "Time Dilation": 0.30,
    "Screened Coulomb": 4.0,
    "Entropy Expansion": 3.0,
}

EXTRAP_TRAIN_DOMAIN: Dict[str, float] = {
    "Time Dilation": 0.30,
    "Screened Coulomb": 4.0,
    "Entropy Expansion": 3.0,
}
EXTRAP_TEST_DOMAIN: Dict[str, float] = {
    "Time Dilation": 0.80,
    "Screened Coulomb": 8.0,
    "Entropy Expansion": 6.0,
}

PYSR_BINARY_OPERATORS: List[str] = ["+", "-", "*", "/"]
PYSR_UNARY_OPERATORS: List[str] = ["exp", "log", "sqrt", "sin", "cos"]

MIN_PYSR_SECONDS: float = 30.0
PYSR_DIMENSIONAL_CONSTRAINT_PENALTY: float = 1000.0
_DEGENERATE_THETA_BOUND: float = 1e6
_EXTRAP_SANITY_BOUND: float = 2.0



# ==============================================================================
# Statistical Utilities
# ==============================================================================

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (point_estimate, lower_bound, upper_bound).
    Preferred over normal approximation when proportions approach 0 or 1.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return p, max(0.0, center - half_width), min(1.0, center + half_width)


# ==============================================================================
# Data Layer
# ==============================================================================

@dataclass
class SharedData:
    X: Dict[str, np.ndarray]
    y_obs: np.ndarray
    y_classical: np.ndarray
    target: np.ndarray
    detected_mode: str
    feature_names: List[str]
    X_extrap: Optional[Dict[str, np.ndarray]] = None
    target_extrap: Optional[np.ndarray] = None


def _assert_generate_data_deterministic(scenario, noise: float, seed: int, domain_max: float) -> None:
    """Assert byte-exact reproducibility of data generation across two calls."""
    X1, y1, yc1, _ = scenario.generate_data(noise_level=noise, seed=seed, domain_max=domain_max)
    X2, y2, yc2, _ = scenario.generate_data(noise_level=noise, seed=seed, domain_max=domain_max)
    assert np.array_equal(y1, y2), f"Non-deterministic y_obs for scenario '{scenario.name}'."
    assert np.array_equal(yc1, yc2), f"Non-deterministic y_classical for scenario '{scenario.name}'."


def build_shared_data(
    scenario,
    noise: float,
    seed: int,
    domain_max: float,
    extrap_domain_max: Optional[float] = None,
    extrap_train_domain_max: Optional[float] = None,
) -> SharedData:
    """Build training arrays and optional held-out extrapolation arrays.

    extrap_train_domain_max sets the training domain for the extrapolation
    sub-experiment when it must differ from the recovery-rate domain_max.
    extrap_domain_max must be strictly greater than extrap_train_domain_max.
    """
    _assert_generate_data_deterministic(scenario, noise, seed, domain_max)

    X, y_obs, y_classical, _ = scenario.generate_data(
        noise_level=noise, seed=seed, domain_max=domain_max
    )
    for c_name, c_val in scenario.classical_constants.items():
        if c_name not in X:
            X[c_name] = np.full_like(y_obs, c_val)

    detected_mode, _ = detect_correction_mode(y_obs, y_classical)
    if detected_mode == "multiplicative":
        safe_cl = np.where(np.abs(y_classical) < 1e-15, 1e-15, y_classical)
        target = y_obs / safe_cl - 1.0
    else:
        target = y_obs - y_classical

    feature_names = list(scenario.classical_variables)

    X_extrap, target_extrap = None, None
    if extrap_domain_max is not None:
        effective_train_dmax = extrap_train_domain_max if extrap_train_domain_max is not None else domain_max
        assert extrap_domain_max > effective_train_dmax, (
            f"extrap_domain_max ({extrap_domain_max}) must exceed effective training domain "
            f"({effective_train_dmax}) for '{scenario.name}'. "
            f"Use EXTRAP_TRAIN_DOMAIN to define a narrower training split."
        )
        Xe, ye, yc_e, _ = scenario.generate_data(
            noise_level=0.0, seed=seed + 1000, domain_max=extrap_domain_max
        )
        for c_name, c_val in scenario.classical_constants.items():
            if c_name not in Xe:
                Xe[c_name] = np.full_like(ye, c_val)
        if detected_mode == "multiplicative":
            safe_cl_e = np.where(np.abs(yc_e) < 1e-15, 1e-15, yc_e)
            target_extrap = ye / safe_cl_e - 1.0
        else:
            target_extrap = ye - yc_e

        # Restrict extrapolation evaluation strictly to points outside the training domain
        if scenario.name == "Time Dilation":
            c_val = scenario.classical_constants.get("c", 1.0)
            mask_extrap = Xe["v"] > (effective_train_dmax * c_val)
        elif scenario.name == "Screened Coulomb":
            mask_extrap = Xe["r"] > effective_train_dmax
        elif scenario.name == "Entropy Expansion":
            mask_extrap = (Xe["dV"] / Xe["V_i"]) > effective_train_dmax
        elif hasattr(scenario, "classical_limit_variable") and scenario.classical_limit_variable in Xe:
            var = scenario.classical_limit_variable
            mask_extrap = Xe[var] > effective_train_dmax
        else:
            mask_extrap = np.ones(len(ye), dtype=bool)

        if np.any(mask_extrap):
            Xe = {k: v[mask_extrap] for k, v in Xe.items()}
            target_extrap = target_extrap[mask_extrap]

        X_extrap = Xe

    return SharedData(
        X=X, y_obs=y_obs, y_classical=y_classical, target=target,
        detected_mode=detected_mode, feature_names=feature_names,
        X_extrap=X_extrap, target_extrap=target_extrap,
    )


def _evaluate_expression_on_data(
    expr_str: str,
    X: Dict[str, np.ndarray],
    feature_names: List[str],
    theta_fit: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    """Evaluate a symbolic expression against input arrays.

    Substitutes theta_* symbols before lambdify so that ADCD expressions
    (which carry unfitted theta_0, theta_1, ... as SymPy symbols) can be
    evaluated on data without NameError.
    """
    try:
        expr = sp.sympify(expr_str)
        if theta_fit:
            sub_dict = {
                sp.Symbol(k): float(v)
                for k, v in theta_fit.items()
                if sp.Symbol(k) in expr.free_symbols
            }
            expr = expr.subs(sub_dict)
        all_names = list(X.keys())
        sym_locals = {name: sp.Symbol(name) for name in all_names}
        free_syms = [sym_locals[n] for n in all_names if sp.Symbol(n) in expr.free_symbols]

        unresolved = expr.free_symbols - set(free_syms) - {sp.pi, sp.E, sp.I}
        if unresolved:
            return None

        n_samples = len(next(iter(X.values())))
        if not free_syms:
            val = float(expr)
            return np.full(n_samples, val)

        fn = sp.lambdify(free_syms, expr, modules=["numpy"])
        args = [X[str(s)] for s in free_syms]
        result = np.asarray(fn(*args), dtype=float)
        if result.ndim == 0:
            result = np.full(n_samples, float(result))
        if not np.all(np.isfinite(result)):
            return None
        return result
    except Exception:
        return None


def _compute_nmse_extrap(
    expr_str: str,
    data: SharedData,
    theta_fit: Optional[Dict[str, float]] = None,
) -> float:
    """NMSE on the held-out extrapolation domain. Returns inf if unavailable."""
    if data.X_extrap is None or data.target_extrap is None or not expr_str:
        return float("inf")
    pred = _evaluate_expression_on_data(expr_str, data.X_extrap, data.feature_names, theta_fit)
    if pred is None:
        return float("inf")
    y_true = data.target_extrap
    var_y = (float(np.var(y_true, ddof=1)) if len(y_true) > 1 else float(np.var(y_true))) + 1e-300
    return float(np.mean((pred - y_true) ** 2) / var_y)


def _extract_numeric_constants_as_theta_fit(expr: sp.Expr) -> Dict[str, float]:
    """Extract non-integer numeric literals as a theta_fit dict for classify_structure."""
    theta_fit: Dict[str, float] = {}
    idx = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Number) and not node.is_Integer:
            theta_fit[f"theta_{idx}"] = float(node)
            idx += 1
    return theta_fit


def _is_physically_sane(theta_fit: Dict[str, Any], bound: float = _DEGENERATE_THETA_BOUND) -> bool:
    """Check if fitted parameter magnitudes remain within physical bounds."""
    for v in theta_fit.values():
        try:
            if abs(float(v)) > bound:
                return False
        except (TypeError, ValueError):
            pass
    return True


# ==============================================================================
# Tier-2 Unit Resolution
# ==============================================================================

def build_units_for_scenario(
    scenario,
    feature_names: List[str],
    detected_mode: str,
) -> Optional[Tuple[List[str], str]]:
    """Extract input and target physical units for PySR dimensional constraints."""
    units_map = getattr(scenario, "variables_with_units", None)
    if not units_map or detected_mode != "multiplicative":
        return None

    x_units = []
    for name in feature_names:
        u = units_map.get(name)
        if u is None:
            return None
        x_units.append(u)

    y_units = ""  # Multiplicative target is dimensionless
    return x_units, y_units


# ==============================================================================
# ADCD Execution
# ==============================================================================

def run_adcd_once(
    scenario,
    noise: float,
    seed: int,
    domain_max: float,
    engine: str,
    use_taxonomy_prior: bool = True,
    data: Optional["SharedData"] = None,
) -> Dict[str, Any]:
    """Run ADCD discovery protocol with or without taxonomy domain prior."""
    sc = copy.deepcopy(scenario)
    sc.engine = engine
    t_cfg = ScenarioThresholdConfig.for_scenario(sc, noise_level=noise)
    t0 = time.time()
    res = run_scenario_protocol(
        scenario=sc,
        seed=seed,
        threshold_cfg=t_cfg,
        noise_level=noise,
        domain_max=domain_max,
        use_taxonomy_prior=use_taxonomy_prior,
    )
    elapsed = time.time() - t0

    ps = res.checks.get("primary_search", {})
    match_level = ps.get("match_level", "none")
    theta_fit = ps.get("theta_fit", {})
    expr_str = ps.get("top_candidate", "")
    is_match_structural = match_level in ("exact", "class_only")

    nmse_extrap = _compute_nmse_extrap(expr_str, data, theta_fit=theta_fit) if data is not None else float("inf")

    theta_sane = _is_physically_sane(theta_fit)
    if data is not None and data.X_extrap is not None:
        extrap_sane = math.isfinite(nmse_extrap) and nmse_extrap < _EXTRAP_SANITY_BOUND
    else:
        extrap_sane = True
    is_match = is_match_structural and theta_sane and extrap_sane

    label = "ADCD" if use_taxonomy_prior else "ADCD-blind"
    return {
        "method": label,
        "tier": res.tier,
        "is_match": is_match,
        "degenerate_fit": is_match_structural and not (theta_sane and extrap_sane),
        "match_level": match_level,
        "discovered_class": ps.get("discovered_class", "unknown"),
        "nmse_train": ps.get("nmse"),
        "expr_str": expr_str,
        "theta_fit": theta_fit,
        "elapsed_seconds": elapsed,
        "nmse_extrap": nmse_extrap,
    }


# ==============================================================================
# PySR Execution
# ==============================================================================

def _count_free_params_in_pysr_expr(expr_str: str, feature_names: List[str]) -> int:
    """Count free parameters in a PySR expression for BIC reselection."""
    try:
        expr = sp.sympify(expr_str)
        feature_syms = {sp.Symbol(n) for n in feature_names}
        known_consts = {sp.pi, sp.E, sp.I}
        free_syms = expr.free_symbols - feature_syms - known_consts
        n_symbol_params = len(free_syms)
        numeric_consts = set()
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Number) and not node.is_Integer:
                numeric_consts.add(float(node))
        return max(1, n_symbol_params + len(numeric_consts))
    except Exception:
        return 1


def _run_pysr_core(
    scenario,
    data: SharedData,
    seed: int,
    timeout_seconds: float,
    method_label: str,
    units: Optional[Tuple[List[str], str]] = None,
) -> Dict[str, Any]:
    """Execute PySR regression search on residual data."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError("PySR is not installed.")
    if pd is None:
        raise ImportError("pandas is required for PySR.")

    X_df = pd.DataFrame({name: data.X[name] for name in data.feature_names})
    y = np.asarray(data.target, dtype=float)

    constructor_kwargs: Dict[str, Any] = dict(
        binary_operators=PYSR_BINARY_OPERATORS,
        unary_operators=PYSR_UNARY_OPERATORS,
        model_selection="best",
        timeout_in_seconds=timeout_seconds,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )

    fit_kwargs: Dict[str, Any] = {}
    if units is not None:
        x_units, y_units = units
        constructor_kwargs["dimensional_constraint_penalty"] = PYSR_DIMENSIONAL_CONSTRAINT_PENALTY
        fit_kwargs["X_units"] = x_units
        fit_kwargs["y_units"] = y_units

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PySRRegressor(**constructor_kwargs)
        model.fit(X_df, y, **fit_kwargs)
    elapsed = time.time() - t0

    try:
        best_expr_sympy = model.sympy()
        best_expr_str = str(best_expr_sympy)
    except Exception:
        best_expr_sympy = sp.sympify("0")
        best_expr_str = "0"

    theta_fit = _extract_numeric_constants_as_theta_fit(best_expr_sympy)
    discovered_class = classify_structure(best_expr_sympy, theta_fit=theta_fit)
    is_match_structural = discovered_class == scenario.correction_class
    # Structural class match evaluated directly from functional form
    is_match = is_match_structural

    try:
        pred = model.predict(X_df)
        var_y = float(np.var(y)) + 1e-300
        nmse_train = float(np.mean((y - pred) ** 2) / var_y)
    except Exception:
        nmse_train = float("nan")

    return {
        "method": method_label,
        "is_match": is_match,
        "degenerate_fit": is_match_structural and not theta_sane,
        "discovered_class": discovered_class,
        "nmse_train": nmse_train,
        "expr_str": best_expr_str,
        "elapsed_seconds": elapsed,
        "n_equations_in_pareto": len(model.equations_) if hasattr(model, "equations_") else None,
        "_model_equations": model.equations_ if hasattr(model, "equations_") else None,
        "_feature_names": data.feature_names,
    }


def run_pysr_once(scenario, data: SharedData, seed: int, timeout_seconds: float) -> Dict[str, Any]:
    """Tier-1: PySR default, no unit information."""
    return _run_pysr_core(scenario, data, seed, timeout_seconds, method_label="PySR", units=None)


def run_pysr_tier2_once(scenario, data: SharedData, seed: int, timeout_seconds: float) -> Dict[str, Any]:
    """Tier-2: PySR + dimensional_constraint_penalty.

    Returns a skip record (is_match=None, note=...) if units are unavailable
    for this scenario/mode combination rather than raising an exception.
    """
    units = build_units_for_scenario(scenario, data.feature_names, data.detected_mode)
    if units is None:
        return {
            "method": "PySR+units",
            "is_match": None,
            "discovered_class": "unknown",
            "note": "units_unavailable_or_additive_mode_unsupported",
        }
    return _run_pysr_core(scenario, data, seed, timeout_seconds, method_label="PySR+units", units=units)


def run_pysr_bic_reselect(
    scenario, data: SharedData, pysr_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Reselect best PySR Pareto candidate by extended BIC (Phase-2 ablation)."""
    eqs = pysr_result.get("_model_equations")
    feature_names = pysr_result.get("_feature_names", data.feature_names)

    if eqs is None or len(eqs) == 0:
        return {"method": "PySR+ADCD_BIC", "is_match": False, "discovered_class": "unknown",
                "note": "no_pareto_front"}

    n_points = len(data.target)
    n_candidates = len(eqs)
    var_y = float(np.var(data.target)) + 1e-300

    best_bic, best_expr_str = None, None
    for _, row in eqs.iterrows():
        try:
            loss = float(row["loss"])
            nmse = loss / var_y
            expr_str_row = str(row.get("sympy_format", row.get("equation", "0")))
            n_params = _count_free_params_in_pysr_expr(expr_str_row, feature_names)
            b = extended_bic_score(nmse, n_params, n_points, n_candidates=n_candidates)
            if best_bic is None or b < best_bic:
                best_bic, best_expr_str = b, expr_str_row
        except Exception:
            continue

    if best_expr_str is None:
        return {"method": "PySR+ADCD_BIC", "is_match": False, "discovered_class": "unknown",
                "note": "reselection_failed"}

    try:
        best_expr_sympy = sp.sympify(best_expr_str)
        theta_fit = _extract_numeric_constants_as_theta_fit(best_expr_sympy)
        discovered_class = classify_structure(best_expr_sympy, theta_fit=theta_fit)
    except Exception:
        discovered_class = "unknown"

    return {
        "method": "PySR+ADCD_BIC",
        "is_match": discovered_class == scenario.correction_class,
        "discovered_class": discovered_class,
        "expr_str": best_expr_str,
        "bic": best_bic,
    }


# ==============================================================================
# Experiment Orchestration
# ==============================================================================

def run_one_combination(
    scenario_name: str,
    noise: float,
    seed: int,
    engine: str,
    domain_max: float,
    extrap_domain_max: Optional[float],
    extrap_train_domain_max: Optional[float],
    min_pysr_seconds: float,
    include_ablation: bool,
    run_adcd_guided: bool = True,
    run_adcd_blind: bool = True,
    run_tier1: bool = True,
    run_tier2: bool = False,
) -> Dict[str, Any]:
    """Run comparative benchmark arms on identical data sample."""
    scenarios = {s.name: s for s in get_all_scenarios()}
    scenario = scenarios[scenario_name]

    data = build_shared_data(
        scenario, noise, seed, domain_max,
        extrap_domain_max=extrap_domain_max,
        extrap_train_domain_max=extrap_train_domain_max,
    )

    adcd_guided_res: Dict[str, Any] = {
        "method": "ADCD", "is_match": None, "elapsed_seconds": 0.0,
        "nmse_extrap": float("inf"),
    }
    adcd_blind_res: Dict[str, Any] = {
        "method": "ADCD-blind", "is_match": None, "elapsed_seconds": 0.0,
        "nmse_extrap": float("inf"),
    }

    if run_adcd_guided:
        adcd_guided_res = run_adcd_once(
            scenario, noise, seed, domain_max, engine,
            use_taxonomy_prior=True, data=data,
        )
    if run_adcd_blind:
        adcd_blind_res = run_adcd_once(
            scenario, noise, seed, domain_max, engine,
            use_taxonomy_prior=False, data=data,
        )

    # PySR timeout matches ADCD runtime (floored at min_pysr_seconds)
    adcd_wall = max(
        adcd_guided_res.get("elapsed_seconds", 0.0),
        adcd_blind_res.get("elapsed_seconds", 0.0),
    )
    pysr_timeout = max(adcd_wall, min_pysr_seconds)

    pysr_res: Dict[str, Any] = {"method": "PySR", "is_match": None}
    if run_tier1:
        try:
            pysr_res = run_pysr_once(scenario, data, seed, timeout_seconds=pysr_timeout)
        except Exception as exc:
            pysr_res = {"method": "PySR", "is_match": None, "error": str(exc)}

    tier2_res: Optional[Dict[str, Any]] = None
    if run_tier2:
        try:
            tier2_res = run_pysr_tier2_once(scenario, data, seed, timeout_seconds=pysr_timeout)
        except Exception as exc:
            tier2_res = {"method": "PySR+units", "is_match": None, "error": str(exc)}

    if run_tier1 and pysr_res.get("is_match") is not None:
        pysr_nmse_extrap = _compute_nmse_extrap(pysr_res.get("expr_str", ""), data)
        pysr_res["nmse_extrap"] = pysr_nmse_extrap
        is_match_structural = pysr_res.get("discovered_class") == scenario.correction_class
        theta_sane = not pysr_res.get("degenerate_fit", False)
        if data.X_extrap is not None:
            extrap_sane = math.isfinite(pysr_nmse_extrap) and pysr_nmse_extrap < _EXTRAP_SANITY_BOUND
        else:
        # Candidate match requires both structural class alignment and finite extrapolation sanity
        pysr_res["is_match"] = is_match_structural and extrap_sane
        pysr_res["degenerate_fit"] = is_match_structural and not extrap_sane
    else:
        pysr_res.setdefault("nmse_extrap", float("inf"))

    if tier2_res is not None:
        if tier2_res.get("is_match") is not None:
            t2_nmse_extrap = _compute_nmse_extrap(tier2_res.get("expr_str", ""), data)
            tier2_res["nmse_extrap"] = t2_nmse_extrap
            if tier2_res.get("is_match") is True:
                if data.X_extrap is not None:
                    extrap_sane = math.isfinite(t2_nmse_extrap) and t2_nmse_extrap < _EXTRAP_SANITY_BOUND
                else:
                    extrap_sane = True
                if not extrap_sane:
                    tier2_res["is_match"] = False
                    tier2_res["degenerate_fit"] = True
        else:
            tier2_res.setdefault("nmse_extrap", float("inf"))

    ablation_res = None
    if include_ablation and run_tier1 and pysr_res.get("is_match") is not None:
        ablation_res = run_pysr_bic_reselect(scenario, data, pysr_res)
        if ablation_res:
            ab_expr = ablation_res.get("expr_str", "")
            ab_nmse_extrap = _compute_nmse_extrap(ab_expr, data)
            ablation_res["nmse_extrap"] = ab_nmse_extrap

            try:
                ab_sympy = sp.sympify(ab_expr)
                ab_theta_fit = _extract_numeric_constants_as_theta_fit(ab_sympy)
            except Exception:
                ab_theta_fit = {}

            ab_theta_sane = _is_physically_sane(ab_theta_fit)
            if data.X_extrap is not None:
                ab_extrap_sane = math.isfinite(ab_nmse_extrap) and ab_nmse_extrap < _EXTRAP_SANITY_BOUND
            else:
                ab_extrap_sane = True

            if ablation_res.get("is_match") is True:
                if not (ab_theta_sane and ab_extrap_sane):
                    ablation_res["is_match"] = False
                    ablation_res["degenerate_fit"] = True

    pysr_res.pop("_model_equations", None)
    pysr_res.pop("_feature_names", None)
    if tier2_res is not None:
        tier2_res.pop("_model_equations", None)
        tier2_res.pop("_feature_names", None)

    return {
        "scenario": scenario_name,
        "noise": noise,
        "seed": seed,
        "pysr_timeout_seconds_used": pysr_timeout,
        "adcd": adcd_guided_res,
        "adcd_blind": adcd_blind_res,
        "pysr": pysr_res,
        "pysr_tier2": tier2_res,
        "ablation_pysr_adcd_bic": ablation_res,
    }


# ==============================================================================
# Aggregation
# ==============================================================================

def _extrap_stats(values: List[float]) -> Dict[str, float]:
    """Compute median and max of finite extrapolation NMSE values."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"median": float("inf"), "max": float("inf"), "n_finite": 0}
    return {
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "n_finite": len(finite),
    }


def aggregate_summary(
    all_results: List[Dict[str, Any]],
    targets: List[str],
    noise_sweep: List[float],
    include_ablation: bool,
    include_tier2: bool = False,
    include_blind: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate per-run results into per-(scenario, noise) statistics."""
    summary: Dict[str, List[Dict[str, Any]]] = {}
    for sc_name in targets:
        rows = []
        for noise in noise_sweep:
            subset = [
                r for r in all_results
                if r["scenario"] == sc_name and abs(r["noise"] - noise) < 1e-9
            ]
            n = len(subset)

            a_succ = sum(1 for r in subset if r["adcd"].get("is_match") is True)
            p_succ = sum(1 for r in subset if r["pysr"].get("is_match") is True)
            a_p, a_lo, a_hi = wilson_interval(a_succ, n)
            p_p, p_lo, p_hi = wilson_interval(p_succ, n)

            row: Dict[str, Any] = {
                "noise": noise, "n": n,
                "adcd_recovery": a_p, "adcd_ci95": [a_lo, a_hi],
                "adcd_nmse_extrap": _extrap_stats(
                    [r["adcd"].get("nmse_extrap", float("inf")) for r in subset]
                ),
                "adcd_degenerate_count": sum(
                    1 for r in subset if r["adcd"].get("degenerate_fit", False)
                ),
                "pysr_recovery": p_p, "pysr_ci95": [p_lo, p_hi],
                "pysr_nmse_extrap": _extrap_stats(
                    [r["pysr"].get("nmse_extrap", float("inf")) for r in subset]
                ),
            }

            if include_blind:
                blind_succ = sum(
                    1 for r in subset if r.get("adcd_blind", {}).get("is_match") is True
                )
                b_p, b_lo, b_hi = wilson_interval(blind_succ, n)
                row["adcd_blind_recovery"] = b_p
                row["adcd_blind_ci95"] = [b_lo, b_hi]
                row["adcd_blind_nmse_extrap"] = _extrap_stats(
                    [r.get("adcd_blind", {}).get("nmse_extrap", float("inf")) for r in subset]
                )

            if include_tier2:
                # Only count seeds where Tier-2 actually ran (is_match is not None).
                t2_valid = [r for r in subset
                            if r.get("pysr_tier2") and r["pysr_tier2"].get("is_match") is not None]
                n_t2 = len(t2_valid)
                t2_succ = sum(1 for r in t2_valid if r["pysr_tier2"]["is_match"] is True)
                t2_p, t2_lo, t2_hi = wilson_interval(t2_succ, n_t2) if n_t2 > 0 else (0.0, 0.0, 0.0)
                row["tier2_recovery"] = t2_p
                row["tier2_ci95"] = [t2_lo, t2_hi]
                row["tier2_n"] = n_t2
                row["tier2_nmse_extrap"] = _extrap_stats(
                    [r["pysr_tier2"].get("nmse_extrap", float("inf")) for r in t2_valid]
                )

            if include_ablation:
                ab_succ = sum(
                    1 for r in subset
                    if r.get("ablation_pysr_adcd_bic") and
                    r["ablation_pysr_adcd_bic"].get("is_match") is True
                )
                ab_p, ab_lo, ab_hi = wilson_interval(ab_succ, n)
                row["ablation_recovery"] = ab_p
                row["ablation_ci95"] = [ab_lo, ab_hi]

            rows.append(row)
        summary[sc_name] = rows
    return summary


def merge_tier2_into_summary(
    summary: Dict[str, List[Dict[str, Any]]],
    tier2_payload: Dict[str, Any],
) -> None:
    """Merge a separate Tier-2 JSON report into an existing Tier-1 summary in-place.

    Matched by (scenario, noise) float comparison, not list index, so minor
    differences in noise_sweep ordering between runs are handled safely.
    """
    t2_summary = tier2_payload.get("summary", {})
    for sc_name, rows in summary.items():
        t2_rows = {r["noise"]: r for r in t2_summary.get(sc_name, [])}
        for row in rows:
            t2_row = t2_rows.get(row["noise"])
            if t2_row is None:
                # Tier-2 data missing for this noise point; insert null placeholders.
                row["tier2_recovery"] = None
                row["tier2_ci95"] = [0.0, 0.0]
                row["tier2_n"] = 0
                row["tier2_nmse_extrap"] = {"median": float("inf"), "max": float("inf"), "n_finite": 0}
            else:
                row["tier2_recovery"] = t2_row.get("tier2_recovery", t2_row.get("pysr_recovery"))
                row["tier2_ci95"] = t2_row.get("tier2_ci95", t2_row.get("pysr_ci95", [0.0, 0.0]))
                row["tier2_n"] = t2_row.get("tier2_n", t2_row.get("n", 0))
                row["tier2_nmse_extrap"] = t2_row.get(
                    "tier2_nmse_extrap",
                    {"median": float("inf"), "max": float("inf"), "n_finite": 0}
                )


# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(
    summary: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
    include_ablation: bool,
    include_tier2: bool = False,
    include_blind: bool = True,
) -> None:
    """Generate recovery rate and extrapolation NMSE figures (PDF and PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot generation.")
        return

    scenarios = list(summary.keys())
    n_panels = len(scenarios)

    COLORS = {
        "adcd":       "#1f77b4",
        "adcd_blind": "#17becf",
        "pysr":       "#ff7f0e",
        "tier2":      "#9467bd",
        "ablation":   "#2ca02c",
    }

    # Recovery rate figure
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, sc_name in zip(axes, scenarios):
        rows = summary[sc_name]
        noise_x = [r["noise"] for r in rows]

        def _plot_arm(rates, lo, hi, color, label, marker, ls, ax=ax, noise_x=noise_x):
            ax.plot(noise_x, rates, marker + ls, color=color, lw=2, label=label, zorder=4)
            ax.fill_between(noise_x, lo, hi, alpha=0.18, color=color)

        _plot_arm(
            [r["adcd_recovery"] * 100 for r in rows],
            [r["adcd_ci95"][0] * 100 for r in rows],
            [r["adcd_ci95"][1] * 100 for r in rows],
            COLORS["adcd"], "ADCD (guided)", "o", "-",
        )
        if include_blind and "adcd_blind_recovery" in rows[0]:
            _plot_arm(
                [r["adcd_blind_recovery"] * 100 for r in rows],
                [r["adcd_blind_ci95"][0] * 100 for r in rows],
                [r["adcd_blind_ci95"][1] * 100 for r in rows],
                COLORS["adcd_blind"], "ADCD (blind)", "v", "-.",
            )
        _plot_arm(
            [r["pysr_recovery"] * 100 for r in rows],
            [r["pysr_ci95"][0] * 100 for r in rows],
            [r["pysr_ci95"][1] * 100 for r in rows],
            COLORS["pysr"], "PySR (default)", "s", "--",
        )
        if include_tier2 and rows[0].get("tier2_recovery") is not None:
            _plot_arm(
                [r.get("tier2_recovery") * 100 if r.get("tier2_recovery") is not None else float("nan")
                 for r in rows],
                [r.get("tier2_ci95", [0, 0])[0] * 100 for r in rows],
                [r.get("tier2_ci95", [0, 0])[1] * 100 for r in rows],
                COLORS["tier2"], "PySR + units", "D", "-.",
            )
        if include_ablation and "ablation_recovery" in rows[0]:
            _plot_arm(
                [r.get("ablation_recovery", 0) * 100 for r in rows],
                [r.get("ablation_ci95", [0, 0])[0] * 100 for r in rows],
                [r.get("ablation_ci95", [0, 0])[1] * 100 for r in rows],
                COLORS["ablation"], "PySR + ADCD-BIC", "^", ":",
            )

        ax.set_xlim(min(noise_x) * 0.9, max(noise_x) * 1.05)
        ax.set_ylim(-5, 105)
        ax.set_xlabel("Noise level (fraction)", fontsize=10)
        ax.set_title(sc_name, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7.5, loc="lower left")

    axes[0].set_ylabel("Structural recovery rate (%)", fontsize=10)
    fig.suptitle(
        "ADCD vs PySR: Structural Recovery Rate vs Noise Level\n"
        "(shaded bands = Wilson 95% CI, n=5 seeds per point)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_comparison_recovery.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] Recovery figure saved to {out_dir}/fig_comparison_recovery.{{pdf,png}}")

    # Extrapolation NMSE figure (log scale, median + max per arm)
    has_extrap = any(
        math.isfinite(row.get("adcd_nmse_extrap", {}).get("median", float("inf")))
        for rows in summary.values() for row in rows
    )
    if not has_extrap:
        return

    fig2, axes2 = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0), sharey=True)
    if n_panels == 1:
        axes2 = [axes2]

    for ax, sc_name in zip(axes2, scenarios):
        rows = summary[sc_name]
        noise_x = [r["noise"] for r in rows]

        def _get_stat(key, stat, rows_=rows):
            return [r.get(key, {}).get(stat, float("nan")) for r in rows_]

        for key, color, label in [
            ("adcd_nmse_extrap",       COLORS["adcd"],       "ADCD (guided)"),
            ("adcd_blind_nmse_extrap", COLORS["adcd_blind"], "ADCD (blind)"),
            ("pysr_nmse_extrap",       COLORS["pysr"],       "PySR (default)"),
        ]:
            if not include_blind and "blind" in key:
                continue
            med = [v if math.isfinite(v) else float("nan") for v in _get_stat(key, "median")]
            mx  = [v if math.isfinite(v) else float("nan") for v in _get_stat(key, "max")]
            ax.semilogy(noise_x, med, "o-", color=color, lw=2, label=label + " (median)", zorder=3)
            ax.semilogy(noise_x, mx,  "x:", color=color, lw=1, label=label + " (max)", zorder=2)

        ax.axhline(y=1.0, color="gray", linestyle=":", lw=1, label="NMSE = 1 (chance)")
        ax.set_xlabel("Noise level (fraction)", fontsize=10)
        ax.set_title(sc_name, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7.5)

    axes2[0].set_ylabel("Extrapolation NMSE (log scale)", fontsize=10)
    fig2.suptitle(
        "Extrapolation NMSE — out-of-training domain (lower = better)\n"
        "solid = median across seeds, dashed = worst-case (max) per noise level",
        fontsize=10, y=1.02,
    )
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(out_dir, f"fig_comparison_extrap.{ext}"),
                     bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"[OK] Extrapolation figure saved to {out_dir}/fig_comparison_extrap.{{pdf,png}}")


# ==============================================================================
# Console Reporting
# ==============================================================================

def print_summary_table(
    summary: Dict[str, List[Dict[str, Any]]],
    include_ablation: bool,
    include_tier2: bool = False,
    include_blind: bool = True,
) -> None:
    """Print recovery rate table with extrapolation NMSE (median / max)."""
    for sc_name, rows in summary.items():
        n = rows[0]["n"]
        print("\n" + "=" * 110)
        print(f" STRUCTURAL RECOVERY RATE vs NOISE — {sc_name}  (n={n} seeds per point)")
        print("-" * 110)

        header = f"{'Noise':<8} | {'ADCD (guided)':<26} | {'PySR (default)':<26}"
        if include_blind and "adcd_blind_recovery" in rows[0]:
            header += f" | {'ADCD (blind)':<26}"
        if include_tier2 and rows[0].get("tier2_recovery") is not None:
            header += f" | {'PySR+units':<26}"
        if include_ablation and "ablation_recovery" in rows[0]:
            header += f" | {'PySR+ADCD-BIC':<26}"
        header += f" | {'ADCD extrap (med/max)':<24} | {'PySR extrap (med/max)':<24}"
        print(header)
        print("-" * 110)

        for row in rows:
            def _fmt_recovery(rate, ci):
                return f"{rate*100:5.0f}% [{ci[0]*100:4.0f}-{ci[1]*100:4.0f}%]"

            def _fmt_extrap(stats):
                if stats.get("n_finite", 0) == 0:
                    return "N/A"
                med = stats.get("median", float("inf"))
                mx  = stats.get("max", float("inf"))
                return f"{med:.3f}/{mx:.3f}"

            line = (
                f"{row['noise']:<8.2f} | "
                f"{_fmt_recovery(row['adcd_recovery'], row['adcd_ci95']):<26} | "
                f"{_fmt_recovery(row['pysr_recovery'], row['pysr_ci95']):<26}"
            )
            if include_blind and "adcd_blind_recovery" in row:
                line += f" | {_fmt_recovery(row['adcd_blind_recovery'], row['adcd_blind_ci95']):<26}"
            if include_tier2 and row.get("tier2_recovery") is not None:
                line += f" | {_fmt_recovery(row['tier2_recovery'], row['tier2_ci95']):<26}"
            if include_ablation and "ablation_recovery" in row:
                line += f" | {_fmt_recovery(row['ablation_recovery'], row['ablation_ci95']):<26}"
            line += (
                f" | {_fmt_extrap(row.get('adcd_nmse_extrap', {})):<24}"
                f" | {_fmt_extrap(row.get('pysr_nmse_extrap', {})):<24}"
            )
            print(line)

    print("=" * 110)


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparative Benchmark Harness: ADCD vs PySR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", choices=["python", "julia"], default="julia")
    parser.add_argument("--scenario", default="all",
                        help="Scenario name or 'all' for all locked scenarios.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--noise-sweep", type=float, nargs="+", default=NOISE_SWEEP)
    parser.add_argument("--min-pysr-seconds", type=float, default=MIN_PYSR_SECONDS)
    parser.add_argument("--include-ablation", action="store_true",
                        help="Run PySR+ADCD_BIC reselection (Phase-2 ablation).")
    parser.add_argument("--include-tier2", action="store_true",
                        help="Run Tier-2 (PySR+units) in the same pass as Tier-1.")
    parser.add_argument("--tier2-only", action="store_true",
                        help="Run only Tier-2; skip ADCD and Tier-1.")
    parser.add_argument("--no-extrap", action="store_true",
                        help="Disable extrapolation evaluation.")
    parser.add_argument("--no-blind", action="store_true",
                        help="Skip ADCD-blind arm (use_taxonomy_prior=False).")
    parser.add_argument("--out", default=None,
                        help="Output JSON path.")
    parser.add_argument("--plot-only", default=None, metavar="JSON_PATH",
                        help="Regenerate figures from existing JSON without running search.")
    parser.add_argument("--merge-tier2", default=None, metavar="TIER2_JSON",
                        help="With --plot-only: splice a Tier-2 JSON into the summary before plotting.")
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        "run_outputs",
        "pysr_comparison_tier2.json" if args.tier2_only else "pysr_comparison.json",
    )
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        with open(args.plot_only, "r", encoding="utf-8") as f:
            saved = json.load(f)
        summary = saved.get("summary", {})
        cfg = saved.get("config", {})
        include_ablation = cfg.get("include_ablation", False)
        include_tier2 = bool(args.merge_tier2)
        include_blind = cfg.get("include_blind", True)

        if args.merge_tier2:
            with open(args.merge_tier2, "r", encoding="utf-8") as f:
                tier2_saved = json.load(f)
            merge_tier2_into_summary(summary, tier2_saved)

        print_summary_table(summary, include_ablation, include_tier2, include_blind)
        plot_results(summary, out_dir, include_ablation, include_tier2, include_blind)
        return 0

    import importlib.util
    if importlib.util.find_spec("pysr") is None:
        print("[ERROR] PySR is not installed.", file=sys.stderr)
        return 1

    targets = LOCKED_SCENARIOS if args.scenario == "all" else [args.scenario]

    run_adcd_guided = not args.tier2_only
    run_adcd_blind  = not args.tier2_only and not args.no_blind
    run_tier1       = not args.tier2_only
    run_tier2       = args.tier2_only or args.include_tier2
    include_blind   = run_adcd_blind

    all_results: List[Dict[str, Any]] = []
    completed_keys = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            if isinstance(prev_data, dict) and "raw_runs" in prev_data:
                for r in prev_data["raw_runs"]:
                    if "scenario" in r and "noise" in r and "seed" in r:
                        all_results.append(r)
                        completed_keys.add((r["scenario"], round(float(r["noise"]), 6), int(r["seed"])))
                print(f"[CHECKPOINT] Resuming from {len(all_results)} existing runs in {out_path}")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint from {out_path}: {e}")

    total = len(targets) * len(args.noise_sweep) * len(args.seeds)

    mode_label = "TIER2-ONLY" if args.tier2_only else (
        "ADCD-guided+ADCD-blind+Tier1" + ("+Tier2" if run_tier2 else "")
    )
    print("=" * 110)
    print(f" ADCD vs PySR Benchmark  [mode: {mode_label}]")
    print(f" Total runs: {total}  ({len(targets)} scenarios x {len(args.noise_sweep)} noise x {len(args.seeds)} seeds)")
    print(f" Extrapolation: {'disabled' if args.no_extrap else 'enabled'}  "
          f"Blind arm: {'disabled' if args.no_blind else 'enabled'}")
    print("=" * 110)

    done = len(all_results)
    for sc_name in targets:
        dmax = DEFAULT_CLEAN_DOMAINS.get(
            sc_name, DOMAIN_RESTRICTIONS.get(sc_name, {}).get("domain_max", 1.0)
        )
        extrap_dmax       = None if args.no_extrap else EXTRAP_TEST_DOMAIN.get(sc_name)
        extrap_train_dmax = None if args.no_extrap else EXTRAP_TRAIN_DOMAIN.get(sc_name)

        for noise in args.noise_sweep:
            for seed in args.seeds:
                run_key = (sc_name, round(float(noise), 6), int(seed))
                if run_key in completed_keys:
                    continue

                t0 = time.time()
                try:
                    r = run_one_combination(
                        sc_name, noise, seed, args.engine, dmax,
                        extrap_dmax, extrap_train_dmax,
                        args.min_pysr_seconds,
                        args.include_ablation and run_tier1,
                        run_adcd_guided=run_adcd_guided,
                        run_adcd_blind=run_adcd_blind,
                        run_tier1=run_tier1,
                        run_tier2=run_tier2,
                    )
                except Exception as exc:
                    r = {
                        "scenario": sc_name, "noise": noise, "seed": seed,
                        "error": str(exc),
                        "adcd":       {"is_match": None, "nmse_extrap": float("inf")},
                        "adcd_blind": {"is_match": None, "nmse_extrap": float("inf")},
                        "pysr":       {"is_match": None, "nmse_extrap": float("inf")},
                        "pysr_tier2": None,
                        "ablation_pysr_adcd_bic": None,
                    }
                all_results.append(r)
                completed_keys.add(run_key)
                done += 1
                elapsed = time.time() - t0

                # Incremental checkpoint save
                try:
                    partial_summary = aggregate_summary(
                        all_results, targets, args.noise_sweep,
                        include_ablation=args.include_ablation and run_tier1,
                        include_tier2=run_tier2,
                        include_blind=include_blind,
                    )
                    checkpoint_payload = {
                        "config": {
                            "engine": args.engine,
                            "seeds": args.seeds,
                            "noise_sweep": args.noise_sweep,
                            "binary_operators": PYSR_BINARY_OPERATORS,
                            "unary_operators": PYSR_UNARY_OPERATORS,
                            "min_pysr_seconds": args.min_pysr_seconds,
                            "include_ablation": args.include_ablation,
                            "include_tier2": run_tier2,
                            "tier2_only": args.tier2_only,
                            "include_blind": include_blind,
                            "dimensional_constraint_penalty": PYSR_DIMENSIONAL_CONSTRAINT_PENALTY if run_tier2 else None,
                            "degenerate_theta_bound": _DEGENERATE_THETA_BOUND,
                            "extrap_sanity_bound": _EXTRAP_SANITY_BOUND,
                            "extrapolation_enabled": not args.no_extrap,
                        },
                        "summary": partial_summary,
                        "raw_runs": all_results,
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_payload, f, indent=2, default=str)
                except Exception:
                    pass

                a_match  = r["adcd"].get("is_match")
                ab_match = r.get("adcd_blind", {}).get("is_match")
                p_match  = r["pysr"].get("is_match")
                t2_match = (r.get("pysr_tier2") or {}).get("is_match")

                parts = [f"[{done:>3}/{total}] {sc_name:<20} noise={noise:<5.2f} seed={seed:<3}"]
                if run_adcd_guided:
                    parts.append(f"ADCD={'MATCH' if a_match else ('miss' if a_match is False else 'N/A'):<5}")
                if run_adcd_blind:
                    parts.append(f"blind={'MATCH' if ab_match else ('miss' if ab_match is False else 'N/A'):<5}")
                if run_tier1:
                    parts.append(f"PySR={'MATCH' if p_match else ('miss' if p_match is False else 'N/A'):<5}")
                if run_tier2:
                    parts.append(f"Tier2={'MATCH' if t2_match else ('miss' if t2_match is False else 'N/A'):<5}")
                parts.append(f"({elapsed:.1f}s)")
                print(" | ".join(parts))

    summary = aggregate_summary(
        all_results, targets, args.noise_sweep,
        include_ablation=args.include_ablation and run_tier1,
        include_tier2=run_tier2,
        include_blind=include_blind,
    )

    print_summary_table(summary, args.include_ablation and run_tier1, run_tier2, include_blind)

    if not args.tier2_only:
        plot_results(summary, out_dir, args.include_ablation and run_tier1, run_tier2, include_blind)

    payload = {
        "config": {
            "engine": args.engine,
            "seeds": args.seeds,
            "noise_sweep": args.noise_sweep,
            "binary_operators": PYSR_BINARY_OPERATORS,
            "unary_operators": PYSR_UNARY_OPERATORS,
            "min_pysr_seconds": args.min_pysr_seconds,
            "include_ablation": args.include_ablation,
            "include_tier2": run_tier2,
            "tier2_only": args.tier2_only,
            "include_blind": include_blind,
            "dimensional_constraint_penalty": PYSR_DIMENSIONAL_CONSTRAINT_PENALTY if run_tier2 else None,
            "degenerate_theta_bound": _DEGENERATE_THETA_BOUND,
            "extrap_sanity_bound": _EXTRAP_SANITY_BOUND,
            "extrapolation_enabled": not args.no_extrap,
        },
        "summary": summary,
        "raw_runs": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[OK] Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
