#!/usr/bin/env python3
"""
benchmark_pysr_comparison.py
==============================================================================
Comparative Benchmark Harness: ADCD vs PySR (Noise Robustness & Extrapolation).

Evaluates structural recovery rate and extrapolation performance across
systematic noise sweeps under identical data, search targets, and feature sets.
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

try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None


# ==============================================================================
# Benchmark Configuration & Default Regimes
# ==============================================================================

NOISE_SWEEP: List[float] = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46]

LOCKED_SCENARIOS: List[str] = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]

DEFAULT_CLEAN_DOMAINS: Dict[str, float] = {
    "Time Dilation": 0.99,
    "Screened Coulomb": 4.0,
    "Entropy Expansion": 3.0,
}

EXTRAP_DOMAIN_MULTIPLIER: Dict[str, float] = {
    "Time Dilation": 0.90,
    "Screened Coulomb": 8.0,
    "Entropy Expansion": 6.0,
}

# Standard out-of-the-box operator basis for PySR
PYSR_BINARY_OPERATORS: List[str] = ["+", "-", "*", "/"]
PYSR_UNARY_OPERATORS: List[str] = ["exp", "log", "sqrt", "sin", "cos"]

MIN_PYSR_SECONDS: float = 30.0


# ==============================================================================
# Statistical Metrics
# ==============================================================================

def wilson_interval(
    successes: int, n: int, z: float = 1.96
) -> Tuple[float, float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    Returns:
        (point_estimate, lower_95, upper_95)
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(
        (p * (1 - p) / n) + (z * z / (4 * n * n))
    )
    return p, max(0.0, center - half_width), min(1.0, center + half_width)


# ==============================================================================
# Data Layer (Shared Representation)
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
    """Verify byte-exact data generation determinism across consecutive calls."""
    X1, y1, yc1, _ = scenario.generate_data(
        noise_level=noise, seed=seed, domain_max=domain_max
    )
    X2, y2, yc2, _ = scenario.generate_data(
        noise_level=noise, seed=seed, domain_max=domain_max
    )
    assert np.array_equal(y1, y2), (
        f"Data generation non-deterministic for scenario '{scenario.name}'."
    )
    assert np.array_equal(yc1, yc2), (
        "Non-deterministic classical prediction detected in data generator."
    )


def build_shared_data(
    scenario,
    noise: float,
    seed: int,
    domain_max: float,
    extrap_domain_max: Optional[float] = None,
) -> SharedData:
    """
    Construct shared training and held-out extrapolation data arrays.
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

    # Generate held-out extrapolation domain (wider domain, noiseless reference)
    X_extrap, target_extrap = None, None
    if extrap_domain_max is not None and extrap_domain_max > domain_max:
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
) -> Optional[np.ndarray]:
    """Evaluate symbolic expression on input dataset array."""
    try:
        expr = sp.sympify(expr_str)
        all_names = list(X.keys())
        sym_locals = {name: sp.Symbol(name) for name in all_names}
        free_syms = [sym_locals[n] for n in all_names if sp.Symbol(n) in expr.free_symbols]
        if not free_syms:
            val = float(expr)
            return np.full(len(next(iter(X.values()))), val)
        fn = sp.lambdify(free_syms, expr, modules=["numpy"])
        args = [X[str(s)] for s in free_syms]
        result = np.asarray(fn(*args), dtype=float)
        if not np.all(np.isfinite(result)):
            return None
        return result
    except Exception:
        return None


def _compute_nmse_extrap(
    expr_str: str,
    data: SharedData,
) -> float:
    """Compute Normalized Mean Squared Error on held-out extrapolation domain."""
    if data.X_extrap is None or data.target_extrap is None:
        return float("inf")
    pred = _evaluate_expression_on_data(expr_str, data.X_extrap, data.feature_names)
    if pred is None:
        return float("inf")
    y_true = data.target_extrap
    var_y = float(np.var(y_true)) + 1e-300
    return float(np.mean((pred - y_true) ** 2) / var_y)


def _extract_numeric_constants_as_theta_fit(expr: sp.Expr) -> Dict[str, float]:
    """Extract fitted numeric literals to assist structural classification."""
    theta_fit: Dict[str, float] = {}
    idx = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Number) and not node.is_Integer:
            theta_fit[f"theta_{idx}"] = float(node)
            idx += 1
    return theta_fit


# ==============================================================================
# ADCD Execution Path
# ==============================================================================

def run_adcd_once(
    scenario,
    noise: float,
    seed: int,
    domain_max: float,
    engine: str,
) -> Dict[str, Any]:
    """Execute single ADCD validation run."""
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
    )
    elapsed = time.time() - t0

    ps = res.checks.get("primary_search", {})
    match_level = ps.get("match_level", "none")

    return {
        "method": "ADCD",
        "tier": res.tier,
        "is_match": match_level in ("exact", "class_only"),
        "match_level": match_level,
        "discovered_class": ps.get("discovered_class", "unknown"),
        "nmse_train": ps.get("nmse"),
        "expr_str": ps.get("top_candidate", ""),
        "elapsed_seconds": elapsed,
    }


# ==============================================================================
# PySR Execution Path
# ==============================================================================

def _count_free_params_in_pysr_expr(
    expr_str: str,
    feature_names: List[str],
) -> int:
    """Count distinct fitted parameters in a symbolic expression for BIC calculation."""
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
        n_numeric_params = len(numeric_consts)

        return max(1, n_symbol_params + n_numeric_params)
    except Exception:
        return 1


def run_pysr_once(
    scenario,
    data: SharedData,
    seed: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Execute single PySR regression run on residual data."""
    if PySRRegressor is None:
        raise ImportError("PySR is not installed.")
    if pd is None:
        raise ImportError("pandas is required for PySR.")

    X_df = pd.DataFrame({name: data.X[name] for name in data.feature_names})
    y = np.asarray(data.target, dtype=float)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PySRRegressor(
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
        model.fit(X_df, y)
    elapsed = time.time() - t0

    try:
        best_expr_sympy = model.sympy()
        best_expr_str = str(best_expr_sympy)
    except Exception:
        best_expr_sympy = sp.sympify("0")
        best_expr_str = "0"

    theta_fit = _extract_numeric_constants_as_theta_fit(best_expr_sympy)
    discovered_class = classify_structure(best_expr_sympy, theta_fit=theta_fit)
    is_match = discovered_class == scenario.correction_class

    try:
        pred = model.predict(X_df)
        var_y = float(np.var(y)) + 1e-300
        nmse_train = float(np.mean((y - pred) ** 2) / var_y)
    except Exception:
        nmse_train = float("nan")

    n_pareto = len(model.equations_) if hasattr(model, "equations_") else None

    return {
        "method": "PySR",
        "is_match": is_match,
        "discovered_class": discovered_class,
        "nmse_train": nmse_train,
        "expr_str": best_expr_str,
        "elapsed_seconds": elapsed,
        "n_equations_in_pareto": n_pareto,
        "_model_equations": model.equations_ if hasattr(model, "equations_") else None,
        "_feature_names": data.feature_names,
    }


def run_pysr_bic_reselect(
    scenario,
    data: SharedData,
    pysr_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Reselect best PySR Pareto candidate using extended BIC score."""
    eqs = pysr_result.get("_model_equations")
    feature_names = pysr_result.get("_feature_names", data.feature_names)

    if eqs is None or len(eqs) == 0:
        return {
            "method": "PySR+ADCD_BIC",
            "is_match": False,
            "discovered_class": "unknown",
            "note": "no_pareto_front",
        }

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

            b = extended_bic_score(
                nmse, n_params, n_points, n_candidates=n_candidates
            )
            if best_bic is None or b < best_bic:
                best_bic = b
                best_expr_str = expr_str_row
        except Exception:
            continue

    if best_expr_str is None:
        return {
            "method": "PySR+ADCD_BIC",
            "is_match": False,
            "discovered_class": "unknown",
            "note": "reselection_failed",
        }

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
    min_pysr_seconds: float,
    include_ablation: bool,
) -> Dict[str, Any]:
    """Execute paired ADCD and PySR runs on identical data sample."""
    scenarios = {s.name: s for s in get_all_scenarios()}
    scenario = scenarios[scenario_name]

    adcd_res = run_adcd_once(scenario, noise, seed, domain_max, engine)
    pysr_timeout = max(adcd_res["elapsed_seconds"], min_pysr_seconds)

    data = build_shared_data(scenario, noise, seed, domain_max, extrap_domain_max)

    try:
        pysr_res = run_pysr_once(scenario, data, seed, timeout_seconds=pysr_timeout)
    except (ImportError, RuntimeError, Exception) as exc:
        pysr_res = {"method": "PySR", "is_match": None, "error": str(exc)}

    adcd_nmse_extrap = _compute_nmse_extrap(adcd_res.get("expr_str", ""), data)
    pysr_nmse_extrap = (
        _compute_nmse_extrap(pysr_res.get("expr_str", ""), data)
        if pysr_res.get("is_match") is not None else float("inf")
    )
    adcd_res["nmse_extrap"] = adcd_nmse_extrap
    pysr_res["nmse_extrap"] = pysr_nmse_extrap

    ablation_res = None
    if include_ablation and pysr_res.get("is_match") is not None:
        ablation_res = run_pysr_bic_reselect(scenario, data, pysr_res)
        if ablation_res:
            ab_expr = ablation_res.get("expr_str", "")
            ablation_res["nmse_extrap"] = _compute_nmse_extrap(ab_expr, data)

    pysr_res.pop("_model_equations", None)
    pysr_res.pop("_feature_names", None)

    return {
        "scenario": scenario_name,
        "noise": noise,
        "seed": seed,
        "pysr_timeout_seconds_used": pysr_timeout,
        "adcd": adcd_res,
        "pysr": pysr_res,
        "ablation_pysr_adcd_bic": ablation_res,
    }


def aggregate_summary(
    all_results: List[Dict[str, Any]],
    targets: List[str],
    noise_sweep: List[float],
    include_ablation: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate run statistics and compute Wilson confidence intervals."""
    summary: Dict[str, List[Dict[str, Any]]] = {}
    for sc_name in targets:
        rows = []
        for noise in noise_sweep:
            subset = [
                r for r in all_results
                if r["scenario"] == sc_name and abs(r["noise"] - noise) < 1e-9
            ]
            n = len(subset)
            a_succ = sum(1 for r in subset if r["adcd"]["is_match"])
            p_succ = sum(1 for r in subset if r["pysr"].get("is_match") is True)

            a_p, a_lo, a_hi = wilson_interval(a_succ, n)
            p_p, p_lo, p_hi = wilson_interval(p_succ, n)

            a_extrap = [r["adcd"].get("nmse_extrap", float("inf")) for r in subset]
            p_extrap = [r["pysr"].get("nmse_extrap", float("inf")) for r in subset]
            a_extrap_finite = [v for v in a_extrap if math.isfinite(v)]
            p_extrap_finite = [v for v in p_extrap if math.isfinite(v)]

            row: Dict[str, Any] = {
                "noise": noise,
                "n": n,
                "adcd_recovery": a_p,
                "adcd_ci95": [a_lo, a_hi],
                "adcd_nmse_extrap_median": (
                    float(np.median(a_extrap_finite)) if a_extrap_finite else float("inf")
                ),
                "pysr_recovery": p_p,
                "pysr_ci95": [p_lo, p_hi],
                "pysr_nmse_extrap_median": (
                    float(np.median(p_extrap_finite)) if p_extrap_finite else float("inf")
                ),
            }

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


# ==============================================================================
# Visualization & Reporting
# ==============================================================================

def plot_results(
    summary: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
    include_ablation: bool,
) -> None:
    """Generate comparative publication figures (PDF and PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not installed -- skipping plot generation.")
        return

    scenarios = list(summary.keys())
    n_panels = len(scenarios)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    ADCD_COLOR = "#1f77b4"
    PYSR_COLOR = "#ff7f0e"
    ABLATION_COLOR = "#2ca02c"

    has_extrap = any(
        math.isfinite(row.get("adcd_nmse_extrap_median", float("inf")))
        for rows in summary.values()
        for row in rows
    )

    for ax, sc_name in zip(axes, scenarios):
        rows = summary[sc_name]
        noise_x = [r["noise"] for r in rows]

        a_rates = [r["adcd_recovery"] * 100 for r in rows]
        a_lo = [r["adcd_ci95"][0] * 100 for r in rows]
        a_hi = [r["adcd_ci95"][1] * 100 for r in rows]

        p_rates = [r["pysr_recovery"] * 100 for r in rows]
        p_lo = [r["pysr_ci95"][0] * 100 for r in rows]
        p_hi = [r["pysr_ci95"][1] * 100 for r in rows]

        ax.plot(noise_x, a_rates, "o-", color=ADCD_COLOR, lw=2,
                label="ADCD (ours)", zorder=3)
        ax.fill_between(noise_x, a_lo, a_hi, alpha=0.18, color=ADCD_COLOR)

        ax.plot(noise_x, p_rates, "s--", color=PYSR_COLOR, lw=2,
                label="PySR (default)", zorder=3)
        ax.fill_between(noise_x, p_lo, p_hi, alpha=0.18, color=PYSR_COLOR)

        if include_ablation and "ablation_recovery" in rows[0]:
            ab_rates = [r.get("ablation_recovery", 0) * 100 for r in rows]
            ab_lo = [r.get("ablation_ci95", [0, 0])[0] * 100 for r in rows]
            ab_hi = [r.get("ablation_ci95", [0, 0])[1] * 100 for r in rows]
            ax.plot(noise_x, ab_rates, "^:", color=ABLATION_COLOR, lw=1.5,
                    label="PySR + ADCD-BIC", zorder=2)
            ax.fill_between(noise_x, ab_lo, ab_hi, alpha=0.12, color=ABLATION_COLOR)

        ax.set_xlim(min(noise_x) * 0.9, max(noise_x) * 1.05)
        ax.set_ylim(-5, 105)
        ax.set_xlabel("Noise level (fraction)", fontsize=10)
        ax.set_title(sc_name, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="lower left")

    axes[0].set_ylabel("Structural recovery rate (%)", fontsize=10)
    fig.suptitle(
        "ADCD vs PySR: Structural Recovery Rate vs Noise Level\n"
        "(shaded bands = Wilson 95% CI, n=5 seeds per point)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"fig_comparison_recovery.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] Recovery figure saved to {out_dir}/fig_comparison_recovery.{{pdf,png}}")

    if has_extrap:
        fig2, axes2 = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0), sharey=True)
        if n_panels == 1:
            axes2 = [axes2]

        for ax, sc_name in zip(axes2, scenarios):
            rows = summary[sc_name]
            noise_x = [r["noise"] for r in rows]
            a_ext = [r.get("adcd_nmse_extrap_median", float("nan")) for r in rows]
            p_ext = [r.get("pysr_nmse_extrap_median", float("nan")) for r in rows]

            a_ext_disp = [v if math.isfinite(v) else float("nan") for v in a_ext]
            p_ext_disp = [v if math.isfinite(v) else float("nan") for v in p_ext]

            ax.semilogy(noise_x, a_ext_disp, "o-", color=ADCD_COLOR, lw=2,
                        label="ADCD (ours)", zorder=3)
            ax.semilogy(noise_x, p_ext_disp, "s--", color=PYSR_COLOR, lw=2,
                        label="PySR (default)", zorder=3)
            ax.axhline(y=1.0, color="gray", linestyle=":", lw=1, label="NMSE = 1.0 (chance)")
            ax.set_xlabel("Noise level (fraction)", fontsize=10)
            ax.set_title(sc_name, fontsize=10, fontweight="bold")
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)

        axes2[0].set_ylabel("Extrapolation NMSE (log scale)", fontsize=10)
        fig2.suptitle(
            "ADCD vs PySR: Extrapolation NMSE (out-of-training domain)\n"
            "(lower = better; median across 5 seeds per point)",
            fontsize=10, y=1.02,
        )
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(out_dir, f"fig_comparison_extrap.{ext}")
            fig2.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig2)
        print(f"[OK] Extrapolation figure saved to {out_dir}/fig_comparison_extrap.{{pdf,png}}")


def print_summary_table(
    summary: Dict[str, List[Dict[str, Any]]],
    include_ablation: bool,
) -> None:
    """Print formatted comparative summary table to stdout."""
    for sc_name, rows in summary.items():
        n = rows[0]["n"]
        header = f" STRUCTURAL RECOVERY RATE vs NOISE -- {sc_name}  (n={n} seeds per point)"
        print("\n" + "=" * 92)
        print(header)
        print("-" * 92)

        cols = f"{'Noise':<8} | {'ADCD':<26} | {'PySR':<26}"
        if include_ablation and "ablation_recovery" in rows[0]:
            cols += f" | {'PySR+ADCD-BIC':<26}"
        cols += f" | {'ADCD extrap NMSE':<18} | {'PySR extrap NMSE':<18}"
        print(cols)
        print("-" * 92)

        for row in rows:
            a_str = (
                f"{row['adcd_recovery']*100:5.0f}%"
                f" [{row['adcd_ci95'][0]*100:4.0f}-{row['adcd_ci95'][1]*100:4.0f}%]"
            )
            p_str = (
                f"{row['pysr_recovery']*100:5.0f}%"
                f" [{row['pysr_ci95'][0]*100:4.0f}-{row['pysr_ci95'][1]*100:4.0f}%]"
            )
            a_ext = row.get("adcd_nmse_extrap_median", float("inf"))
            p_ext = row.get("pysr_nmse_extrap_median", float("inf"))
            a_ext_str = f"{a_ext:.3f}" if math.isfinite(a_ext) else "N/A"
            p_ext_str = f"{p_ext:.3f}" if math.isfinite(p_ext) else "N/A"

            line = f"{row['noise']:<8.2f} | {a_str:<26} | {p_str:<26}"
            if include_ablation and "ablation_recovery" in row:
                ab_str = (
                    f"{row['ablation_recovery']*100:5.0f}%"
                    f" [{row['ablation_ci95'][0]*100:4.0f}-{row['ablation_ci95'][1]*100:4.0f}%]"
                )
                line += f" | {ab_str:<26}"
            line += f" | {a_ext_str:<18} | {p_ext_str:<18}"
            print(line)
    print("=" * 92)


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparative Benchmark Harness: ADCD vs PySR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine", choices=["python", "julia"], default="julia",
        help="ADCD backend engine.",
    )
    parser.add_argument(
        "--scenario", default="all",
        help="Scenario name to run, or 'all' for all locked scenarios.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds for replication.",
    )
    parser.add_argument(
        "--noise-sweep", type=float, nargs="+", default=NOISE_SWEEP,
        help="Noise levels to evaluate.",
    )
    parser.add_argument(
        "--min-pysr-seconds", type=float, default=MIN_PYSR_SECONDS,
        help="Minimum PySR timeout in seconds.",
    )
    parser.add_argument(
        "--include-ablation", action="store_true",
        help="Evaluate PySR+ADCD_BIC reselection.",
    )
    parser.add_argument(
        "--no-extrap", action="store_true",
        help="Disable extrapolation evaluation.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: run_outputs/pysr_comparison.json).",
    )
    parser.add_argument(
        "--plot-only", default=None, metavar="JSON_PATH",
        help="Regenerate figures from existing JSON report without running search.",
    )
    args = parser.parse_args()

    out_path = args.out or os.path.join("run_outputs", "pysr_comparison.json")
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        with open(args.plot_only, "r", encoding="utf-8") as f:
            saved = json.load(f)
        summary = saved.get("summary", {})
        include_ablation = saved.get("config", {}).get("include_ablation", False)
        print_summary_table(summary, include_ablation)
        plot_results(summary, out_dir, include_ablation)
        return 0

    if PySRRegressor is None:
        print("[ERROR] PySR is not installed.", file=sys.stderr)
        return 1

    targets = LOCKED_SCENARIOS if args.scenario == "all" else [args.scenario]
    all_results: List[Dict[str, Any]] = []

    total = len(targets) * len(args.noise_sweep) * len(args.seeds)
    print("=" * 92)
    print(
        f" ADCD vs PySR Benchmark\n"
        f" Total runs: {total}  "
        f"({len(targets)} scenario(s) x {len(args.noise_sweep)} noise levels x {len(args.seeds)} seeds)\n"
        f" Extrapolation evaluation: {'disabled' if args.no_extrap else 'enabled'}"
    )
    print("=" * 92)

    done = 0
    for sc_name in targets:
        dmax = DEFAULT_CLEAN_DOMAINS.get(
            sc_name, DOMAIN_RESTRICTIONS.get(sc_name, {}).get("domain_max", 1.0)
        )
        extrap_dmax = (
            None if args.no_extrap
            else EXTRAP_DOMAIN_MULTIPLIER.get(sc_name)
        )

        for noise in args.noise_sweep:
            for seed in args.seeds:
                t0 = time.time()
                try:
                    r = run_one_combination(
                        sc_name, noise, seed, args.engine, dmax, extrap_dmax,
                        args.min_pysr_seconds, args.include_ablation,
                    )
                except Exception as exc:
                    r = {
                        "scenario": sc_name, "noise": noise, "seed": seed,
                        "error": str(exc),
                        "adcd": {"is_match": False, "nmse_extrap": float("inf")},
                        "pysr": {"is_match": None, "nmse_extrap": float("inf")},
                        "ablation_pysr_adcd_bic": None,
                    }
                all_results.append(r)
                done += 1

                a_match = r["adcd"].get("is_match", False)
                p_match = r["pysr"].get("is_match")
                elapsed = time.time() - t0
                print(
                    f"[{done:>3}/{total}] {sc_name:<20} noise={noise:<5.2f} seed={seed:<3}"
                    f" | ADCD={'MATCH' if a_match else 'miss':<5}"
                    f" PySR={'MATCH' if p_match else ('miss' if p_match is False else 'N/A'):<5}"
                    f" ({elapsed:.1f}s)"
                )

    summary = aggregate_summary(all_results, targets, args.noise_sweep, args.include_ablation)

    print_summary_table(summary, args.include_ablation)
    plot_results(summary, out_dir, args.include_ablation)

    payload = {
        "config": {
            "engine": args.engine,
            "seeds": args.seeds,
            "noise_sweep": args.noise_sweep,
            "binary_operators": PYSR_BINARY_OPERATORS,
            "unary_operators": PYSR_UNARY_OPERATORS,
            "min_pysr_seconds": args.min_pysr_seconds,
            "include_ablation": args.include_ablation,
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
