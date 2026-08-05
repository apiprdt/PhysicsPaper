"""
adcd.quickfit — Deterministic High-Level Wrapper
=================================================
No LLM. No external calls. No randomness.

DESIGN PHILOSOPHY
-----------------
This is a PURELY RULE-BASED wrapper that automates the only manual step in
the current ADCD interface: choosing ratio_symbol candidates. Everything else
(the 4-gate validation protocol, BIC ranking, identifiability check,
determinism guarantee) is handled by the existing core engine, which this
module NEVER modifies.

THREE-LAYER ARCHITECTURE (separation of concerns):
  Layer 0 — Core engine:   correction_orchestrator.py  ← NOT TOUCHED
  Layer 1 — This file:     quickfit.py                 ← auto-generates input
  Layer 2 — (future opt.)  LLM front-end               ← populates ParsedInput

USAGE (minimum viable, no domain expertise required)
-----------------------------------------------------
    import adcd
    import pandas as pd

    df = pd.read_csv("my_experiment.csv")

    result = adcd.quickfit(
        data              = df,
        target            = "y_obs",          # column: what you measured
        classical_formula = "m * v",           # the classical law you know
        variables         = {
            "m": "mass",                       # column name → physical dimension
            "v": "velocity",
        }
    )
    result.show()

WHAT USER MUST PROVIDE (minimum, irreducible)
----------------------------------------------
  1. data (DataFrame)       — raw experimental or synthetic data
  2. target (str)           — name of the observed output column
  3. classical_formula (str)— string formula of the known classical law
                              (e.g. "m * v", "k * x", "G * m * M / r**2")
  4. variables (dict)       — {column_name: physical_dimension} mapping
                              supported dimensions: "mass", "length", "time",
                              "velocity", "force", "energy", "charge",
                              "temperature", "dimensionless"

WHY classical_formula IS NON-NEGOTIABLE
----------------------------------------
ADCD searches for a *correction* to a known law. Without knowing the
classical prediction, there is no deviation to search for. This is a
fundamental requirement of the correction-first methodology, not an
engineering limitation. It cannot be auto-detected from data alone without
introducing the exact LLM/memorization risks this architecture avoids.

HOW RATIO AUTO-GENERATION WORKS
---------------------------------
Given variables {"v": "velocity", "c": "velocity", "m": "mass"}:

Step 1 — Dimensional consistency filter:
  Only pairs of variables with the SAME dimension can form a valid
  dimensionless ratio. E.g., v/c is dimensionless; v/m is not.

Step 2 — Candidate generation per valid pair (a, b):
  For each dimensionless pair (a, b) where dim(a) == dim(b):
    • a / b                  (raw ratio, no free parameter)
    • a / theta_new          (ratio with free scale parameter)
    • a**2 / b**2            (squared ratio)
    • (a**2 + b**2) / b**2  (quadratic combination)
  Limit: max_ratios (default 4) per unique dimension group, so the search
  space stays exhaustively enumerable on a CPU in seconds.

Step 3 — Report search_space_size honestly before any fitting begins,
  so the user can see exactly what the engine is searching over.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# DIMENSION TABLE
# Maps user-friendly dimension names to [M, L, T] exponents (SI basis)
# =====================================================================

_DIM_VECTORS: Dict[str, List[int]] = {
    # Base dimensions
    "mass":           [1, 0, 0],
    "length":         [0, 1, 0],
    "time":           [0, 0, 1],
    "temperature":    [0, 0, 0],   # treated as independent here; dimensionless in ratio
    "charge":         [0, 0, 0],   # EM base: same treatment
    "dimensionless":  [0, 0, 0],

    # Derived — common in physics corrections
    "velocity":       [0, 1, -1],  # L T^-1
    "acceleration":   [0, 1, -2],  # L T^-2
    "force":          [1, 1, -2],  # M L T^-2
    "energy":         [1, 2, -2],  # M L^2 T^-2
    "momentum":       [1, 1, -1],  # M L T^-1
    "frequency":      [0, 0, -1],  # T^-1
    "pressure":       [1, -1, -2], # M L^-1 T^-2
    "density":        [1, -3, 0],  # M L^-3
    "volume":         [0, 3, 0],   # L^3
    "area":           [0, 2, 0],   # L^2
    "wavenumber":     [0, -1, 0],  # L^-1
    "angle":          [0, 0, 0],   # dimensionless
    "number":         [0, 0, 0],   # dimensionless count
    "number_density": [0, -3, 0],  # L^-3
}


def _dim_vec(dimension: str) -> List[int]:
    """Return [M, L, T] exponent vector for a dimension string."""
    key = dimension.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in _DIM_VECTORS:
        warnings.warn(
            f"[quickfit] Unknown dimension '{dimension}'. Treating as 'dimensionless'. "
            f"Supported: {sorted(_DIM_VECTORS.keys())}",
            UserWarning,
            stacklevel=4,
        )
        return [0, 0, 0]
    return _DIM_VECTORS[key]


def _is_same_dimension(dim_a: str, dim_b: str) -> bool:
    """Return True iff two variables have identical dimensional vector."""
    return _dim_vec(dim_a) == _dim_vec(dim_b)


# =====================================================================
# RATIO CANDIDATE GENERATOR (Buckingham-π inspired, fully deterministic)
# =====================================================================

def _generate_ratio_candidates(
    variables: Dict[str, str],
    max_ratios: int = 4,
) -> List[str]:
    """
    Return a list of dimensionless ratio candidate strings, one per line.

    For each pair of variables (a, b) with the same physical dimension:
      • a/b, a**2/b**2, a/theta_new, a**2/theta_new
    Bounded to max_ratios candidates per dimension group.

    Returns strings like "v/c", "v**2/c**2", "v/theta_0", etc.
    These are later inserted into enumerate_candidates() as ratio_symbol.
    """
    var_names = list(variables.keys())
    candidates: List[str] = []
    seen: set = set()

    # theta counter starts at 0, incremented per new free-scale ratio
    theta_idx = 0

    # Group variables by dimension
    dim_groups: Dict[str, List[str]] = {}
    for var, dim in variables.items():
        vec_key = str(_dim_vec(dim))
        dim_groups.setdefault(vec_key, []).append(var)

    for vec_key, group in dim_groups.items():
        group_candidates: List[str] = []

        if len(group) >= 2:
            # Pairs within same dimension group → dimensionless ratio
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j:
                        continue
                    a, b = group[i], group[j]

                    r1 = f"{a}/{b}"
                    r2 = f"{a}**2/{b}**2"
                    if r1 not in seen:
                        group_candidates.append(r1)
                        seen.add(r1)
                    if r2 not in seen:
                        group_candidates.append(r2)
                        seen.add(r2)

        # Also add free-scale ratios: var / theta_new  (for any berdimensi var)
        if vec_key != str([0, 0, 0]):  # skip purely dimensionless
            for var in group:
                r3 = f"{var}/theta_{theta_idx}"
                r4 = f"{var}**2/theta_{theta_idx + 1}"
                theta_idx += 2
                if r3 not in seen:
                    group_candidates.append(r3)
                    seen.add(r3)
                if r4 not in seen:
                    group_candidates.append(r4)
                    seen.add(r4)

        # Cap per dimension group to avoid search space explosion
        candidates.extend(group_candidates[:max_ratios])

    if not candidates:
        # Fallback: generate a generic free-parameter ratio for first variable
        first_var = var_names[0] if var_names else "x"
        candidates = [f"{first_var}/theta_0"]
        warnings.warn(
            "[quickfit] No valid dimensionless ratios found from provided variable dimensions. "
            "Using generic free-parameter ratio as fallback. "
            "Check your 'variables' dimension mapping.",
            UserWarning,
            stacklevel=3,
        )

    return candidates


# =====================================================================
# FORMULA EVALUATOR (pure SymPy + NumPy, no LLM)
# =====================================================================

def _evaluate_formula(
    formula: str,
    data_arrays: Dict[str, np.ndarray],
    constants: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Evaluate a classical formula string against numpy data arrays.

    E.g. formula="m * v", data_arrays={"m": arr_m, "v": arr_v}
    → returns arr_m * arr_v element-wise.

    Uses numpy's eval via a safe namespace — no exec of arbitrary code,
    only math operations and the explicitly provided variable names.
    """
    import sympy as sp

    # Build safe evaluation namespace: only numpy math + user's variable arrays
    ns: Dict = {
        "np": np,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }
    ns.update(data_arrays)
    if constants:
        ns.update(constants)

    try:
        result = eval(formula, {"__builtins__": {}}, ns)  # noqa: S307
        return np.asarray(result, dtype=float)
    except Exception as exc:
        raise ValueError(
            f"[quickfit] Could not evaluate classical_formula='{formula}' "
            f"with variables {list(data_arrays.keys())}.\n"
            f"Error: {exc}\n"
            f"Tip: Use Python arithmetic operators: *, **, /, +, -"
        ) from exc


# =====================================================================
# MAIN PUBLIC API
# =====================================================================

def quickfit(
    data,                                    # pd.DataFrame
    target: str,                             # column name: observed output
    classical_formula: str,                  # e.g. "m * v"
    variables: Dict[str, str],              # {col_name: dimension_string}
    constants: Optional[Dict[str, float]] = None,  # e.g. {"c": 3e8}
    mode: str = "auto",                      # "auto" | "multiplicative" | "additive"
    max_ratios: int = 4,                     # max ratio candidates per dim group
    seed: int = 42,
    noise_level: float = 0.0,
    max_iterations: int = 1,
    verbose: bool = True,
):
    """
    Discover the algebraic correction to a known classical law from data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data. Must contain columns listed in `variables` and `target`.
    target : str
        Column name of the observed (anomalous) output.
    classical_formula : str
        Python arithmetic string of the classical law, e.g. ``"m * v"`` or
        ``"0.5 * m * v**2"``. Variables must match column names in `data`.
    variables : dict
        Mapping of column names to physical dimension strings.
        Example: ``{"m": "mass", "v": "velocity", "c": "velocity"}``
        Supported dimensions: mass, length, time, velocity, energy, force,
        momentum, temperature, pressure, volume, density, dimensionless, etc.
    constants : dict, optional
        Physical constants not present as columns, e.g. ``{"c": 3e8}``.
        These are included in formula evaluation but not in ratio generation.
    mode : str
        Correction type: ``"auto"`` (default), ``"multiplicative"``, or
        ``"additive"``. Auto-detection uses rank-correlation statistics.
    max_ratios : int
        Maximum ratio candidates per dimension group. Default 4.
        Increase if you suspect the correct ratio is not being found.
        Decrease if you need faster runtime.
    seed : int
        Random seed. ADCD is deterministic — this only affects data
        generation for synthetic scenarios passed via anomaly_scenarios.
    noise_level : float
        Noise level injected during data generation (for synthetic runs).
        For real data, leave at 0.0.
    max_iterations : int
        Maximum optimization iterations. Default 1 (single pass).
    verbose : bool
        Print progress, search space size, and detected mode.

    Returns
    -------
    ADCDResult
        Same result object as ``adcd.fit()``. Call ``.summary()`` or
        ``.plot_residuals()`` on the result.

    Examples
    --------
    >>> import adcd, pandas as pd
    >>> df = pd.read_csv("relativistic_momentum.csv")
    >>> result = adcd.quickfit(
    ...     data=df,
    ...     target="p_obs",
    ...     classical_formula="m * v",
    ...     variables={"m": "mass", "v": "velocity", "c": "velocity"},
    ...     constants={"c": 3e8},
    ... )
    >>> result.summary()
    """
    import pandas as pd
    from adcd.mode_detection import detect_correction_mode
    from adcd.asymptotic_dictionary_proposer_v3 import (
        AsymptoticDictionaryProposerV3,
        GrammarBudget,
        enumerate_candidates,
        PRIMITIVE_REGISTRY,
    )
    from adcd.api import fit  # core engine — not modified

    # ------------------------------------------------------------------
    # 1. Extract arrays from DataFrame
    # ------------------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "[quickfit] 'data' must be a pandas DataFrame. "
            "Example: pd.read_csv('my_data.csv')"
        )

    if target not in data.columns:
        raise ValueError(
            f"[quickfit] Column '{target}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    for col in variables:
        if col not in data.columns and (constants is None or col not in constants):
            raise ValueError(
                f"[quickfit] Column '{col}' listed in variables not found in data. "
                f"Available columns: {list(data.columns)}"
            )

    # Build X dict from variable columns that exist as data columns
    X: Dict[str, np.ndarray] = {}
    data_arrays: Dict[str, np.ndarray] = {}
    for col in variables:
        if col in data.columns:
            X[col] = data[col].to_numpy(dtype=float)
            data_arrays[col] = X[col]

    y_obs = data[target].to_numpy(dtype=float)

    # Add constants to evaluation namespace
    if constants:
        data_arrays.update({k: np.full_like(y_obs, v) for k, v in constants.items()})
        for k, v in constants.items():
            if k not in X:
                X[k] = np.full_like(y_obs, v)

    # ------------------------------------------------------------------
    # 2. Evaluate classical formula
    # ------------------------------------------------------------------
    if verbose:
        print(f"[quickfit] Evaluating classical formula: y_classical = {classical_formula}")
    y_classical = _evaluate_formula(classical_formula, data_arrays, constants)

    # ------------------------------------------------------------------
    # 3. Auto-detect correction mode
    # ------------------------------------------------------------------
    if mode == "auto":
        detected_mode, confidence = detect_correction_mode(y_obs, y_classical)
        if verbose:
            print(
                f"[quickfit] Mode detection → '{detected_mode}' "
                f"(confidence={confidence:.2f})"
            )
        if confidence < 0.6:
            warnings.warn(
                f"[quickfit] Mode detection confidence is low ({confidence:.2f}). "
                "The dataset may not clearly distinguish additive from multiplicative. "
                "Consider specifying mode='multiplicative' or mode='additive' explicitly.",
                UserWarning,
                stacklevel=2,
            )
        correction_mode = detected_mode
    else:
        correction_mode = mode
        if verbose:
            print(f"[quickfit] Mode: '{correction_mode}' (user-specified)")

    # ------------------------------------------------------------------
    # 4. Auto-generate ratio candidates (Buckingham-π, deterministic)
    # ------------------------------------------------------------------
    ratio_candidates = _generate_ratio_candidates(variables, max_ratios=max_ratios)

    if verbose:
        print(f"\n[quickfit] Auto-generated {len(ratio_candidates)} ratio candidates:")
        for r in ratio_candidates:
            print(f"  • {r}")

    # Count total candidate expressions across all ratios
    total_exprs = 0
    for ratio in ratio_candidates:
        cands = enumerate_candidates(ratio, GrammarBudget())
        total_exprs += len(cands)

    if verbose:
        print(
            f"\n[quickfit] Search space: {len(ratio_candidates)} ratios × "
            f"~{total_exprs // max(len(ratio_candidates), 1)} grammar forms "
            f"= {total_exprs} total candidates (exhaustive, deterministic)"
        )
        print("[quickfit] Running core ADCD engine (4-gate validation)...\n")

    # ------------------------------------------------------------------
    # 5. Determine limit variable (variable with largest range relative
    #    to its mean — heuristic for "which variable governs the regime")
    # ------------------------------------------------------------------
    limit_variable: Optional[str] = None
    best_cv = -1.0
    for col, arr in X.items():
        if len(arr) > 1 and np.mean(np.abs(arr)) > 1e-15:
            cv = np.std(arr) / np.mean(np.abs(arr))
            if cv > best_cv:
                best_cv = cv
                limit_variable = col

    if limit_variable is None:
        limit_variable = list(X.keys())[0]

    if verbose:
        print(f"[quickfit] Inferred classical limit variable: '{limit_variable}' "
              f"(highest coefficient of variation={best_cv:.3f})")
        print("[quickfit] If this is wrong, call adcd.fit() directly with "
              "limit_variable='<your_variable>'.\n")

    # ------------------------------------------------------------------
    # 6. Run the core engine once per ratio candidate, keep best result
    # ------------------------------------------------------------------
    best_result = None
    best_nmse = float("inf")

    for ratio_sym in ratio_candidates:
        proposer = AsymptoticDictionaryProposerV3(
            ratio_symbol=ratio_sym,
            budget=GrammarBudget(),
        )
        try:
            result = fit(
                X=X,
                y_obs=y_obs,
                y_classical=y_classical,
                classical_expr=classical_formula,
                variables_with_units=variables,
                correction_mode=correction_mode,
                limit_variable=limit_variable,
                proposer=proposer,
                max_iterations=max_iterations,
                seed=seed,
                noise_level=noise_level,
                verbose=False,
            )
            # Compare by NMSE of best candidate
            sr = result.search_result
            bayesian = sr.bayesian_output
            if bayesian and hasattr(bayesian, "best_nmse") and bayesian.best_nmse < best_nmse:
                best_nmse = bayesian.best_nmse
                best_result = result
            elif best_result is None:
                best_result = result  # take first if no NMSE available yet
        except Exception as exc:
            logger.warning(f"[quickfit] Ratio '{ratio_sym}' failed: {exc}")
            continue

    if best_result is None:
        raise RuntimeError(
            "[quickfit] No ratio candidate produced a valid result. "
            "Possible causes:\n"
            "  1. classical_formula evaluates to zero or near-zero for your data.\n"
            "  2. No dimensionless ratio pairs exist for the given variable dimensions.\n"
            "  3. The true correction lies outside the 5-primitive dictionary.\n"
            "Try adcd.fit() directly for more control."
        )

    if verbose:
        print(
            f"[quickfit] ✓ Done. Best ratio: "
            f"search complete across {len(ratio_candidates)} candidates."
        )

    return best_result
