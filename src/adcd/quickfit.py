"""
adcd.quickfit — Deterministic High-Level Wrapper
=================================================
No LLM. No external calls. No randomness beyond the core engine's seed.

DESIGN PHILOSOPHY
-----------------
This module automates the only manual step in adcd.fit(): providing
ratio_symbol candidates. The core 4-gate validation protocol, BIC ranking,
identifiability check, and determinism guarantee are handled entirely by the
existing engine (correction_orchestrator.py), which this module never modifies.

ARCHITECTURE (three-layer separation)
--------------------------------------
  Layer 0 — Core engine   (correction_orchestrator.py)  ← NOT TOUCHED
  Layer 1 — This file     (quickfit.py)                 ← builds + validates input
  Layer 2 — (future opt.) LLM front-end                 ← schema-constrained only

CRITICAL DESIGN DECISIONS (auditor-reviewed)
---------------------------------------------
1. Formula evaluation uses sympy sympify() + lambdify(), NOT eval().
   Raw eval() with empty __builtins__ is not a real sandbox — Python object
   introspection can escape it without __builtins__. sympify/lambdify is a
   genuine mathematical parser that cannot execute arbitrary Python code.

2. ALL ratio candidates are passed to a SINGLE pipeline run via
   MultiRatioProposer (defined here), NOT separate pipeline runs per ratio.
   Running N separate pipelines and picking best NMSE is multiple-testing
   without BIC correction — equivalent to p-hacking, directly contradicting
   the extended_bic_score() mechanism in the core engine (§3.6 of the paper).
   With MultiRatioProposer, BIC naturally sees ALL candidates tried at once.

3. limit_variable auto-detection uses the same confidence-gated warning
   pattern as mode_detection — it announces its guess, gives confidence, and
   warns loudly when confidence is low. It does NOT run silently.

USAGE (minimum viable)
-----------------------
    import adcd, pandas as pd

    df = pd.read_csv("my_experiment.csv")

    result = adcd.quickfit(
        data              = df,
        target            = "t_obs",
        classical_formula = "t_0",
        variables         = {"v": "velocity", "t_0": "time", "c": "velocity"},
        constants         = {"c": 3e8},
    )
    result.summary()

WHAT CANNOT BE AUTOMATED (irreducible requirements)
-----------------------------------------------------
  classical_formula : Requires domain knowledge. Without it there is no
      "deviation" to compute. Auto-detection from column names would require
      an LLM guessing physics laws — the exact risk this architecture avoids.
  variables (dimension mapping) : Required for dimensionally valid ratio
      generation. Guessing dimensions from column names is heuristic and
      unverifiable without a human.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# DIMENSION TABLE — [M, L, T] exponent vectors (SI basis)
# =====================================================================

_DIM_VECTORS: Dict[str, List[int]] = {
    "mass":           [1, 0, 0],
    "length":         [0, 1, 0],
    "time":           [0, 0, 1],
    "temperature":    [0, 0, 0],   # independent; treated dimensionless in ratios
    "charge":         [0, 0, 0],
    "dimensionless":  [0, 0, 0],
    "angle":          [0, 0, 0],
    "number":         [0, 0, 0],
    # Derived
    "velocity":       [0, 1, -1],
    "acceleration":   [0, 1, -2],
    "force":          [1, 1, -2],
    "energy":         [1, 2, -2],
    "momentum":       [1, 1, -1],
    "frequency":      [0, 0, -1],
    "pressure":       [1, -1, -2],
    "density":        [1, -3, 0],
    "volume":         [0, 3, 0],
    "area":           [0, 2, 0],
    "wavenumber":     [0, -1, 0],
    "number_density": [0, -3, 0],
}

_SUPPORTED_DIMS = sorted(_DIM_VECTORS.keys())


def _dim_vec(dimension: str) -> List[int]:
    key = dimension.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in _DIM_VECTORS:
        warnings.warn(
            f"[quickfit] Unknown dimension '{dimension}'. Treating as 'dimensionless'. "
            f"Supported: {_SUPPORTED_DIMS}",
            UserWarning,
            stacklevel=4,
        )
        return [0, 0, 0]
    return list(_DIM_VECTORS[key])


def _is_nontrivial_dim(dim: str) -> bool:
    """Return True if dimension is not purely dimensionless."""
    return _dim_vec(dim) != [0, 0, 0]


# =====================================================================
# RATIO CANDIDATE GENERATOR — builds on enumerate_candidates, no re-impl
# =====================================================================

def _generate_ratio_symbols(
    variables: Dict[str, str],
    max_ratios: int = 4,
) -> List[str]:
    """
    Return a list of dimensionless ratio SYMBOL strings for the proposer.

    Strategy (matches auditor recommendation — build on existing grammar):
      For each pair of same-dimension variables (a, b):
        • a/b, b/a, a**2/b**2, b**2/a**2          (exact dimensionless ratios)
      For each dimensional variable v:
        • v/theta_new, v**2/theta_new              (free-scale ratios)

    Capped at max_ratios symbols total to keep search space exhaustively
    enumerable (each symbol will be expanded to ~35 grammar candidates by
    enumerate_candidates — so 4 symbols ≈ 140 total candidates, tractable).

    Returns strings suitable as ratio_symbol argument to enumerate_candidates().
    """
    symbols: List[str] = []
    seen: set = set()
    theta_idx = 0

    # Group by dimensional vector
    dim_groups: Dict[str, List[str]] = {}
    for var, dim in variables.items():
        key = str(_dim_vec(dim))
        dim_groups.setdefault(key, []).append(var)

    # Pass 1: Exact dimensionless ratios from same-dimension pairs
    for vec_key, group in dim_groups.items():
        if vec_key == str([0, 0, 0]):
            continue  # skip dimensionless vars — ratio of two dimensionless is trivial
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                for sym in [f"{a}/{b}", f"{b}/{a}", f"{a}**2/{b}**2", f"{b}**2/{a}**2"]:
                    if sym not in seen:
                        symbols.append(sym)
                        seen.add(sym)

    # Pass 2: Free-scale ratios for dimensional variables (var/theta)
    for var, dim in variables.items():
        if not _is_nontrivial_dim(dim):
            continue
        for sym in [f"{var}/theta_{theta_idx}", f"{var}**2/theta_{theta_idx + 1}"]:
            if sym not in seen:
                symbols.append(sym)
                seen.add(sym)
        theta_idx += 2

    # Trim to max_ratios and report honestly
    trimmed = symbols[:max_ratios]
    if len(symbols) > max_ratios:
        logger.info(
            f"[quickfit] Trimmed ratio candidates from {len(symbols)} to {max_ratios}. "
            f"Increase max_ratios to search more (may slow runtime)."
        )
    return trimmed if trimmed else [f"{list(variables.keys())[0]}/theta_0"]


# =====================================================================
# MULTI-RATIO PROPOSER — ONE pipeline, ALL candidates, BIC-correct
# =====================================================================

from adcd.llm_proposer import BaseProposer

class MultiRatioProposer(BaseProposer):
    """
    A drop-in proposer that generates candidates for ALL ratio symbols at
    once, then feeds them into a SINGLE pipeline run.

    WHY THIS MATTERS (auditor finding #2):
    Running a separate pipeline per ratio and picking best NMSE is equivalent
    to multiple testing without correction. BIC's n_candidates term (the
    extended_bic_score penalty 2*ln(n)*k) works correctly only when it sees
    the FULL set of candidates that were tried. By aggregating all candidates
    here, BIC naturally accounts for all ratios tested simultaneously.
    """

    def __init__(
        self,
        ratio_symbols: List[str],
        budget=None,
        exclude_primitives: Optional[List[str]] = None,
    ):
        from adcd.asymptotic_dictionary_proposer_v3 import (
            GrammarBudget, enumerate_candidates, PRIMITIVE_REGISTRY,
        )
        self.ratio_symbols = ratio_symbols
        self.budget = budget or GrammarBudget()
        exclude = set(exclude_primitives or [])
        self._active_primitives = {
            k: v for k, v in PRIMITIVE_REGISTRY.items() if k not in exclude
        }
        # Pre-build all candidate strings (deduplicated, deterministic)
        seen: set = set()
        self._candidates: List[str] = []
        for ratio in ratio_symbols:
            for cand in enumerate_candidates(
                ratio_symbol=ratio,
                budget=self.budget,
                active_primitives=self._active_primitives,
            ):
                if cand not in seen:
                    self._candidates.append(cand)
                    seen.add(cand)

        # Expose sources dict so orchestrator can optionally log them
        self.sources: Dict[str, str] = {c: "deterministic_grammar" for c in self._candidates}

    def propose(self, context) -> List[str]:
        # Return ALL candidates — BIC handles ranking and penalisation
        return self._candidates

    def search_space_size(self) -> int:
        return len(self._candidates)


# =====================================================================
# SAFE FORMULA EVALUATOR — sympy sympify + lambdify, no raw eval()
# =====================================================================

def _evaluate_formula(
    formula: str,
    variables: Dict[str, str],
    data_arrays: Dict[str, np.ndarray],
    constants: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Evaluate a classical formula string against numpy data arrays.

    Uses sympy.sympify() + sympy.lambdify(), NOT eval(). This is a genuine
    mathematical parser: it cannot execute arbitrary Python code or escape
    via object introspection (unlike eval() with empty __builtins__).

    Raises ValueError with a helpful message if the formula is invalid or
    references variables not present in data_arrays/constants.
    """
    import sympy as sp

    # Build symbol namespace: only variables explicitly declared by user
    all_names = list(data_arrays.keys())
    if constants:
        all_names += [k for k in constants if k not in data_arrays]

    sym_locals = {name: sp.Symbol(name) for name in all_names}

    try:
        expr = sp.sympify(formula, locals=sym_locals)
    except Exception as exc:
        raise ValueError(
            f"[quickfit] Cannot parse classical_formula='{formula}'.\n"
            f"  Error: {exc}\n"
            f"  Tip: Use standard arithmetic: m*v, 0.5*m*v**2, G*m*M/r**2, etc.\n"
            f"  Known variable names: {all_names}"
        ) from exc

    # Reject anything sympy couldn't reduce to a genuine symbolic expression
    if not isinstance(expr, sp.Basic):
        raise ValueError(
            f"[quickfit] classical_formula='{formula}' did not parse to a "
            f"mathematical expression (got {type(expr).__name__}). "
            f"Use standard arithmetic operators only: +, -, *, **, /."
        )


    # Build evaluation namespace: numpy arrays + scalar constants
    eval_ns: Dict = {}
    eval_ns.update(data_arrays)
    if constants:
        # Broadcast scalar constants to array shape
        ref = next(iter(data_arrays.values()))
        eval_ns.update({k: np.full_like(ref, v) for k, v in constants.items()
                        if k not in eval_ns})

    free_syms = [str(s) for s in expr.free_symbols]
    missing = [s for s in free_syms if s not in eval_ns]
    if missing:
        raise ValueError(
            f"[quickfit] Formula '{formula}' references unknown names: {missing}.\n"
            f"  Declared variables: {list(variables.keys())}\n"
            f"  Constants: {list((constants or {}).keys())}\n"
            f"  Add missing names to 'variables' or 'constants'."
        )

    try:
        func = sp.lambdify(list(expr.free_symbols), expr, modules="numpy")
        args = {str(s): eval_ns[str(s)] for s in expr.free_symbols}
        result = func(**args)
        return np.asarray(result, dtype=float)
    except Exception as exc:
        raise ValueError(
            f"[quickfit] Formula '{formula}' evaluated but produced an error: {exc}"
        ) from exc


# =====================================================================
# LIMIT VARIABLE INFERENCE — confidence-gated, not silent
# =====================================================================

@dataclass
class _DetectedLimitVar:
    name: str
    confidence: float
    method: str

def _infer_limit_variable(X: Dict[str, np.ndarray]) -> _DetectedLimitVar:
    """
    Heuristic guess: variable with the highest coefficient of variation
    (std/mean) is most likely to span a wide relative range -- and
    therefore most likely to contain the classical-limit information.

    HONESTY NOTE: this has no principled connection to the actual
    physics. It is a plausible weak proxy, nothing more. Confidence
    is deliberately capped at 0.7 regardless of how sharp the
    statistical signal looks, because the heuristic itself does not
    merit high confidence.
    """
    cvs = {
        col: float(np.std(arr) / (np.mean(np.abs(arr)) + 1e-30))
        for col, arr in X.items()
        if len(arr) > 1
    }
    if not cvs:
        first = next(iter(X))
        return _DetectedLimitVar(first, 0.0, "fallback_single_variable")

    ranked = sorted(cvs.items(), key=lambda kv: kv[1], reverse=True)
    best, best_cv = ranked[0]

    if len(ranked) >= 2:
        sep = (best_cv - ranked[1][1]) / (best_cv + 1e-12)
        conf = min(0.70, 0.40 + 0.30 * sep)
    else:
        conf = 0.40

    return _DetectedLimitVar(best, conf, "coefficient_of_variation")


# =====================================================================
# MAIN PUBLIC API
# =====================================================================

def quickfit(
    data,
    target: str,
    classical_formula: str,
    variables: Dict[str, str],
    constants: Optional[Dict[str, float]] = None,
    mode: str = "auto",
    limit_variable: Optional[str] = None,
    max_ratios: int = 4,
    seed: int = 42,
    noise_level: float = 0.0,
    max_iterations: int = 3,

    verbose: bool = True,
):
    """
    Discover the algebraic correction to a known classical law from data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data containing the columns declared in `variables` and `target`.
    target : str
        Name of the observed (anomalous) output column.
    classical_formula : str
        Arithmetic string of the classical law, e.g. ``"m * v"`` or
        ``"0.5 * m * v**2"``. Evaluated via sympy (safe mathematical parser).
    variables : dict
        Mapping of column names to physical dimension strings, e.g.
        ``{"m": "mass", "v": "velocity", "c": "velocity"}``.
        Supported: mass, length, time, velocity, energy, force, momentum,
        temperature, pressure, volume, density, dimensionless, and more.
    constants : dict, optional
        Scalar physical constants not present as columns, e.g. ``{"c": 3e8}``.
    mode : str
        ``"auto"`` (default), ``"multiplicative"``, or ``"additive"``.
        Auto-detection uses rank-correlation statistics with confidence reporting.
    limit_variable : str, optional
        The variable governing the classical limit (e.g. ``"v"`` for
        relativistic corrections). If None, inferred heuristically with a
        confidence score and explicit warning when confidence is low.
    max_ratios : int
        Max dimensionless ratio candidates to generate. Default 4.
        Each ratio expands to ~35 grammar candidates; 4 → ~140 total.
    seed : int
        Random seed passed to the core engine's multi-start optimizer.
        ADCD's search is deterministic given the same seed — this affects
        both synthetic data generation and optimizer restart initialisation.
    noise_level : float
        Noise level for synthetic data generation. Use 0.0 for real data.
    max_iterations : int
        Maximum orchestrator iterations. Default 1.
    verbose : bool
        Print mode, search space, and limit variable diagnostics.

    Returns
    -------
    ADCDResult
        Same result type as ``adcd.fit()``.

    Examples
    --------
    >>> import adcd, pandas as pd
    >>> df = pd.read_csv("relativistic_time.csv")
    >>> result = adcd.quickfit(
    ...     data=df,
    ...     target="t_obs",
    ...     classical_formula="t_0",
    ...     variables={"v": "velocity", "t_0": "time", "c": "velocity"},
    ...     constants={"c": 3e8},
    ... )
    >>> result.summary()

    Notes
    -----
    What quickfit cannot automate (irreducible inputs):
      - classical_formula: requires domain physics knowledge. Auto-detecting
        the correct classical law from column names would require an LLM —
        the exact risk this architecture avoids.
      - variables (dimension map): required for dimensionally valid ratio
        generation. Guessing from column names is unreliable and unverifiable.
    """
    import pandas as pd
    from adcd.mode_detection import detect_correction_mode
    from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget
    from adcd.api import fit

    # ------------------------------------------------------------------
    # 1. Validate and extract arrays from DataFrame
    # ------------------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "[quickfit] 'data' must be a pandas DataFrame. "
            "Load with pd.read_csv() or pd.DataFrame()."
        )

    if target not in data.columns:
        raise ValueError(
            f"[quickfit] Column '{target}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    missing_cols = [
        col for col in variables
        if col not in data.columns and col not in (constants or {})
    ]
    if missing_cols:
        raise ValueError(
            f"[quickfit] Columns {missing_cols} listed in variables are "
            f"not found in data or constants. "
            f"Available data columns: {list(data.columns)}"
        )

    y_obs = data[target].to_numpy(dtype=float)

    X: Dict[str, np.ndarray] = {}
    data_arrays: Dict[str, np.ndarray] = {}
    for col in variables:
        if col in data.columns:
            arr = data[col].to_numpy(dtype=float)
            X[col] = arr
            data_arrays[col] = arr

    if constants:
        for k, v in constants.items():
            arr = np.full_like(y_obs, v)
            data_arrays[k] = arr
            if k not in X:
                X[k] = arr

    # ------------------------------------------------------------------
    # 2. Evaluate classical formula — sympy, not eval()
    # ------------------------------------------------------------------
    if verbose:
        print(f"[quickfit] Classical formula: {classical_formula}")

    y_classical = _evaluate_formula(
        formula=classical_formula,
        variables=variables,
        data_arrays=data_arrays,
        constants=constants,
    )

    zero_frac = float(np.mean(np.abs(y_classical) < 1e-15))
    if zero_frac > 0.1:
        warnings.warn(
            f"[quickfit] {zero_frac:.0%} of y_classical values are near zero. "
            "Multiplicative residual will be undefined or very large at those points.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 3. Auto-detect correction mode (with confidence)
    # ------------------------------------------------------------------
    if mode == "auto":
        detected_mode, confidence = detect_correction_mode(y_obs, y_classical)
        if verbose:
            print(
                f"[quickfit] Mode → '{detected_mode}' "
                f"(confidence={confidence:.2f})"
            )
        if confidence < 0.6:
            warnings.warn(
                f"[quickfit] Mode detection confidence is low ({confidence:.2f}). "
                "Specify mode='multiplicative' or mode='additive' explicitly if unsure.",
                UserWarning,
                stacklevel=2,
            )
        correction_mode = detected_mode
    else:
        correction_mode = mode
        if verbose:
            print(f"[quickfit] Mode: '{correction_mode}' (user-specified)")

    # ------------------------------------------------------------------
    # 4. Auto-generate ratio symbols (dimensionally valid)
    # ------------------------------------------------------------------
    ratio_syms = _generate_ratio_symbols(variables, max_ratios=max_ratios)

    # Build ONE proposer with ALL ratios → single pipeline, BIC-correct
    proposer = MultiRatioProposer(
        ratio_symbols=ratio_syms,
        budget=GrammarBudget(),
    )
    n_total = proposer.search_space_size()

    if verbose:
        print(f"\n[quickfit] Ratio candidates ({len(ratio_syms)}):")
        for r in ratio_syms:
            print(f"  • {r}")
        print(
            f"[quickfit] Total grammar candidates (single BIC-corrected search): "
            f"{n_total}"
        )

    # ------------------------------------------------------------------
    # 5. Infer limit variable (confidence-gated, warns on low confidence)
    # ------------------------------------------------------------------
    if limit_variable is not None:
        if verbose:
            print(f"[quickfit] Classical limit variable: '{limit_variable}' (user-specified)")
        lv = limit_variable
    else:
        detected = _infer_limit_variable(X)
        lv = detected.name
        if verbose:
            print(
                f"[quickfit] Guessed classical limit variable: '{lv}' "
                f"(confidence={detected.confidence:.2f}, "
                f"method={detected.method})"
            )
        warnings.warn(
            f"[quickfit] `limit_variable` was not specified -- guessed as "
            f"'{lv}' with confidence={detected.confidence:.2f}. "
            f"This heuristic has NO guaranteed connection to the actual "
            f"physics of which variable governs the classical limit. A wrong "
            f"guess silently misconfigures the ARC gate with no error message. "
            f"For any result you plan to publish or rely on, call quickfit() "
            f"again with limit_variable='{lv}' (or the correct variable) "
            f"explicitly after inspecting this result.",
            UserWarning, stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 6. Run the core engine ONCE with MultiRatioProposer
    # ------------------------------------------------------------------
    if verbose:
        print("\n[quickfit] Running core ADCD engine (4-gate, BIC-ranked)...\n")

    result = fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        classical_expr=classical_formula,
        variables_with_units=variables,
        correction_mode=correction_mode,
        limit_variable=lv,
        proposer=proposer,
        max_iterations=max_iterations,
        seed=seed,
        noise_level=noise_level,
        verbose=False,
    )

    if verbose:
        print("[quickfit] Done. Call result.summary() to view results.\n")

    return result
