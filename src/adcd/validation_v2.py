"""
validation_v2.py
================
Blind three-tier identifiability protocol with equal-pool self-ablation.

This module replaces ground-truth-dependent positive controls with an empirical,
independent protocol suitable for real-world experimental data:

1. Disclose candidate pool size |C| and model priors before fitting.
2. Filter out structurally degenerate candidates (is_degenerate=True) before selection.
3. Detection gate: Delta-BIC(null -> M1) > 10.0 (where null = uncorrected baseline).
4. Blind self-ablation: M1 = best candidate; rival M2 = best candidate from a DIFFERENT
   primitive family evaluated on the SAME POOL SIZE (so search-space penalties cancel).
5. Verdict determination:
   - UNIQUELY IDENTIFIABLE if detected and Delta-BIC(M2 -> M1) > 10.0.
   - EQUIVALENCE CLASS if detected and Delta-BIC(M2 -> M1) <= 10.0 (returns all models within
     Delta-BIC <= 10.0 and their shared low-order asymptotic series).
   - WITHHELD otherwise (signal does not rise above measurement noise or all candidates degenerate).
6. Optional adequacy check against a flexible reference (isotonic monotone regression,
   default off). Misfit produces a warning, not a veto.

NOTE: This module strictly operates blindly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.optimize
import sympy as sp

logger = logging.getLogger(__name__)

BIC_SIGNIFICANCE_THRESHOLD = 10.0


class ProtocolVerdict(str, Enum):
    UNIQUELY_IDENTIFIABLE = "UNIQUELY IDENTIFIABLE"
    EQUIVALENCE_CLASS = "EQUIVALENCE CLASS"
    WITHHELD = "WITHHELD"


@dataclass
class CandidateFit:
    """Fitted candidate model representation for validation."""
    expr: str
    bic: float
    nmse: float
    k_params: int = 1
    family: str = "unknown"
    theta_fit: Dict[str, float] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    is_degenerate: bool = False
    origin_type: str = "literature"
    citation: str = ""
    is_novel_claim: bool = False


@dataclass
class ThreeTierProtocolResult:
    """Formal audit verdict and diagnostics under the blind protocol."""
    verdict: ProtocolVerdict
    delta_bic_vs_null: float
    delta_bic_vs_rival: float
    best_candidate: Optional[CandidateFit]
    rival_candidate: Optional[CandidateFit]
    equivalence_class: List[CandidateFit] = field(default_factory=list)
    shared_series_explanation: str = ""
    adequacy_warning: str = ""
    search_space_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "delta_bic_vs_null": float(self.delta_bic_vs_null),
            "delta_bic_vs_rival": float(self.delta_bic_vs_rival),
            "best_candidate": self.best_candidate.expr if self.best_candidate else None,
            "rival_candidate": self.rival_candidate.expr if self.rival_candidate else None,
            "equivalence_class": [c.expr for c in self.equivalence_class],
            "shared_series_explanation": self.shared_series_explanation,
            "adequacy_warning": self.adequacy_warning,
            "search_space_size": self.search_space_size,
        }


def infer_family_from_expr(expr_str: str) -> str:
    """
    Heuristically infer the structural primitive family of an expression string
    when not explicitly supplied in ClaimCard.candidate_families.
    """
    s = expr_str.replace(" ", "").lower()
    if "d_lor" in s or ("sqrt(1-" in s and "+1)" in s):
        return "D_lor"
    if "d_exp" in s or "exp(" in s:
        return "D_exp"
    if "d_log" in s or "log(" in s or "ln(" in s:
        return "D_log"
    if "d_osc" in s or "cos(" in s or "sin(" in s:
        return "D_osc"
    if "d_sat" in s or "tanh(" in s or "atan(" in s:
        return "D_sat"
    if "d_sqrt_inv" in s or "1/sqrt(" in s or "sqrt(" in s:
        return "D_sqrt_inv"
    if "d_rat" in s or "/(1-" in s or "/(1+" in s:
        return "D_rat"
    if "d_pow" in s or "**" in s or "^" in s:
        return "D_pow"
    return f"custom:{expr_str.strip()}"


def _derive_shared_series_explanation(
    equiv_cands: List[CandidateFit],
    expansion_var: str = "u",
    limit_dir: str = "0",
    order: int = 3,
) -> str:
    """
    Derive low-order Taylor/asymptotic expansion for all candidates in the equivalence class
    to explain mathematically why the data cannot separate them.
    """
    if len(equiv_cands) <= 1:
        return ""

    u = sp.Symbol(expansion_var, real=True, positive=True)
    x0 = sp.oo if limit_dir in ("oo", "inf", "+oo") else 0
    expansions = []

    for cand in equiv_cands:
        try:
            sym_expr = sp.sympify(cand.expr)
            free_syms = sorted(sym_expr.free_symbols, key=lambda s: s.name)

            # Separate fitted parameters from physical variables
            param_syms = [
                s for s in free_syms
                if s.name in cand.theta_fit or s.name.startswith("theta")
            ]
            var_syms = [s for s in free_syms if s not in param_syms]

            sub_dict = {}
            for s in param_syms:
                sub_dict[s] = float(cand.theta_fit.get(s.name, 1.0))

            # Map the primary limit variable to u, and any auxiliary scales to 1.0
            if any(s.name == expansion_var for s in var_syms):
                for s in var_syms:
                    sub_dict[s] = u if s.name == expansion_var else 1.0
            elif len(var_syms) >= 1:
                sub_dict[var_syms[0]] = u
                for s in var_syms[1:]:
                    sub_dict[s] = 1.0

            expr_sub = sym_expr.subs(sub_dict)
            taylor_series = sp.series(expr_sub, u, x0, n=order + 1).removeO()
            expansions.append(f"  • {cand.expr[:40]:<40} ≈ {str(taylor_series)}")
        except Exception:
            expansions.append(f"  • {cand.expr[:40]:<40} (Series expansion unavailable)")

    header = (
        f"Asymptotic Series Explanation:\n"
        f"The {len(equiv_cands)} candidates in this equivalence class agree at leading orders:\n"
    )
    return header + "\n".join(expansions)


def _check_flexible_adequacy(
    best_candidate: CandidateFit,
    adequacy_data: Dict[str, np.ndarray],
) -> str:
    """
    Optional adequacy check: compares best candidate against a non-parametric
    isotonic monotone regression reference (trying both increasing and decreasing).
    Unlike an exact interpolator, isotonic regression fits only the monotone signal
    component without zeroing out random noise.
    """
    try:
        if best_candidate.y_pred is None:
            return ""

        x = np.asarray(adequacy_data["x"], dtype=np.float64)
        y_obs = np.asarray(adequacy_data["y_obs"], dtype=np.float64)
        sigma_y = np.asarray(
            adequacy_data.get("sigma_y", np.full_like(y_obs, 0.01)), dtype=np.float64
        )
        N = len(y_obs)
        if N < 10:
            return ""

        sort_idx = np.argsort(x)
        y_s = y_obs[sort_idx]
        sig_s = np.maximum(1e-12, sigma_y[sort_idx])
        weights = 1.0 / (sig_s ** 2)

        # Evaluate both monotone increasing and monotone decreasing isotonic regressions
        iso_inc = scipy.optimize.isotonic_regression(y_s, weights=weights, increasing=True)
        iso_dec = scipy.optimize.isotonic_regression(y_s, weights=weights, increasing=False)

        chi2_inc = float(np.sum(((y_s - iso_inc.x) / sig_s) ** 2))
        chi2_dec = float(np.sum(((y_s - iso_dec.x) / sig_s) ** 2))

        if chi2_inc <= chi2_dec:
            chi2_iso = chi2_inc
            k_iso = len(np.unique(np.round(iso_inc.x, decimals=10)))
        else:
            chi2_iso = chi2_dec
            k_iso = len(np.unique(np.round(iso_dec.x, decimals=10)))

        y_pred_s = np.asarray(best_candidate.y_pred, dtype=np.float64)[sort_idx]
        chi2_best = float(np.sum(((y_s - y_pred_s) / sig_s) ** 2))

        # Penalize isotonic effective degrees of freedom (number of constant blocks) via AIC/BIC-like buffer
        dof_buffer = max(0, k_iso - best_candidate.k_params) * 2.0 + 25.0

        if chi2_best > chi2_iso + dof_buffer:
            excess = chi2_best - chi2_iso
            return (
                f"Adequacy Warning: Parametric model ({best_candidate.expr[:35]}) "
                f"shows structured misfit relative to non-parametric monotone reference "
                f"(excess chi2 = {excess:.1f}, iso blocks = {k_iso})."
            )
    except Exception as e:
        logger.debug(f"Adequacy check skipped due to error: {e}")

    return ""


def run_three_tier_audit(
    candidates: List[CandidateFit],
    null_fit: Optional[Union[Dict[str, float], float]] = None,
    search_space_size: int = 1,
    check_adequacy: bool = False,
    adequacy_data: Optional[Dict[str, np.ndarray]] = None,
    expansion_var: str = "u",
    limit_dir: str = "0",
    null_bic: Optional[float] = None,
    u_var: Optional[str] = None,
    u_data: Optional[np.ndarray] = None,
    y_obs: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> ThreeTierProtocolResult:
    """
    Execute the blind three-tier validation protocol on fitted candidates.
    Supports both dictionary `null_fit={'bic': ...}` and scalar `null_bic=...`.
    """
    if u_var is not None:
        expansion_var = u_var

    if null_bic is not None:
        resolved_null_bic = float(null_bic)
    elif isinstance(null_fit, dict):
        resolved_null_bic = float(null_fit.get("bic", float("inf")))
    elif isinstance(null_fit, (int, float, np.floating)):
        resolved_null_bic = float(null_fit)
    else:
        resolved_null_bic = float("inf")

    if adequacy_data is None and u_data is not None and y_obs is not None:
        adequacy_data = {
            "x": np.asarray(u_data, dtype=np.float64),
            "y_obs": np.asarray(y_obs, dtype=np.float64),
        }
        if sigma is not None:
            adequacy_data["sigma_y"] = np.asarray(sigma, dtype=np.float64)

    if not candidates:
        return ThreeTierProtocolResult(
            verdict=ProtocolVerdict.WITHHELD,
            delta_bic_vs_null=0.0,
            delta_bic_vs_rival=0.0,
            best_candidate=None,
            rival_candidate=None,
            search_space_size=search_space_size,
        )

    # Sort all candidates by BIC ascending, then filter out degenerate candidates
    sorted_all = sorted(candidates, key=lambda c: c.bic)
    sorted_cands = [c for c in sorted_all if not c.is_degenerate]

    if not sorted_cands:
        # All candidates were flagged as degenerate
        m_degen = sorted_all[0]
        return ThreeTierProtocolResult(
            verdict=ProtocolVerdict.WITHHELD,
            delta_bic_vs_null=float(resolved_null_bic - m_degen.bic),
            delta_bic_vs_rival=0.0,
            best_candidate=m_degen,
            rival_candidate=None,
            search_space_size=search_space_size,
        )

    m1 = sorted_cands[0]

    # Step 2: Detection gate (Null vs Best Non-Degenerate Model)
    delta_bic_null = float(resolved_null_bic - m1.bic)

    if delta_bic_null <= BIC_SIGNIFICANCE_THRESHOLD:
        return ThreeTierProtocolResult(
            verdict=ProtocolVerdict.WITHHELD,
            delta_bic_vs_null=delta_bic_null,
            delta_bic_vs_rival=0.0,
            best_candidate=m1,
            rival_candidate=None,
            search_space_size=search_space_size,
        )

    # Step 3: Blind self-ablation from EQUAL POOL of non-degenerate candidates
    # If all candidates have family="unknown", treat each distinct expression as its own rival family
    if m1.family != "unknown":
        rival_cands = [c for c in sorted_cands if c.family != m1.family]
    else:
        rival_cands = [c for c in sorted_cands[1:] if c.expr != m1.expr]

    if not rival_cands and len(sorted_cands) > 1:
        m2 = next((c for c in sorted_cands[1:] if c.expr != m1.expr), sorted_cands[1])
    else:
        m2 = rival_cands[0] if rival_cands else None

    if m2 is not None:
        delta_bic_rival = float(m2.bic - m1.bic)
    else:
        delta_bic_rival = float("inf")

    # Step 4: Three-tier verdict determination
    if delta_bic_rival > BIC_SIGNIFICANCE_THRESHOLD:
        verdict = ProtocolVerdict.UNIQUELY_IDENTIFIABLE
        equiv_class = [m1]
        shared_series = ""
    else:
        verdict = ProtocolVerdict.EQUIVALENCE_CLASS
        equiv_class = [
            c for c in sorted_cands if (c.bic - m1.bic) <= BIC_SIGNIFICANCE_THRESHOLD
        ]
        shared_series = _derive_shared_series_explanation(
            equiv_class, expansion_var=expansion_var, limit_dir=limit_dir
        )

    # Step 5: Optional adequacy check against flexible monotone reference
    adequacy_warn = ""
    if check_adequacy and adequacy_data is not None:
        adequacy_warn = _check_flexible_adequacy(m1, adequacy_data)

    return ThreeTierProtocolResult(
        verdict=verdict,
        delta_bic_vs_null=delta_bic_null,
        delta_bic_vs_rival=delta_bic_rival,
        best_candidate=m1,
        rival_candidate=m2,
        equivalence_class=equiv_class,
        shared_series_explanation=shared_series,
        adequacy_warning=adequacy_warn,
        search_space_size=search_space_size,
    )
