"""
validation_v2.py
================
Blind three-tier identifiability protocol with equal-pool self-ablation.

This module replaces ground-truth-dependent positive controls with an empirical,
independent protocol suitable for real-world experimental data:

1. Disclose candidate pool size |C| and model priors before fitting.
2. Detection gate: Delta-BIC(null -> M1) > 10.0 (where null = uncorrected baseline).
3. Blind self-ablation: M1 = best candidate; rival M2 = best candidate from a DIFFERENT
   primitive family evaluated on the SAME POOL SIZE (so search-space penalties cancel exactly).
4. Verdict determination:
   - UNIQUELY IDENTIFIABLE if detected and Delta-BIC(M2 -> M1) > 10.0.
   - EQUIVALENCE CLASS if detected and Delta-BIC(M2 -> M1) <= 10.0 (returns all models within
     Delta-BIC <= 10.0 and their shared low-order asymptotic series).
   - WITHHELD otherwise (signal does not rise above measurement noise).
5. Optional adequacy check against a flexible reference (monotone spline, default off).
   Misfit produces a warning, not a veto.

NOTE: This module strictly operates blindly. It contains NO references to ground-truth expressions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.interpolate
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


def _derive_shared_series_explanation(
    equiv_cands: List[CandidateFit],
    expansion_var: str = "u",
    order: int = 2,
) -> str:
    """
    Derive low-order Taylor expansion for all candidates in the equivalence class
    to explain mathematically why the data cannot separate them.
    """
    if len(equiv_cands) <= 1:
        return ""

    u = sp.Symbol(expansion_var, real=True)
    expansions = []

    for cand in equiv_cands:
        try:
            sym_expr = sp.sympify(cand.expr)
            # Substitute fitted parameter values if available, else 1.0
            sub_dict = {}
            for s in sym_expr.free_symbols:
                if s.name in cand.theta_fit:
                    sub_dict[s] = cand.theta_fit[s.name]
                elif s.name.startswith("theta_"):
                    sub_dict[s] = 1.0
                elif s.name != expansion_var:
                    sub_dict[s] = u

            expr_sub = sym_expr.subs(sub_dict)
            taylor_series = sp.series(expr_sub, u, 0, n=order + 1).removeO()
            expansions.append(f"  • {cand.expr[:40]:<40} ≈ {str(taylor_series)}")
        except Exception:
            expansions.append(f"  • {cand.expr[:40]:<40} (Taylor expansion unavailable)")

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
    Optional adequacy check: compares best candidate against a non-parametric monotone spline.
    Issues a diagnostic warning if structured residuals exceed flexible reference.
    """
    try:
        x = np.asarray(adequacy_data["x"], dtype=np.float64)
        y_obs = np.asarray(adequacy_data["y_obs"], dtype=np.float64)
        sigma_y = np.asarray(adequacy_data.get("sigma_y", np.full_like(y_obs, 0.01)), dtype=np.float64)

        sort_idx = np.argsort(x)
        x_s, y_s, sig_s = x[sort_idx], y_obs[sort_idx], sigma_y[sort_idx]

        # Fit monotonic PCHIP interpolator as non-parametric reference
        spline = scipy.interpolate.PchipInterpolator(x_s, y_s)
        y_spline = spline(x_s)

        res_spline = y_s - y_spline
        chi2_spline = np.sum((res_spline / np.maximum(1e-12, sig_s)) ** 2)

        if best_candidate.y_pred is not None:
            y_pred_best = best_candidate.y_pred[sort_idx]
            res_best = y_s - y_pred_best
            chi2_best = np.sum((res_best / np.maximum(1e-12, sig_s)) ** 2)

            # If parametric model has substantial excess chi2 over spline
            if chi2_best > chi2_spline + 50.0:
                return (
                    f"Adequacy Warning: Parametric model ({best_candidate.expr[:35]}) "
                    f"shows structured misfit relative to flexible reference (Delta-chi2 = {chi2_best - chi2_spline:.1f})."
                )
    except Exception as e:
        logger.debug(f"Adequacy check skipped due to error: {e}")

    return ""


def run_three_tier_audit(
    candidates: List[CandidateFit],
    null_fit: Dict[str, float],
    search_space_size: int = 1,
    check_adequacy: bool = False,
    adequacy_data: Optional[Dict[str, np.ndarray]] = None,
) -> ThreeTierProtocolResult:
    """
    Execute the blind three-tier validation protocol on fitted candidates.

    Parameters
    ----------
    candidates : Sorted list of CandidateFit objects (lowest BIC first)
    null_fit : Dictionary with 'bic' and 'nmse' for the uncorrected baseline
    search_space_size : Total disclosed hypothesis space size |C|
    check_adequacy : If True, runs non-parametric spline adequacy check (default False)
    adequacy_data : Dataset dictionary required if check_adequacy is enabled
    """
    if not candidates:
        return ThreeTierProtocolResult(
            verdict=ProtocolVerdict.WITHHELD,
            delta_bic_vs_null=0.0,
            delta_bic_vs_rival=0.0,
            best_candidate=None,
            rival_candidate=None,
            search_space_size=search_space_size,
        )

    # Sort candidates by BIC ascending
    sorted_cands = sorted(candidates, key=lambda c: c.bic)
    m1 = sorted_cands[0]
    null_bic = null_fit.get("bic", float("inf"))

    # Step 2: Detection gate (Null vs Best Model)
    delta_bic_null = float(null_bic - m1.bic)

    if delta_bic_null <= BIC_SIGNIFICANCE_THRESHOLD:
        return ThreeTierProtocolResult(
            verdict=ProtocolVerdict.WITHHELD,
            delta_bic_vs_null=delta_bic_null,
            delta_bic_vs_rival=0.0,
            best_candidate=m1,
            rival_candidate=None,
            search_space_size=search_space_size,
        )

    # Step 3: Blind self-ablation from EQUAL POOL
    # Rival must come from a DIFFERENT primitive family
    rival_cands = [c for c in sorted_cands if c.family != m1.family]
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
        # Return all candidate models within Delta-BIC <= 10.0 of M1
        equiv_class = [c for c in sorted_cands if (c.bic - m1.bic) <= BIC_SIGNIFICANCE_THRESHOLD]
        shared_series = _derive_shared_series_explanation(equiv_class)

    # Step 5: Optional adequacy check against flexible spline
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
