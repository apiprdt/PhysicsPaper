"""
Tests for Phase 3: Blind three-tier verdict, equal pools self-ablation, and ground-truth independence.
"""

import os
import re
import numpy as np

from adcd.validation_v2 import (
    ProtocolVerdict,
    CandidateFit,
    run_three_tier_audit,
)


def test_three_tier_uniquely_identifiable():
    """When best candidate beats all alternative families by Delta-BIC > 10, verdict is UNIQUELY IDENTIFIABLE."""
    # M1 is exponential, M2 is rational with Delta-BIC = 25.0
    cands = [
        CandidateFit(
            expr="theta_0 * (exp(-r/theta_1) - 1.0)",
            bic=-1600.0, nmse=2e-4, k_params=2, family="exponential"
        ),
        CandidateFit(
            expr="theta_0 * (r/theta_1 / (1.0 - r/theta_1))",
            bic=-1575.0, nmse=3e-3, k_params=2, family="rational"
        ),
        CandidateFit(
            expr="theta_0 * log(1.0 + r/theta_1)",
            bic=-1570.0, nmse=4e-3, k_params=2, family="logarithmic"
        ),
    ]
    null_fit = {"bic": -100.0, "nmse": 1.0}

    verdict_res = run_three_tier_audit(
        candidates=cands,
        null_fit=null_fit,
        search_space_size=60,
    )

    assert verdict_res.verdict == ProtocolVerdict.UNIQUELY_IDENTIFIABLE
    assert verdict_res.delta_bic_vs_null > 10.0
    assert verdict_res.delta_bic_vs_rival == 25.0
    assert len(verdict_res.equivalence_class) == 1
    assert verdict_res.best_candidate.expr == cands[0].expr


def test_three_tier_equivalence_class():
    """When detected but rival family is within 10 BIC units, verdict is EQUIVALENCE CLASS."""
    # M1 is log, M2 is exp with Delta-BIC = 2.5 <= 10.0
    cands = [
        CandidateFit(expr="theta_0 * log(1.0 + x)", bic=-600.0, nmse=0.046, k_params=1, family="logarithmic"),
        CandidateFit(expr="theta_0 * (exp(-x) - 1.0)", bic=-597.5, nmse=0.046, k_params=1, family="exponential"),
        CandidateFit(expr="theta_0 * tanh(x)", bic=-597.0, nmse=0.046, k_params=1, family="trigonometric"),
    ]
    null_fit = {"bic": -100.0, "nmse": 1.0}

    verdict_res = run_three_tier_audit(
        candidates=cands,
        null_fit=null_fit,
        search_space_size=184,
    )

    assert verdict_res.verdict == ProtocolVerdict.EQUIVALENCE_CLASS
    assert verdict_res.delta_bic_vs_null > 10.0
    assert verdict_res.delta_bic_vs_rival == 2.5
    assert len(verdict_res.equivalence_class) == 3
    # Shared Taylor series explanation must be present
    assert verdict_res.shared_series_explanation != ""


def test_three_tier_withheld():
    """When best candidate does not beat the null baseline by Delta-BIC > 10, verdict is WITHHELD."""
    # Weak signal: Delta-BIC vs null is only 4.0 <= 10.0
    cands = [
        CandidateFit(expr="theta_0 * (r / (1.0 - r))", bic=-104.0, nmse=0.95, k_params=1, family="rational"),
        CandidateFit(expr="theta_0 * (exp(-r) - 1.0)", bic=-102.0, nmse=0.97, k_params=1, family="exponential"),
    ]
    null_fit = {"bic": -100.0, "nmse": 1.0}

    verdict_res = run_three_tier_audit(
        candidates=cands,
        null_fit=null_fit,
        search_space_size=10,
    )

    assert verdict_res.verdict == ProtocolVerdict.WITHHELD
    assert verdict_res.delta_bic_vs_null == 4.0


def test_equal_pool_cancels_search_space_penalty():
    """Under equal pool protocol, changing the pool size penalty leaves Delta-BIC(M2 -> M1) invariant."""
    cands = [
        CandidateFit(expr="M1", bic=0.0, nmse=0.01, k_params=1, family="fam1"),
        CandidateFit(expr="M2", bic=15.0, nmse=0.02, k_params=1, family="fam2"),
    ]
    null_fit = {"bic": 100.0, "nmse": 1.0}

    res_pool_10 = run_three_tier_audit(cands, null_fit, search_space_size=10)
    res_pool_1000 = run_three_tier_audit(cands, null_fit, search_space_size=1000)

    # In equal pool self-ablation, rival comparison must be completely invariant to pool size
    assert res_pool_10.delta_bic_vs_rival == res_pool_1000.delta_bic_vs_rival == 15.0


def test_no_ground_truth_in_validation_v2():
    """
    CRITICAL PROTOCOL INTEGRITY TEST:
    validation_v2.py must NOT reference ground truth, correction_expr, or true_primitive anywhere in its code.
    """
    filepath = os.path.join(os.path.dirname(__file__), "..", "src", "adcd", "validation_v2.py")
    assert os.path.exists(filepath), f"Missing validation_v2.py at {filepath}"

    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    forbidden_patterns = [
        r"\bcorrection_expr\b",
        r"\btrue_primitive\b",
        r"\bground_truth\b",
        r"\b_hidden\b",
        r"\bmatch_level\b",
    ]

    for pat in forbidden_patterns:
        matches = re.findall(pat, code)
        assert len(matches) == 0, (
            f"Protocol integrity violation: found forbidden ground-truth reference '{pat}' "
            f"in validation_v2.py ({len(matches)} occurrences)."
        )


def test_optional_adequacy_spline_warning():
    """When check_adequacy is enabled, a structured misfit against a flexible spline issues a warning."""
    x = np.linspace(0.1, 5.0, 100)
    # Target is an oscillatory wave that a monotonic line cannot fit
    y_true = np.sin(x)
    y_obs = y_true + np.random.RandomState(42).normal(0, 0.01, size=len(x))
    sigma_y = np.full_like(y_obs, 0.01)

    cands = [
        CandidateFit(
            expr="theta_0 * x",
            bic=-50.0,
            nmse=0.5,
            k_params=1,
            family="linear",
            y_pred=0.1 * x,
        )
    ]
    null_fit = {"bic": 0.0, "nmse": 1.0}

    verdict_res = run_three_tier_audit(
        candidates=cands,
        null_fit=null_fit,
        search_space_size=10,
        check_adequacy=True,
        adequacy_data={"x": x, "y_obs": y_obs, "sigma_y": sigma_y},
    )

    assert verdict_res.adequacy_warning != ""
    assert "Adequacy Warning" in verdict_res.adequacy_warning
    # Crucially, adequacy warning does not veto the verdict
    assert verdict_res.verdict in [ProtocolVerdict.UNIQUELY_IDENTIFIABLE, ProtocolVerdict.EQUIVALENCE_CLASS]


def test_three_tier_custom_unlabeled_candidates_fallback():
    """Verify that imported candidates with family='unknown' (e.g. from PySR) don't falsely yield delta_bic = inf."""
    cands = [
        CandidateFit(expr="0.187 * x", bic=-500.0, nmse=0.01, k_params=1, family="unknown"),
        CandidateFit(expr="0.185 * x + 0.05 * x**2", bic=-498.0, nmse=0.01, k_params=2, family="unknown"),
    ]
    null_fit = {"bic": 0.0, "nmse": 1.0}

    res = run_three_tier_audit(candidates=cands, null_fit=null_fit, search_space_size=2)
    # delta_bic_vs_rival should be 2.0 (not inf!), leading to EQUIVALENCE CLASS
    assert res.delta_bic_vs_rival == 2.0
    assert res.verdict == ProtocolVerdict.EQUIVALENCE_CLASS
    assert res.rival_candidate.expr == cands[1].expr
