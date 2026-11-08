"""
tests/test_crossover_spike.py
=============================
Unit tests for Phase 6 Crossover Spike (`spike/crossover.py`).

Verifies:
1. Exact numerical limits at u -> 0 (including u = 1e-16) and u -> oo for all 5
   convex crossover families and 4 smooth-max/min CES families (Criterion 4).
2. Exact numerical limits at u -> 0 and u -> oo for Planck, Debye, Michaelis-Menten,
   and RAR physical testbeds.
3. Exact reproduction of Phase 6 Preliminary Results from the Brief:
   - Planck Asimov chi^2 (96.4, 1605, 8770, 772494, and v1 sat 1399).
   - RAR crossing-pole detection at x = 1 and 2-param smooth-max RMS = 0.00995 dex.
   - Debye crossing-pole detection at u_c = (4*pi^4/5)^(1/3) ~ 4.2709 and alpha = 3.
4. Locked two-limit vs free one-limit Delta-BIC hypothesis test (Question 3).
5. Single-seed three-tier crossover audit behavior on Michaelis-Menten, Planck, and Debye.
"""

from __future__ import annotations

import numpy as np

from spike.crossover import (
    ALL_CONVEX_FAMILIES,
    audit_crossover_case_single_seed,
    debye_exact,
    eval_convex_crossover,
    eval_smooth_max,
    eval_smooth_min,
    michaelis_menten_exact,
    planck_exact,
    planck_true_weight,
    rar_exact,
    reproduce_preliminary_results,
    evaluate_second_limit_hypothesis,
    verify_crossover_limits,
)


def test_convex_weights_small_and_large_limits() -> None:
    """Every convex crossover family must verify both u -> 0 (down to 1e-16) and u -> oo limits."""
    u_tiny = np.array([0.0, 1e-16, 1e-12, 1e-8], dtype=np.float64)
    u_huge = np.array([1e6, 1e9, 1e12], dtype=np.float64)
    A_tiny = np.full_like(u_tiny, 2.5)
    B_tiny = np.full_like(u_tiny, 0.3)
    A_huge = np.full_like(u_huge, 2.5)
    B_huge = np.full_like(u_huge, 0.3)

    for fam in ALL_CONVEX_FAMILIES:
        y_tiny = eval_convex_crossover(u_tiny, A_tiny, B_tiny, fam, s=1.2, p=1.4)
        y_huge = eval_convex_crossover(u_huge, A_huge, B_huge, fam, s=1.2, p=1.4)
        assert np.all(np.isfinite(y_tiny)), f"{fam} produced non-finite values at tiny u"
        assert np.all(np.isfinite(y_huge)), f"{fam} produced non-finite values at huge u"
        assert np.isclose(y_tiny[0], 2.5, rtol=1e-15, atol=1e-15)
        assert np.isclose(y_tiny[1], 2.5, rtol=1e-12, atol=1e-12)
        assert np.allclose(y_huge, 0.3, rtol=1e-4, atol=1e-4)

        passed, err_small, err_large = verify_crossover_limits(
            eval_fn=lambda u, f=fam: eval_convex_crossover(u, np.full_like(u, 2.5), np.full_like(u, 0.3), f, s=1.0, p=1.3),
            A_fn=lambda u: np.full_like(u, 2.5),
            B_fn=lambda u: np.full_like(u, 0.3),
            u_small=1e-7,
            u_large=1e5,
            rtol_small=1e-4,
            rtol_large=1e-4,
        )
        assert passed, f"{fam} failed limit verification: err_small={err_small}, err_large={err_large}"


def test_smooth_ces_limits() -> None:
    """Smooth-max and smooth-min CES families must verify both asymptotic limits."""
    # For A(u) = 1, B(u) = u^(-1/2) (like RAR):
    # As u -> 0, B(u) >> A(u), so smooth_max(A, s*B) -> s*B(u)
    # As u -> oo, A(u) >> B(u), so smooth_max(A, s*B) -> A(u)
    passed_max, err_s_max, err_l_max = verify_crossover_limits(
        eval_fn=lambda u: eval_smooth_max(np.ones_like(u), u ** (-0.5), n=1.7, s=1.0),
        A_fn=lambda u: u ** (-0.5),
        B_fn=lambda u: np.ones_like(u),
        u_small=1e-8,
        u_large=1e8,
        rtol_small=1e-3,
        rtol_large=1e-3,
    )
    assert passed_max, f"smooth_max failed limits: {err_s_max}, {err_l_max}"

    # For A(u) = 1, B(u) = u^(-3) (like Debye):
    # As u -> 0, B(u) -> oo so smooth_min(A, s*B) -> A(u) = 1
    # As u -> oo, B(u) -> 0 so smooth_min(A, s*B) -> s*B(u)
    passed_min, err_s_min, err_l_min = verify_crossover_limits(
        eval_fn=lambda u: eval_smooth_min(np.ones_like(u), u ** (-3.0), n=1.2, s=1.0),
        A_fn=lambda u: np.ones_like(u),
        B_fn=lambda u: u ** (-3.0),
        u_small=1e-6,
        u_large=1e5,
        rtol_small=1e-3,
        rtol_large=1e-3,
    )
    assert passed_min, f"smooth_min failed limits: {err_s_min}, {err_l_min}"


def test_solved_physical_systems_limits_and_precision() -> None:
    """Verify Planck, Debye, Michaelis-Menten, and RAR at u = 1e-16, u -> 0, and u -> oo."""
    # 1. Planck at x = 1e-16
    x_tiny = np.array([1e-16, 1e-12, 1e-8], dtype=np.float64)
    y_p_tiny = planck_exact(x_tiny)
    assert np.all(np.isfinite(y_p_tiny))
    assert np.isclose(y_p_tiny[0], 1.0 - 0.5e-16, rtol=1e-15, atol=1e-15)

    w_p = planck_true_weight(np.logspace(-6, 2, 100))
    assert np.all((w_p >= 0.0) & (w_p <= 1.0))

    # 2. Debye at u -> 0 (Dulong-Petit = 1) and u -> oo (4*pi^4 / (5*u^3))
    u_deb_small = np.array([1e-16, 1e-8, 1e-4], dtype=np.float64)
    y_deb_small = debye_exact(u_deb_small)
    assert np.allclose(y_deb_small, 1.0, rtol=1e-7, atol=1e-7)

    u_deb_large = np.array([50.0, 100.0], dtype=np.float64)
    y_deb_large = debye_exact(u_deb_large)
    b_deb_large = (4.0 * (np.pi ** 4)) / (5.0 * (u_deb_large ** 3))
    assert np.allclose(y_deb_large, b_deb_large, rtol=1e-4)

    # 3. Michaelis-Menten at u -> 0 and u -> oo
    passed_mm, _, _ = verify_crossover_limits(
        eval_fn=michaelis_menten_exact,
        A_fn=lambda u: u,
        B_fn=lambda u: np.ones_like(u),
        u_small=1e-7,
        u_large=1e7,
        rtol_small=1e-5,
        rtol_large=1e-5,
    )
    assert passed_mm

    # 4. RAR at x -> 0 (x^-1/2) and x -> oo (1)
    passed_rar, _, _ = verify_crossover_limits(
        eval_fn=rar_exact,
        A_fn=lambda x: x ** (-0.5),
        B_fn=lambda x: np.ones_like(x),
        u_small=1e-9,
        u_large=1e4,
        rtol_small=1e-3,
        rtol_large=1e-3,
    )
    assert passed_rar


def test_reproduce_phase6_preliminary_numbers() -> None:
    """
    Verify exact reproduction of the Phase 6 Preliminary Results in the Brief:
    - Planck Asimov chi^2: convex_exp_p = 96.4, convex_tanh = 1605, convex_exp = 8770,
      convex_rat = 772494, v1 sat = 1399.
    - RAR: crossing pole at x = 1 (convex invalid) and 2-param smooth-max RMS = 0.0099 dex.
    - Debye: crossing pole at u_c = 4.2709 and small-u divergence alpha = 3.0.
    """
    res = reproduce_preliminary_results()

    # Planck checks
    p = res["planck"]
    assert p["validity"].is_valid is True
    assert p["validity"].has_crossing_pole is False
    assert np.isclose(p["convex_exp_p_chi2"], 96.40, atol=0.5)
    assert np.isclose(p["convex_tanh_chi2"], 1605.17, atol=1.0)
    assert np.isclose(p["convex_exp_chi2"], 8770.07, atol=2.0)
    assert np.isclose(p["convex_rat_chi2"], 772493.57, atol=20.0)
    assert np.isclose(p["v1_sat_chi2"], 1399.25, atol=1.0)
    # Criterion 3 on Planck: best crossover beats v1 one-limit
    assert p["convex_exp_p_chi2"] < p["v1_sat_chi2"]

    # RAR checks
    r = res["rar"]
    assert r["validity"].is_valid is False
    assert r["validity"].has_crossing_pole is True
    assert r["validity"].crossing_u is not None
    assert np.isclose(r["validity"].crossing_u, 1.0, atol=0.02)
    assert r["validity"].w_req_min < -10.0
    assert r["validity"].w_req_max > 10.0
    assert np.isclose(r["smooth_max_2p_rms_dex"], 0.00995, atol=0.0005)

    # Debye checks
    d = res["debye"]
    assert d["validity_t3"].is_valid is False
    assert d["validity_t3"].has_crossing_pole is True
    expected_uc = float(((4.0 * (np.pi ** 4)) / 5.0) ** (1.0 / 3.0))
    assert np.isclose(d["validity_t3"].crossing_u, expected_uc, atol=0.05)
    assert np.isclose(d["validity_t3"].small_u_divergence_exponent, 3.0, atol=0.05)
    assert d["smooth_min_2p_chi2"] < d["v1_sat_chi2"]


def test_locked_vs_free_second_limit_hypothesis() -> None:
    """
    Question 3: A second limit should be a testable hypothesis via Delta-BIC(free - locked).
    - True second limit B = x * exp(-x) on Planck is confirmed (Delta-BIC > +10).
    - Misspecified second limit B = 0.5 on Planck is decisively rejected (Delta-BIC < -10).
    """
    x = np.logspace(-2.0, 1.0, 200)
    y = planck_exact(x)
    sigma = 0.01 * y
    A = np.ones_like(x)
    B_true = x * np.exp(-x)
    B_wrong = np.full_like(x, 0.5)

    res_true = evaluate_second_limit_hypothesis(x, y, sigma, A, B_true)
    assert res_true.second_limit_verdict == "LOCKED_SECOND_LIMIT_CONFIRMED"
    assert res_true.delta_bic_free_minus_locked > 100.0

    res_wrong = evaluate_second_limit_hypothesis(x, y, sigma, A, B_wrong)
    assert res_wrong.second_limit_verdict == "SECOND_LIMIT_REJECTED"
    assert res_wrong.delta_bic_free_minus_locked < -100.0


def test_single_seed_crossover_audit() -> None:
    """Test single-seed crossover audits on Michaelis-Menten, Planck, and Debye."""
    # Michaelis-Menten: convex_rat is exact -> truth in equivalence class, no false-UNIQUE
    mm = audit_crossover_case_single_seed("michaelis_menten", seed=0, include_exact_truth_in_pool=False)
    assert mm["truth_in_equiv"] is True
    assert mm["is_false_unique"] is False
    assert mm["beats_v1_one_limit"] is True

    # Planck with elementary pool only: beats v1 one-limit, but elementary approximation
    # triggers False-UNIQUE because exact Bose-Einstein form is not in elementary pool
    pl_elem = audit_crossover_case_single_seed("planck", seed=0, include_exact_truth_in_pool=False)
    assert pl_elem["beats_v1_one_limit"] is True
    assert pl_elem["is_false_unique"] is True

    # Planck with exact truth included in pool: recovers exact_truth with 0% False-UNIQUE
    pl_exact = audit_crossover_case_single_seed("planck", seed=0, include_exact_truth_in_pool=True)
    assert pl_exact["truth_in_equiv"] is True
    assert pl_exact["is_true_unique"] is True
    assert pl_exact["is_false_unique"] is False
