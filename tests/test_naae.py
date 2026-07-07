"""
test_naae.py
============
Comprehensive tests for the Noise-Aware Asymptotic Estimator (NAAE).

Tests cover:
- Clean data exponent recovery (tight tolerance)
- Noisy data comparison vs RAS (statistically meaningful threshold)
- limit_direction="oo" (infinity limit, previously untested)
- Negative exponents (inverse power law like Yukawa/Coulomb)
- Multiplicative correction type
- Extreme-scale exponent (T^4 Stefan-Boltzmann, gamma=4)
- Low data count graceful failure
"""

import numpy as np
import pytest
from adcd.naae import estimate_exponent_naae
from adcd.residual_analyzer import compute_ras


class TestNAAECleanData:
    """NAAE should recover parameters exactly on noise-free data."""

    def test_additive_positive_exponent(self):
        """y_obs = x + 0.5*x^2, gamma=2.0, A=0.5 — baseline clean test."""
        x = np.linspace(0.1, 2.0, 120)
        y_cls = x
        y_obs = x + 0.5 * x**2
        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        assert res["leading_exponent"] is not None
        assert abs(res["leading_exponent"] - 2.0) < 0.05
        assert abs(res["prefactor"] - 0.5) < 0.05
        assert res["fit_quality"] > 0.99
        assert res["degenerate"] is False

    def test_additive_negative_exponent(self):
        """y_obs = x + 0.3*x^(-2), gamma=-2.0 — Yukawa-type inverse power law."""
        x = np.linspace(0.5, 3.0, 120)
        y_cls = x
        y_obs = x + 0.3 * x**(-2)
        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        assert res["leading_exponent"] is not None
        # Inverse exponent; allow +/- 0.15 tolerance
        assert abs(res["leading_exponent"] - (-2.0)) < 0.15, (
            f"Expected ~-2.0, got {res['leading_exponent']:.4f}"
        )

    def test_stefan_boltzmann_exponent(self):
        """y_obs = x + 0.01*x^4, gamma=4.0 — Stefan-Boltzmann scaling.
        Previously impossible with gamma bounds of [-6,6] but now [-8,8]."""
        x = np.linspace(0.1, 1.5, 120)
        y_cls = x
        y_obs = x + 0.01 * x**4
        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        assert res["leading_exponent"] is not None
        assert abs(res["leading_exponent"] - 4.0) < 0.20, (
            f"Stefan-Boltzmann T^4 scaling: expected ~4.0, got {res['leading_exponent']:.4f}"
        )

    def test_multiplicative_correction(self):
        """y_obs = x * (1 + 0.1*x), multiplicative mode."""
        x = np.linspace(0.1, 2.0, 100)
        y_cls = x
        y_obs = x * (1 + 0.1 * x)
        res = estimate_exponent_naae(x, y_obs, y_cls, "multiplicative", "0")
        assert res["leading_exponent"] is not None
        assert abs(res["leading_exponent"] - 1.0) < 0.10, (
            f"Multiplicative exponent: expected ~1.0, got {res['leading_exponent']:.4f}"
        )

    def test_infinity_limit_direction(self):
        """Detect scaling at large x: y_obs = x^2 + 0.05*x^3, limit_direction='oo'."""
        x = np.linspace(1.0, 10.0, 120)
        y_cls = x**2
        y_obs = x**2 + 0.05 * x**3
        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "oo")
        assert res["leading_exponent"] is not None
        assert abs(res["leading_exponent"] - 3.0) < 0.20, (
            f"Infinity-limit exponent: expected ~3.0, got {res['leading_exponent']:.4f}"
        )


class TestNAAEVsRAS:
    """NAAE should outperform RAS under noisy conditions.
    
    The claim is precise: NAAE error <= RAS error in absolute exponent recovery.
    Tolerance is tightened from 0.6 to 0.25 (8.3% of the true exponent value 3.0),
    which is meaningful and consistent with the SNR budget of the test scenario.
    """

    def test_noisy_small_correction_better_than_ras(self):
        """delta = 0.05*x^3 with SNR~5: NAAE closer to gamma=3.0 than log-space RAS."""
        rng = np.random.RandomState(42)
        x = np.linspace(0.1, 1.0, 120)
        y_cls = x
        delta_true = 0.05 * x**3
        noise = 0.002 * rng.randn(120)  # SNR = std(delta)/std(noise) ~ 0.05*0.3/0.002 ~ 7.5
        y_obs = y_cls + delta_true + noise

        res_naae = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        res_ras = compute_ras(x_vals=x, delta_vals=delta_true + noise, limit_val=0.0)

        naae_exp = res_naae["leading_exponent"]
        ras_exp  = res_ras["leading_exponent"]

        assert naae_exp is not None, "NAAE returned None exponent"

        err_naae = abs(naae_exp - 3.0)
        # NAAE must recover within 0.25 of the true exponent (< 8.3% relative error)
        assert err_naae < 0.25, (
            f"NAAE error {err_naae:.4f} exceeds 0.25 threshold. NAAE={naae_exp:.4f}"
        )
        # And NAAE must outperform RAS when RAS has an exponent estimate
        if ras_exp is not None:
            err_ras = abs(ras_exp - 3.0)
            assert err_naae <= err_ras, (
                f"NAAE ({err_naae:.4f}) should be better than RAS ({err_ras:.4f}) under noise"
            )

    def test_very_noisy_graceful_degradation(self):
        """At extremely low SNR (~1), NAAE should at least not crash and return finite values."""
        rng = np.random.RandomState(99)
        x = np.linspace(0.1, 1.0, 50)
        y_cls = x
        delta_true = 0.001 * x**2  # tiny signal
        noise = 0.05 * rng.randn(50)  # dominant noise
        y_obs = y_cls + delta_true + noise

        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        # Should return a dict with finite or None values — never crash
        assert isinstance(res, dict)
        assert "leading_exponent" in res
        if res["leading_exponent"] is not None:
            assert np.isfinite(res["leading_exponent"])


class TestNAAEEdgeCases:
    """Edge cases for robustness."""

    def test_too_few_points(self):
        """Less than 4 points should return None gracefully."""
        x = np.array([0.1, 0.2, 0.3])
        y_cls = x
        y_obs = x + 0.1 * x**2
        res = estimate_exponent_naae(x, y_obs, y_cls)
        assert res["leading_exponent"] is None
        assert res["prefactor"] is None
        assert res["fit_quality"] == 0.0

    def test_nonuniform_x_distribution(self):
        """Percentile-based data selection must work correctly for log-spaced x."""
        x = np.logspace(-2, 1, 100)  # 0.01 to 10, highly non-uniform
        y_cls = x
        y_obs = x + 0.5 * x**2
        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        assert res["leading_exponent"] is not None
        assert abs(res["leading_exponent"] - 2.0) < 0.15, (
            f"Log-spaced x: expected ~2.0, got {res['leading_exponent']:.4f}"
        )

    def test_degeneracy_detection(self):
        """When anomaly exponent matches baseline, degenerate=True should be flagged."""
        x = np.linspace(0.1, 2.0, 100)
        y_cls = x          # classical: linear in x (exponent 1)
        y_obs = 1.5 * x    # anomaly is also linear (exponent 1) — degenerate
        res = estimate_exponent_naae(x, y_obs, y_cls, "additive", "0")
        # gamma_est ~ 1, slope of classical ~ 1, so |gamma_est - slope| < 0.25
        assert res["degenerate"] is True
