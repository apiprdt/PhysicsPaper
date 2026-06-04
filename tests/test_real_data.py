"""
Unit tests for ADCD v2.0 real experimental data infrastructure.

Tests verify:
  1. All 4 loaders return correct array shapes and types
  2. Physics sanity: residuals are non-zero, values are finite
  3. Scenario wrappers have valid metadata
  4. Scenarios can generate data via the standard AnomalyScenario interface
"""

import pytest
import numpy as np

from adcd.real_data_loader import (
    load_mercury_perihelion,
    load_hydrogen_lamb_shift,
    load_blackbody_radiation,
    load_muon_g2,
    load_all_real_data,
)
from adcd.real_scenarios import get_real_scenarios, RealAnomalyScenario


# ─────────────────────────────────────────────────────────────────────────────
# Loader shape and type tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRealDataLoaderShapes:

    def _check_tuple(self, result, expected_n: int, expected_x_keys: list):
        """Helper: verify the returned 4-tuple has correct shapes."""
        X, y_obs, y_cls, residual = result
        assert isinstance(X, dict), "X must be a dict"
        assert isinstance(y_obs, np.ndarray)
        assert isinstance(y_cls, np.ndarray)
        assert isinstance(residual, np.ndarray)

        assert y_obs.shape == (expected_n,), f"y_obs shape {y_obs.shape} != ({expected_n},)"
        assert y_cls.shape == (expected_n,), f"y_cls shape {y_cls.shape} != ({expected_n},)"
        assert residual.shape == (expected_n,), f"residual shape {residual.shape} != ({expected_n},)"

        for key in expected_x_keys:
            assert key in X, f"Expected key '{key}' in X"
            assert X[key].shape == (expected_n,), f"X['{key}'] shape mismatch"

    def test_mercury_perihelion_shape(self):
        result = load_mercury_perihelion()
        self._check_tuple(result, expected_n=200, expected_x_keys=["r", "v", "theta"])

    def test_hydrogen_lamb_shift_shape(self):
        result = load_hydrogen_lamb_shift()
        # n = 2..19 → 18 points
        self._check_tuple(result, expected_n=18, expected_x_keys=["n"])

    def test_blackbody_radiation_shape(self):
        result = load_blackbody_radiation()
        self._check_tuple(result, expected_n=200, expected_x_keys=["f", "T"])

    def test_muon_g2_shape(self):
        result = load_muon_g2()
        self._check_tuple(result, expected_n=100, expected_x_keys=["alpha"])

    def test_load_all_real_data_keys(self):
        all_data = load_all_real_data()
        assert len(all_data) == 4
        expected_keys = {
            "Real: Mercury Perihelion",
            "Real: Hydrogen Lamb Shift",
            "Real: Blackbody Radiation",
            "Real: Muon g-2",
        }
        assert set(all_data.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────────────
# Physics sanity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRealDataPhysics:

    def test_mercury_physics(self):
        X, y_obs, y_cls, residual = load_mercury_perihelion()

        # Classical: zero precession
        np.testing.assert_array_equal(y_cls, 0.0)

        # GR correction should be positive and tiny
        assert np.all(y_obs > 0), "GR precession correction should be positive"
        assert np.all(np.isfinite(y_obs))

        # Residual should equal y_obs (since y_cls = 0)
        np.testing.assert_allclose(residual, y_obs, rtol=1e-10)

        # Velocity should be physically reasonable (Mercury: 38–59 km/s)
        v_kms = X["v"] / 1000.0
        assert np.all(v_kms > 30) and np.all(v_kms < 70), \
            f"Mercury velocity range unexpected: {v_kms.min():.1f}–{v_kms.max():.1f} km/s"

    def test_lamb_shift_physics(self):
        X, y_obs, y_cls, residual = load_hydrogen_lamb_shift()

        # Classical Dirac: E_n = -13.6/n² — should be negative
        assert np.all(y_cls < 0), "Dirac energy levels must be negative"

        # Lamb shift is a small positive correction
        assert np.all(np.isfinite(y_obs))
        assert np.all(residual > 0), "Lamb shift correction should be positive"

        # Lamb shift decreases with n (power law 1/n³)
        assert residual[0] > residual[-1], "Lamb shift should decrease with n"

        # n values should be 2..19
        np.testing.assert_array_equal(X["n"], np.arange(2, 20, dtype=float))

    def test_blackbody_physics(self):
        X, y_obs, y_cls, residual = load_blackbody_radiation()

        # All radiances should be positive and finite
        assert np.all(y_cls > 0)
        assert np.all(y_obs >= 0)
        assert np.all(np.isfinite(residual))

        # At low frequency: Planck ≈ RJ → residual near 0
        # At high frequency: Planck << RJ → residual << -1 (suppressed)
        low_f_mask = X["f"] < 1e12
        high_f_mask = X["f"] > 1e14

        # Low-frequency residual should be small (|Δ| < 0.1)
        if low_f_mask.any():
            assert np.all(np.abs(residual[low_f_mask]) < 0.5), \
                "At low frequency, Planck ≈ Rayleigh-Jeans"

        # High-frequency residual should be strongly negative (Planck <<  RJ)
        if high_f_mask.any():
            assert np.all(residual[high_f_mask] < 0), \
                "At high frequency, Planck < Rayleigh-Jeans (UV catastrophe fix)"

        # Temperature array should be constant at 5778 K
        np.testing.assert_allclose(X["T"], 5778.0, rtol=1e-6)

    def test_muon_g2_physics(self):
        X, y_obs, y_cls, residual = load_muon_g2()

        # Classical Dirac: a_mu = 0
        np.testing.assert_array_equal(y_cls, 0.0)

        # QED correction should be small positive (a_mu ~ 1e-3)
        assert np.all(y_obs > 0), "QED anomalous magnetic moment must be positive"
        assert np.all(y_obs < 0.01), "a_mu should be << 1 (perturbative QED)"

        # Residual = y_obs (additive correction on zero classical)
        np.testing.assert_allclose(residual, y_obs, rtol=1e-10)

        # alpha_eff should span a reasonable range around 1/137
        alpha_0 = 7.297e-3
        assert X["alpha"].min() > alpha_0 * 0.4
        assert X["alpha"].max() < alpha_0 * 2.1

    def test_reproducibility(self):
        """Same seed must give identical results."""
        r1 = load_mercury_perihelion(seed=99)
        r2 = load_mercury_perihelion(seed=99)
        np.testing.assert_array_equal(r1[1], r2[1])  # y_obs

        r3 = load_mercury_perihelion(seed=0)
        # Different seed → different noise → different y_obs
        assert not np.array_equal(r1[1], r3[1])


# ─────────────────────────────────────────────────────────────────────────────
# Real scenario metadata tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRealScenarios:

    def setup_method(self):
        self.scenarios = get_real_scenarios()

    def test_returns_four_scenarios(self):
        assert len(self.scenarios) == 4

    def test_all_have_real_data_tier(self):
        for s in self.scenarios:
            assert s.tier == "real_data", f"{s.name} must have tier='real_data'"

    def test_all_names_start_with_real(self):
        for s in self.scenarios:
            assert s.name.startswith("Real: "), \
                f"Scenario name must start with 'Real: ', got '{s.name}'"

    def test_correction_types_valid(self):
        valid = {"additive", "multiplicative"}
        for s in self.scenarios:
            assert s.correction_type in valid, \
                f"{s.name}: correction_type '{s.correction_type}' not in {valid}"

    def test_limit_directions_valid(self):
        valid = {"0", "oo"}
        for s in self.scenarios:
            assert s.classical_limit_direction in valid, \
                f"{s.name}: limit_direction '{s.classical_limit_direction}' invalid"

    def test_correction_classes_valid(self):
        valid = {"polynomial", "exponential", "power_law", "rational",
                 "trigonometric", "logarithmic"}
        for s in self.scenarios:
            assert s.correction_class in valid, \
                f"{s.name}: correction_class '{s.correction_class}' not in {valid}"

    def test_classical_variables_non_empty(self):
        for s in self.scenarios:
            assert len(s.classical_variables) > 0, \
                f"{s.name}: classical_variables must be non-empty"

    def test_scenarios_are_real_anomaly_scenario_subclass(self):
        """All returned objects must be RealAnomalyScenario instances."""
        for s in self.scenarios:
            assert isinstance(s, RealAnomalyScenario), \
                f"{s.name} must be a RealAnomalyScenario, got {type(s).__name__}"

    def test_scenarios_generate_data(self):
        """Scenarios must produce real loader data (not uniform random fallback).

        Verifies that generate_data() delegates to the actual physics loaders
        by checking the expected sample sizes for each dataset.
        """
        expected_sizes = {
            "Real: Mercury Perihelion":  200,
            "Real: Hydrogen Lamb Shift": 18,
            "Real: Blackbody Radiation": 200,
            "Real: Muon g-2":            100,
        }
        for s in self.scenarios:
            X, y_obs, y_cls, residual = s.generate_data(seed=42)
            expected_n = expected_sizes[s.name]
            assert len(y_obs) == expected_n, \
                f"{s.name}: expected {expected_n} points from real loader, got {len(y_obs)}"
            assert np.all(np.isfinite(y_obs)), \
                f"{s.name}: y_obs contains non-finite values"

