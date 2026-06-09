"""Tests for scale-adaptive NMSE on extreme dynamic-range residuals."""

import numpy as np
import pytest

from adcd.jax_optimizer import JAXOptimizer
from adcd.metrics import _nmse, evaluate_correction, _evaluate_delta_array
from adcd.real_scenarios import get_real_scenarios


def test_nmse_not_saturated_for_pulsar_scale():
    """Degenerate zero model must not appear as near-perfect fit."""
    from adcd.real_data_loader import load_binary_pulsar_decay

    _, _, _, residual = load_binary_pulsar_decay()
    mse_zero = float(np.mean(residual ** 2))
    nmse_zero = _nmse(mse_zero, residual)
    assert nmse_zero > 0.5, "Zero predictor should have NMSE near 1, not ~0"


def test_jax_optimizer_rejects_zero_degenerate_pulsar():
    from adcd.real_data_loader import load_binary_pulsar_decay

    X, _, _, residual = load_binary_pulsar_decay()
    opt = JAXOptimizer()
    bad = opt.optimize("-theta_0**4/P**4", X, residual, ["P"], seed=42)
    good = opt.optimize("theta_0 * P**(-5.0/3.0)", X, residual, ["P"], seed=42)
    assert good.nmse < bad.nmse


def test_evaluate_correction_pulsar_power_law():
    scenario = [s for s in get_real_scenarios() if "Pulsar" in s.name][0]
    X, y, y0, _ = scenario.generate_data()
    A = scenario.correction_constants["theta_0"]
    ev = evaluate_correction(
        "theta_0 * P**(-5.0/3.0)",
        scenario,
        X,
        y,
        y0,
        {"theta_0": A},
    )
    assert ev.nmse_residual < 1e-2
    assert ev.class_match


def test_evaluate_delta_array_matches_manual():
    from adcd.real_data_loader import load_binary_pulsar_decay

    X, y, _, _ = load_binary_pulsar_decay()
    theta = {"theta_0": 1.35e-7}
    delta = _evaluate_delta_array(
        "theta_0 * P**(-5.0/3.0)", X, theta, {"G": 6.674e-11, "c": 2.998e8}, len(y)
    )
    assert np.all(delta > 0)
    assert np.all(np.isfinite(delta))
