"""
Tests for compute_ras (Residual Asymptotic Signature) exponent recovery.

Implements Fase 0 of ADCD_Upgrade_Plan_v2.md §4.1:
  Verify that the nonlinear-fit RAS estimator recovers gamma accurately
  under realistic narrow-window + additive noise-floor conditions.

The old log-log regression had systematic bias:
  gamma=2 -> estimated ~1.7 (15% error)
  gamma=4 -> estimated ~2.0 (49% error)

After the nonlinear fix all errors should be <5%.
"""
import numpy as np
import pytest
from adcd.residual_analyzer import compute_ras

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_power_law_signal(
    gamma: float,
    n_points: int = 200,
    x_max: float = 0.15,   # narrow window near limit (realistic: near v->0)
    noise_floor: float = 0.0,
    noise_pct: float = 0.0,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic residual delta = A * x^gamma + noise.

    Two noise modes (can combine):
    - noise_floor: fixed additive floor (mimics absolute measurement floor)
    - noise_pct: proportional noise = noise_pct * |signal| (mimics ADCD benchmarks)

    For gamma >= 3 with narrow x windows, the signal at x -> 0 reaches 1e-12
    or below, making a fixed noise_floor the dominant component everywhere near
    the limit (SNR << 1). Use noise_pct for those cases — it matches how
    the 9 benchmark scenarios add noise (proportional to observed value).
    """
    rng = np.random.default_rng(seed)
    # x uniformly in (0, x_max] — narrow window near classical limit at 0
    x = rng.uniform(1e-4, x_max, n_points)
    A = 1.0
    signal = A * x ** gamma
    noise = (noise_floor * rng.standard_normal(n_points)
             + noise_pct * signal * rng.standard_normal(n_points))
    delta = signal + noise
    return x, delta


# ---------------------------------------------------------------------------
# Table of (gamma, tolerance) test cases
# Covers all structural families present in the 9 benchmark scenarios:
#   gamma=2   -> relativistic KE, nonlinear drag (leading order)
#   gamma=4   -> anharmonic spring (leading correction term)
#   gamma=1   -> linear screened-Coulomb (first-order expansion)
#   gamma=1.5 -> fractional power-law (blind benchmark)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma,tol_pct", [
    (1.0, 5.0),   # linear
    (2.0, 5.0),   # quadratic (relativistic KE)
    (3.0, 5.0),   # cubic
    (4.0, 5.0),   # quartic (anharmonic spring — was 49% error before fix)
    (1.5, 5.0),   # fractional power-law
])
def test_ras_exponent_noiseless(gamma, tol_pct):
    """Noiseless case: nonlinear fit should recover gamma within tol_pct%."""
    x, delta = _make_power_law_signal(gamma=gamma, noise_floor=0.0)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)

    assert result["leading_exponent"] is not None, (
        f"gamma={gamma}: fit returned None (fit failed to converge)"
    )
    recovered = result["leading_exponent"]
    error_pct = abs(recovered - gamma) / gamma * 100
    assert error_pct < tol_pct, (
        f"gamma={gamma}: recovered {recovered:.4f}, "
        f"error={error_pct:.1f}% > {tol_pct}%"
    )


@pytest.mark.parametrize("gamma,tol_pct", [
    # Additive floor case: only valid where signal >> floor inside the window.
    # For gamma=2, signal at x=0.15 is 0.0225 >> noise_floor=1e-4 (SNR~225).
    # For gamma=4, signal at x=0.15 is ~5e-4 and drops to 1e-12 at x=0.001,
    # making additive floor the dominant component near the limit (SNR~2e-8).
    # That case is tested via proportional noise below.
    (2.0, 5.0),
])
def test_ras_exponent_with_additive_noise_floor(gamma, tol_pct):
    """
    Additive noise floor (1e-4) on narrow window for gamma=2.
    Verifies nonlinear fit handles floor noise better than log-log.
    """
    x, delta = _make_power_law_signal(gamma=gamma, noise_floor=1e-4)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)

    assert result["leading_exponent"] is not None, (
        f"gamma={gamma} additive-noise: fit returned None"
    )
    recovered = result["leading_exponent"]
    error_pct = abs(recovered - gamma) / gamma * 100
    assert error_pct < tol_pct, (
        f"gamma={gamma} (additive noise): recovered {recovered:.4f}, "
        f"error={error_pct:.1f}% > {tol_pct}%"
    )


@pytest.mark.parametrize("gamma,tol_pct", [
    (2.0, 5.0),
    (4.0, 5.0),   # critical case: old log-log gave ~49% error; proportional
                  # noise matches actual ADCD benchmark scenario conditions
])
def test_ras_exponent_with_proportional_noise(gamma, tol_pct):
    """
    Proportional noise (10% of signal) — matches how the 9 ADCD benchmark
    scenarios add noise (sigma = noise_pct * |y_obs|).

    This is the physically realistic noise model for the benchmark scenarios.
    For gamma=4 (Anharmonic Spring), proportional noise preserves SNR across
    the entire x-window, allowing the fit to see the true exponent.
    """
    x, delta = _make_power_law_signal(gamma=gamma, noise_pct=0.10)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)

    assert result["leading_exponent"] is not None, (
        f"gamma={gamma} proportional-noise: fit returned None"
    )
    recovered = result["leading_exponent"]
    error_pct = abs(recovered - gamma) / gamma * 100
    assert error_pct < tol_pct, (
        f"gamma={gamma} (proportional noise): recovered {recovered:.4f}, "
        f"error={error_pct:.1f}% > {tol_pct}%"
    )


def test_ras_gamma_std_returned():
    """gamma_std should be a finite positive float (from covariance matrix)."""
    x, delta = _make_power_law_signal(gamma=2.0, noise_floor=1e-4)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)

    assert result["gamma_std"] is not None, "gamma_std should not be None on clean fit"
    assert result["gamma_std"] >= 0.0, "gamma_std must be non-negative"
    assert np.isfinite(result["gamma_std"]), "gamma_std must be finite"


def test_ras_gamma_std_larger_under_noise():
    """
    Uncertainty estimate (gamma_std) should be larger when noise floor is
    bigger — the covariance matrix should reflect increased uncertainty.
    """
    x_clean, d_clean = _make_power_law_signal(gamma=2.0, noise_floor=0.0)
    x_noisy, d_noisy = _make_power_law_signal(gamma=2.0, noise_floor=1e-2)

    r_clean = compute_ras(x_vals=x_clean, delta_vals=d_clean, limit_val=0.0)
    r_noisy = compute_ras(x_vals=x_noisy, delta_vals=d_noisy, limit_val=0.0)

    std_clean = r_clean.get("gamma_std")
    std_noisy = r_noisy.get("gamma_std")

    if std_clean is not None and std_noisy is not None:
        assert std_noisy >= std_clean, (
            f"Noisy gamma_std ({std_noisy:.4f}) should be >= clean ({std_clean:.4f})"
        )


def test_ras_insufficient_points_returns_unknown():
    """With fewer than 5 valid points, compute_ras should return 'unknown'."""
    x = np.array([0.01, 0.02, 0.03])
    delta = np.array([1e-4, 4e-4, 9e-4])
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)
    assert result["leading_exponent"] is None
    assert result["suggested_class"] == "unknown"


def test_ras_classification_polynomial():
    """Near-integer gamma -> suggested_class == 'polynomial'."""
    x, delta = _make_power_law_signal(gamma=2.0, noise_floor=0.0)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)
    assert result["suggested_class"] == "polynomial", (
        f"Expected 'polynomial', got '{result['suggested_class']}'"
    )


def test_ras_classification_power_law():
    """Fractional gamma -> suggested_class == 'power_law'."""
    x, delta = _make_power_law_signal(gamma=1.5, noise_floor=0.0)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)
    assert result["suggested_class"] == "power_law", (
        f"Expected 'power_law', got '{result['suggested_class']}'"
    )


def test_ras_fit_quality_high_for_clean_signal():
    """R^2 of nonlinear fit should be >0.95 for a clean power-law signal."""
    x, delta = _make_power_law_signal(gamma=2.0, noise_floor=0.0)
    result = compute_ras(x_vals=x, delta_vals=delta, limit_val=0.0)
    assert result["fit_quality"] > 0.95, (
        f"Expected R^2 > 0.95, got {result['fit_quality']:.4f}"
    )


def test_analyze_residual_exposes_ras_gamma_std():
    """analyze_residual should pass ras_gamma_std through to ResidualFeatures."""
    from adcd.residual_analyzer import analyze_residual
    x, delta = _make_power_law_signal(gamma=2.0, noise_floor=1e-4)
    features = analyze_residual(x, delta, classical_limit_val=0.0)

    assert hasattr(features, "ras_gamma_std"), (
        "ResidualFeatures must have ras_gamma_std field"
    )
    assert features.ras_gamma_std is not None
    assert features.ras_gamma_std >= 0.0
