import numpy as np
import pytest
from adcd.naae import estimate_exponent_naae
from adcd.residual_analyzer import compute_ras

def test_naae_exponent_estimation_clean():
    """Verify NAAE estimates exponents exactly on clean data."""
    # Model: y_obs = y_classical + A * x**gamma
    # y_classical = x
    # A = 0.5, gamma = 2.0
    x = np.linspace(0.1, 2.0, 100)
    y_cls = x
    y_obs = x + 0.5 * (x ** 2)
    
    res = estimate_exponent_naae(
        x_vals=x,
        y_obs=y_obs,
        y_classical=y_cls,
        correction_type="additive",
        limit_direction="0",
    )
    
    assert res["leading_exponent"] is not None
    assert abs(res["leading_exponent"] - 2.0) < 0.05
    assert abs(res["prefactor"] - 0.5) < 0.05
    assert res["fit_quality"] > 0.99
    assert res["degenerate"] is False

def test_naae_vs_ras_noisy():
    """Verify NAAE is more robust to noise bias than standard RAS on small residuals.
    
    For a small residual term delta = 0.05 * x**3 with observational noise,
    log-space RAS tends to underestimate or bias the exponent heavily,
    while linear joint NAAE should yield a closer estimate.
    """
    rng = np.random.RandomState(42)
    x = np.linspace(0.1, 1.0, 100)
    y_cls = x
    # A small L1 correction term with a realistic noise floor
    delta = 0.05 * (x ** 3)
    noise = 0.001 * rng.randn(100)
    y_obs = y_cls + delta + noise
    
    # NAAE fit
    res_naae = estimate_exponent_naae(
        x_vals=x,
        y_obs=y_obs,
        y_classical=y_cls,
        correction_type="additive",
        limit_direction="0",
    )
    
    # Standard RAS fit
    res_ras = compute_ras(
        x_vals=x,
        delta_vals=delta + noise,
        limit_val=0.0,
    )
    
    naae_exponent = res_naae["leading_exponent"]
    ras_exponent = res_ras["leading_exponent"]
    
    print(f"NAAE estimated: {naae_exponent}, RAS estimated: {ras_exponent}")
    
    # Expect NAAE to be closer to 3.0 than RAS under noisy conditions
    if naae_exponent is not None and ras_exponent is not None:
        err_naae = abs(naae_exponent - 3.0)
        err_ras = abs(ras_exponent - 3.0)
        # NAAE should out-perform RAS and be close to the true exponent 3.0
        assert err_naae < 0.6
        assert err_naae < err_ras
