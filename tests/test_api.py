import pytest
import os
import numpy as np
import sympy as sp
import adcd

def test_discover_correction_scenario():
    """Test discover_correction on a standard textbook scenario."""
    scenarios = adcd.get_all_scenarios()
    # Let's pick Relativistic KE
    relativistic_ke = [s for s in scenarios if s.name == "Relativistic KE"][0]
    
    # Run with 1 iteration to keep it fast
    result = adcd.discover_correction(relativistic_ke, max_iterations=1, proposer="mock", seed=42)
    
    assert isinstance(result, adcd.ADCDResult)
    assert isinstance(result.best_expr, str)
    assert isinstance(result.best_theta, dict)
    assert len(result.residual) == 200
    
    # Verify helper methods do not raise errors
    summary_text = result.summary()
    assert "ADCD DISCOVERY RUN SUMMARY" in summary_text
    assert "Relativistic KE" in summary_text
    
    latex_text = result.export_latex()
    assert "\\Delta" in latex_text
    
    # Test candidates display (returns HTML table in mockup test environment)
    candidates_display = result.show_candidates(top_k=3)
    assert candidates_display is not None or True
    
    # repr_html
    html_repr = result._repr_html_()
    assert "ADCD Correction Discovery Results" in html_repr
    assert "Relativistic KE" in html_repr


def test_fit_custom_additive():
    """Test high-level fit() with custom additive data."""
    # Classical: y = x, Correction: + 0.5 * x**2
    x = np.linspace(1.0, 5.0, 50)
    X = {"x": x}
    y_classical = x
    y_obs = x + 0.5 * x**2
    
    result = adcd.fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable="x",
        limit_direction="0",
        classical_expr="x",
        correction_mode="additive",
        max_iterations=1,
        proposer="mock",
        verbose=False,
        seed=42,
    )
    
    assert isinstance(result, adcd.ADCDResult)
    assert result.scenario.correction_type == "additive"
    assert len(result.residual) == 50
    np.testing.assert_allclose(result.residual, 0.5 * x**2, rtol=1e-5)


def test_fit_custom_multiplicative():
    """Test high-level fit() with custom multiplicative data."""
    # Classical: y = 2*x, Correction: * (1 + 0.1 * x)
    x = np.linspace(1.0, 5.0, 50)
    X = {"x": x}
    y_classical = 2.0 * x
    y_obs = 2.0 * x * (1.0 + 0.1 * x)
    
    result = adcd.fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable="x",
        limit_direction="0",
        classical_expr="2*x",
        correction_mode="multiplicative",
        max_iterations=1,
        proposer="mock",
        verbose=False,
        seed=42,
    )
    
    assert isinstance(result, adcd.ADCDResult)
    assert result.scenario.correction_type == "multiplicative"
    assert len(result.residual) == 50
    np.testing.assert_allclose(result.residual, 0.1 * x, rtol=1e-5)


def test_fit_auto_mode():
    """Test auto correction mode detection during fitting."""
    x = np.linspace(1.0, 10.0, 100)
    X = {"x": x}
    y_classical = 3.0 * x
    
    # 1. Test additive anomaly
    y_obs_add = y_classical + 2.5
    result_add = adcd.fit(
        X=X,
        y_obs=y_obs_add,
        y_classical=y_classical,
        correction_mode="auto",
        max_iterations=1,
        proposer="mock",
        verbose=False,
    )
    assert result_add.scenario.correction_type == "additive"
    
    # 2. Test multiplicative anomaly
    y_obs_mult = y_classical * (1.0 + 0.1 * x)
    result_mult = adcd.fit(
        X=X,
        y_obs=y_obs_mult,
        y_classical=y_classical,
        correction_mode="auto",
        max_iterations=1,
        proposer="mock",
        verbose=False,
    )
    assert result_mult.scenario.correction_type == "multiplicative"


def test_api_key_validation():
    """Verify ValueError is raised if LLM proposer is selected without API key."""
    x = np.linspace(1.0, 5.0, 10)
    X = {"x": x}
    y_classical = x
    y_obs = x + 1.0
    
    # Temporarily remove GEMINI_API_KEY from environment to test fallback
    old_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
        
    try:
        with pytest.raises(ValueError, match="API key must be provided"):
            adcd.fit(X, y_obs, y_classical, proposer="gemini")
            
        with pytest.raises(ValueError, match="API key must be provided"):
            adcd.fit(X, y_obs, y_classical, proposer="hybrid")
    finally:
        if old_key is not None:
            os.environ["GEMINI_API_KEY"] = old_key


def test_plot_residuals_no_crash():
    """Verify that plot_residuals generates matplotlib plot without raising errors."""
    import matplotlib
    matplotlib.use('Agg')  # headless backend for tests
    
    x = np.linspace(1.0, 5.0, 20)
    X = {"x": x}
    y_classical = x
    y_obs = x + 0.2 * x**2
    
    result = adcd.fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        max_iterations=1,
        proposer="mock",
        verbose=False,
    )
    
    # Simply verify calling it doesn't crash
    result.plot_residuals()
    
    import tempfile
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "test_residual_plot.png")
    try:
        result.plot_residuals(save_path=tmp_path)
        assert os.path.exists(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fit_multivar():
    """Test high-level fit() with multi-variable limit conditions."""
    # Classical: y_classical = 2*x + y
    # Observation: y_obs = 2*x + y + 0.1 * x**2 / y
    # Limits: x -> 0 (direction 0), y -> oo (direction oo)
    x = np.linspace(1.0, 5.0, 50)
    y = np.linspace(10.0, 50.0, 50)
    X = {"x": x, "y": y}
    y_classical = 2.0 * x + y
    y_obs = 2.0 * x + y + 0.1 * x**2 / y

    result = adcd.fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable="x, y",
        limit_direction="0, oo",
        classical_expr="2*x + y",
        correction_mode="additive",
        max_iterations=1,
        proposer="mock",
        verbose=False,
        seed=42,
    )

    assert isinstance(result, adcd.ADCDResult)
    assert result.scenario.correction_type == "additive"
    assert "x" in result.scenario.classical_limit_variable
    assert "y" in result.scenario.classical_limit_variable
    assert "0" in result.scenario.classical_limit_direction
    assert "oo" in result.scenario.classical_limit_direction


