"""
tests/test_julia_bridge.py
==========================
Tests the Python-Julia bridge interface (ADCDJuliaEngine).
"""
import numpy as np

from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig, JuliaEngineData


def test_julia_engine_coulomb_correction():
    """Test that Python can call Julia ADCDEngine to discover Coulomb SR correction."""
    n = 80
    r_vals = np.linspace(0.1, 10.0, n)
    v_vals = np.linspace(1e6, 0.9e8, n)
    c_val = 3e8
    k_e = 8.99e9
    q1 = 1.6e-19
    q2 = 1.6e-19

    y_cl = k_e * q1 * q2 / (r_vals ** 2)
    true_theta = 0.15
    beta = (v_vals / c_val) ** 2
    y_obs = y_cl * (1.0 + true_theta * beta)

    engine = ADCDJuliaEngine()

    config = JuliaEngineConfig(
        domain="lorentz_special_relativity",
        target_dim="dimensionless",
        input_vars=["v", "c"],
        known_constants={"c": c_val, "k_e": k_e},
        bic_threshold=6.0,
        nmse_coarse=1.5,
        nmse_fine=0.05,
        n_restarts=10,
    )

    data = JuliaEngineData(
        y_classical=y_cl,
        y_obs=y_obs,
        vars={
            "r": r_vals,
            "v": v_vals,
            "c": np.full(n, c_val),
            "q": np.full(n, q1),
            "Q": np.full(n, q2),
        },
    )

    result = engine.run(config, data)

    assert result.n_proposals_generated >= 3
    assert result.gate_stats["n_pass_gate_a"] >= 1
    assert result.gate_stats["n_identifiable"] >= 1
    assert len(result.identifiable) >= 1

    best = result.best
    assert best is not None
    assert best.is_identifiable
    assert best.delta_bic >= 6.0
    assert best.nmse < 0.05


def test_julia_engine_dimensional_gate():
    """Test calling Julia's hard dimensional gate directly from Python."""
    engine = ADCDJuliaEngine()

    # Dimensionless ratio: v/c
    valid_ratio = {
        "op": "div",
        "args": [{"sym": "v"}, {"sym": "c"}]
    }
    assert engine.verify_dimension(valid_ratio, "dimensionless") is True

    # Bare velocity symbol as dimensionless -> must be rejected
    invalid = {"sym": "v"}
    assert engine.verify_dimension(invalid, "dimensionless") is False


if __name__ == "__main__":
    print("Running Python-Julia bridge tests...")
    test_julia_engine_coulomb_correction()
    print("test_julia_engine_coulomb_correction: PASSED")
    test_julia_engine_dimensional_gate()
    print("test_julia_engine_dimensional_gate: PASSED")
    print("All python julia bridge tests passed!")
