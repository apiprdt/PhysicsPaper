"""
Tests for Phase 2.2: Scale-invariant candidate screening and closed-form WLS amplitude solution.
"""

import time
import numpy as np

from adcd.scale_invariant_screening import ScaleInvariantScreener


def test_scale_invariance_under_1e6_multiplication():
    """
    Test: multiply an input column by 1e6; the selected structure must be identical
    and the dimensionless parameters equal within a tight empirical tolerance.
    """
    rng = np.random.RandomState(42)
    N = 200
    r_base = np.linspace(0.2, 4.0, N)
    
    # Ground truth: y = 2.0 * (exp(-r / 1.5) - 1.0)
    y_true = 2.0 * (np.exp(-r_base / 1.5) - 1.0)
    noise = rng.normal(0, 0.01 * np.abs(y_true))
    y_obs = y_true + noise

    candidates = [
        "theta_0 * (exp(-r/theta_1) - 1.0)",
        "theta_0 * (r/theta_1 / (1.0 - r/theta_1))",
        "theta_0 * log(1.0 + r/theta_1)",
        "theta_0 * tanh(r/theta_1)",
    ]

    screener_base = ScaleInvariantScreener(
        X={"r": r_base},
        y_obs=y_obs,
    )
    ranked_base = screener_base.rank_candidates(candidates)

    # Now multiply input column r by 1e6 (e.g. converting from m to micrometers)
    r_scaled = r_base * 1e6
    screener_scaled = ScaleInvariantScreener(
        X={"r": r_scaled},
        y_obs=y_obs,
    )
    ranked_scaled = screener_scaled.rank_candidates(candidates)

    # 1. Selected structure MUST be identical (same top candidate)
    assert ranked_base[0]["expr"] == ranked_scaled[0]["expr"]
    assert ranked_base[0]["expr"] == "theta_0 * (exp(-r/theta_1) - 1.0)"

    # 2. Entire ranking order must be identical
    assert [c["expr"] for c in ranked_base] == [c["expr"] for c in ranked_scaled]

    # 3. NMSE and RSS must be identical
    assert np.isclose(ranked_base[0]["nmse"], ranked_scaled[0]["nmse"], rtol=1e-5)

    # 4. Amplitude theta_0 must be equal within tolerance
    assert np.isclose(ranked_base[0]["theta_0"], ranked_scaled[0]["theta_0"], rtol=1e-3)

    # 5. Inner scale theta_1 must scale exactly by 1e6
    ratio_theta1 = ranked_scaled[0]["theta_1"] / ranked_base[0]["theta_1"]
    assert np.isclose(ratio_theta1, 1e6, rtol=1e-3)


def test_scale_invariance_small_scale_1e_minus_6():
    """Verify invariance when input column is downscaled by 1e-6."""
    rng = np.random.RandomState(99)
    N = 100
    x_base = np.linspace(0.1, 2.0, N)
    y_obs = 1.5 * np.log1p(x_base / 0.8) + rng.normal(0, 0.01, size=N)

    candidates = [
        "theta_0 * log(1.0 + x/theta_1)",
        "theta_0 * (exp(-x/theta_1) - 1.0)",
    ]

    screener1 = ScaleInvariantScreener(X={"x": x_base}, y_obs=y_obs)
    res1 = screener1.rank_candidates(candidates)

    screener2 = ScaleInvariantScreener(X={"x": x_base * 1e-6}, y_obs=y_obs)
    res2 = screener2.rank_candidates(candidates)

    assert res1[0]["expr"] == res2[0]["expr"]
    assert res1[0]["expr"] == "theta_0 * log(1.0 + x/theta_1)"
    assert np.isclose(res1[0]["nmse"], res2[0]["nmse"], rtol=1e-5)


def test_screening_benchmark_timing():
    """Measure and report screening timing without claiming unverified speedups."""
    N = 500
    rng = np.random.RandomState(42)
    X = {"x": np.linspace(0.1, 5.0, N)}
    y_obs = np.tanh(X["x"] / 2.0) + rng.normal(0, 0.01, size=N)

    candidates = [
        "theta_0 * tanh(x/theta_1)",
        "theta_0 * (exp(-x/theta_1) - 1.0)",
        "theta_0 * log(1.0 + x/theta_1)",
        "theta_0 * (x/theta_1 / (1.0 + x/theta_1))",
    ] * 25  # 100 candidates total

    screener = ScaleInvariantScreener(X=X, y_obs=y_obs)
    
    t0 = time.perf_counter()
    ranked = screener.rank_candidates(candidates)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(ranked) == 100
    # Log measured timing
    print(f"\n[BENCHMARK] Screened 100 candidates on N={N} in {elapsed_ms:.2f} ms ({elapsed_ms/100:.3f} ms/candidate)")
    assert elapsed_ms > 0.0
