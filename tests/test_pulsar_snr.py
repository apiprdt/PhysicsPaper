"""
test_pulsar_snr.py
==================
Formal pytest integration of the Task 3.0 pulsar timing SNR validation.

Validates that under realistic modern timing precision (sigma=0.2 microseconds),
ADCD can recover the sub-leading L2 timing anomaly from PSR B1913+16-type data
with: SNR >= 5.0, exponent recovery error < 0.15, and NMSE < 0.05.

Physical model:
- L1 (2.5PN): cumulative timing shift ~ 0.5 * (-2.4e-12) * t^2 / P_0  (~17 seconds)
- L2 (sub-leading): cumulative timing shift ~ 0.5 * (-2.4e-18) * t^2 / P_0  (~17 microseconds)
- Timing noise: sigma = 0.2 microseconds (state-of-the-art pulsar timing array precision)

Reference: Peters (1964) Phys. Rev. 136, B1224 (2.5PN gravitational radiation decay)
"""

import numpy as np
import pytest
from scipy.optimize import minimize

# PSR B1913+16 physical constants
P_0 = 27906.98          # Orbital period (seconds)
T_TOTAL = 20 * 365.25 * 24 * 3600  # 20 years in seconds
DP_DT_L1 = -2.4e-12    # 2.5PN period decay rate (s/s)
DP_DT_L2 = -2.4e-18    # Sub-leading period decay rate (s/s)
NOISE_STD = 0.2e-6      # Modern timing precision: 0.2 microseconds


def _simulate_timing_residuals(n_points=120, noise_std=NOISE_STD, seed=42):
    """
    Returns (t, dt_obs, dt_true) where:
    - t: observation epochs over last 40% of 20 years (high accumulation region)
    - dt_obs: observed timing residual (L1 + L2 + noise)
    - dt_true: noiseless timing residual (L1 + L2)
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0.8 * T_TOTAL, T_TOTAL, n_points)
    # Cumulative timing shift: integral of (dP/dt * dt) / P_0 approximated as:
    dt_L1 = 0.5 * DP_DT_L1 * (t**2) / P_0
    dt_L2 = 0.5 * DP_DT_L2 * (t**2) / P_0
    noise = noise_std * rng.randn(n_points)
    dt_obs = dt_L1 + dt_L2 + noise
    dt_true = dt_L1 + dt_L2
    return t, dt_obs, dt_true


class TestPulsarSNRValidation:
    """
    Decision gate for Fase 3 (Pulsar Timing Infrastructure).
    These tests assert that the L2 recovery problem is tractable with modern instruments.
    """

    def test_l2_snr_above_threshold(self):
        """L2 signal SNR must be >= 5.0 after L1 subtraction at modern timing precision."""
        t, dt_obs, dt_true = _simulate_timing_residuals()
        dt_L1 = 0.5 * DP_DT_L1 * (t**2) / P_0
        delta_obs  = dt_obs  - dt_L1
        delta_true = dt_true - dt_L1

        signal_std = np.std(delta_true)
        noise_std  = np.std(delta_obs - delta_true)
        snr = signal_std / noise_std

        assert snr >= 5.0, (
            f"L2 SNR={snr:.2f} is below threshold 5.0. "
            f"Signal std={signal_std*1e6:.2f} µs, noise={noise_std*1e6:.2f} µs"
        )

    def test_l2_exponent_recovery(self):
        """ADCD power-law fit must recover time exponent within 0.15 of true value 2.0."""
        t, dt_obs, _ = _simulate_timing_residuals()
        dt_L1 = 0.5 * DP_DT_L1 * (t**2) / P_0
        delta_obs = dt_obs - dt_L1

        def loss_fn(params):
            log_amp, exponent = params
            amp = -np.exp(log_amp)
            pred = amp * (t ** exponent)
            return np.mean((pred - delta_obs) ** 2)

        res = minimize(
            loss_fn,
            np.array([np.log(1e-21), 1.8]),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-16, "maxiter": 5000},
        )
        recovered_exponent = res.x[1]

        assert abs(recovered_exponent - 2.0) < 0.15, (
            f"Time scaling exponent recovery error too large: "
            f"got {recovered_exponent:.4f}, expected 2.0000 ± 0.15"
        )

    def test_l2_nmse_below_threshold(self):
        """Residual NMSE of best-fit L2 model must be < 0.05 (>95% variance explained)."""
        t, dt_obs, _ = _simulate_timing_residuals()
        dt_L1 = 0.5 * DP_DT_L1 * (t**2) / P_0
        delta_obs = dt_obs - dt_L1

        def loss_fn(params):
            log_amp, exponent = params
            amp = -np.exp(log_amp)
            pred = amp * (t ** exponent)
            return np.mean((pred - delta_obs) ** 2)

        res = minimize(
            loss_fn,
            np.array([np.log(1e-21), 1.8]),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-16, "maxiter": 5000},
        )
        nmse = res.fun / (np.var(delta_obs) + 1e-30)

        assert nmse < 0.05, (
            f"L2 residual NMSE={nmse:.4f} exceeds 0.05. "
            f"L2 signal may be indistinguishable from noise at given precision."
        )

    def test_l2_unrecoverable_at_1us_noise(self):
        """At old 1.0 µs timing precision, SNR drops < 3.0 — confirming instrument requirement."""
        t, dt_obs, dt_true = _simulate_timing_residuals(noise_std=1.0e-6)
        dt_L1 = 0.5 * DP_DT_L1 * (t**2) / P_0
        delta_obs  = dt_obs  - dt_L1
        delta_true = dt_true - dt_L1

        signal_std = np.std(delta_true)
        noise_std  = np.std(delta_obs - delta_true)
        snr = signal_std / noise_std

        # At 1.0 µs precision, SNR ~ 2.0; too low to reliably detect L2
        assert snr < 3.0, (
            f"Expected SNR < 3.0 at 1µs precision, got {snr:.2f}. "
            f"This test validates the instrument precision requirement."
        )
