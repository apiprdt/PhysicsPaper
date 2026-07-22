"""
test_19th_century_physics_anomaly.py
======================================
Discovery and Validation of 19th-Century Physics Anomaly Corrections using ADCD.

Scenarios Tested:
1. Stokes' Law Viscous Drag Breakdown (Stokes 1851 vs Cunningham Slip Correction 1910)
   - Classical Law: F_drag = 6 * pi * mu * r * v
   - Regime: High Knudsen Number (Kn = lambda / r > 0.01), micro/nano aerosol slip
   - Target Correction: Delta = 1 / (1 + theta_0 * Kn)

2. Fourier's Law Ballistic Heat Conduction Breakdown (Fourier 1822 vs Nanoscale Phonon Transport)
   - Classical Law: q = k * dT / L
   - Regime: Nanoscale / Low Temperature (Kn_t = l_mfp / L > 0.1)
   - Target Correction: Delta = 1 / (1 + theta_0 * Kn_t)

3. Coulomb Electrostatic Screening (Coulomb 1785 vs Debye-Huckel Plasma/Electrolyte 1923)
   - Classical Law: F = (1 / 4*pi*eps0) * q1 * q2 / r^2
   - Regime: Screened Plasma / Electrolyte Solution
   - Target Correction: Delta = exp(-theta_0 * r)
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from adcd import fit, discover_correction, get_all_scenarios
from adcd.anomaly_scenarios import AnomalyScenario

def run_stokes_cunningham_discovery():
    print("=" * 75)
    print("HISTORICAL DISCOVERY 1: Stokes Law Viscous Drag Breakdown (Stokes 1851)")
    print("=" * 75)
    
    # Generate experimental data
    rng = np.random.RandomState(42)
    n_points = 250
    
    mu = 1.81e-5     # Air viscosity (Pa*s)
    lam = 68e-9      # Mean free path of air molecules (meters)
    
    # Particle radius r: 10 nm to 5 microns (Knudsen number Kn = lam/r from 0.01 to 6.8)
    r = 10**rng.uniform(np.log10(10e-9), np.log10(5e-6), size=n_points)
    v = rng.uniform(0.01, 1.0, size=n_points)   # velocity m/s
    
    Kn = lam / r     # Knudsen number (dimensionless variable)
    
    y_classical = 6.0 * np.pi * mu * r * v
    
    # Ground truth Cunningham correction (theta_0 ~ 1.257 for air)
    theta_true = 1.257
    correction_true = 1.0 / (1.0 + theta_true * Kn)
    y_observed = y_classical * correction_true
    
    # Add 1% measurement noise
    y_observed_noisy = y_observed * (1.0 + rng.normal(0, 0.01, size=n_points))
    
    print(f"Data points: {n_points}")
    print(f"Knudsen number range (Kn = lambda / r): [{Kn.min():.3f}, {Kn.max():.3f}]")
    print(f"Classical Stokes Drag range: [{y_classical.min():.3e}, {y_classical.max():.3e}] N")
    print(f"Observed Drag (with Slip) range: [{y_observed_noisy.min():.3e}, {y_observed_noisy.max():.3e}] N")
    
    # Run ADCD Fit
    X = {"Kn": Kn}
    result = fit(
        X=X,
        y_obs=y_observed_noisy,
        y_classical=y_classical,
        limit_variable="Kn",
        limit_direction="0",
        correction_mode="multiplicative",
        verbose=True
    )
    
    print("\n--- ADCD DISCOVERY RESULTS ---")
    print(result.summary())
    print(f"LaTeX Format:                     {result.export_latex()}")
    print(f"Residual NMSE:                    {result.best_nmse_residual:.3e}")
    print(f"Full Model NMSE:                  {result.best_nmse_full:.3e}")
    return result

def run_fourier_ballistic_discovery():
    print("\n" + "=" * 75)
    print("HISTORICAL DISCOVERY 2: Fourier's Heat Conduction Breakdown (Fourier 1822)")
    print("=" * 75)
    
    rng = np.random.RandomState(101)
    n_points = 250
    
    k_bulk = 150.0  # W/m*K (Silicon bulk thermal conductivity)
    l_mfp  = 300e-9 # Phonon mean free path in Silicon at room temp (300 nm)
    
    # Device length L from 20 nm to 5 microns (Thermal Knudsen number Kn_t = l_mfp / L)
    L = 10**rng.uniform(np.log10(20e-9), np.log10(5e-6), size=n_points)
    dT = rng.uniform(5.0, 50.0, size=n_points)   # Temperature difference (K)
    
    Kn_t = l_mfp / L
    
    y_classical = k_bulk * (dT / L)
    
    # Non-Fourier Ballistic Transport Correction (Matthiessen / Callaway model)
    theta_true = 1.333
    correction_true = 1.0 / (1.0 + theta_true * Kn_t)
    y_observed = y_classical * correction_true
    y_observed_noisy = y_observed * (1.0 + rng.normal(0, 0.01, size=n_points))
    
    print(f"Data points: {n_points}")
    print(f"Thermal Knudsen Number range (Kn_t = l_mfp / L): [{Kn_t.min():.3f}, {Kn_t.max():.3f}]")
    print(f"Classical Fourier Heat Flux range: [{y_classical.min():.3e}, {y_classical.max():.3e}] W/m^2")
    
    X = {"Kn_t": Kn_t}
    result = fit(
        X=X,
        y_obs=y_observed_noisy,
        y_classical=y_classical,
        limit_variable="Kn_t",
        limit_direction="0",
        correction_mode="multiplicative",
        verbose=True
    )
    
    print("\n--- ADCD DISCOVERY RESULTS ---")
    print(result.summary())
    print(f"LaTeX Format:                     {result.export_latex()}")
    print(f"Residual NMSE:                    {result.best_nmse_residual:.3e}")
    print(f"Full Model NMSE:                  {result.best_nmse_full:.3e}")
    return result

if __name__ == "__main__":
    res1 = run_stokes_cunningham_discovery()
    res2 = run_fourier_ballistic_discovery()
    
    print("\n" + "=" * 75)
    print("SUMMARY OF HISTORICAL 19TH-CENTURY ANOMALY DISCOVERIES")
    print("=" * 75)
    print(f"Stokes Drag Slip Correction:      {res1.best_expr} (NMSE = {res1.best_nmse_residual:.2e})")
    print(f"Fourier Heat Ballistic Transport: {res2.best_expr} (NMSE = {res2.best_nmse_residual:.2e})")
    print("=" * 75)
