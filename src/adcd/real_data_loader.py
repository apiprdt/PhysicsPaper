"""
Real experimental data loaders for ADCD v2.0.

Each loader generates "synthetic-real hybrid" data — using real physical constants
and parameters from actual experiments, but computing the data programmatically.
The physics is correct and the noise models are realistic.

All loaders return (X_dict, y_obs, y_classical, residual), matching the format
from AnomalyScenario.generate_data().
"""

import numpy as np
from typing import Dict, Tuple


def load_mercury_perihelion(
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Mercury orbital precession data from JPL DE440 parameters.

    Classical Newtonian 2-body gives zero precession; GR predicts
    δφ_per_orbit = 6πGM/(c²·a(1-e²)) ≈ 5.02e-7 rad/orbit (42.98 arcsec/century).

    Leading-order Schwarzschild correction scales as (v/c)² along the orbit.
    We expose the dimensionless parameter beta = (v/c)² for stable optimization.

    Returns 200 points parametrized by true anomaly θ ∈ [0, 2π).
    Correction type: additive (classical prediction is zero).
    """
    rng = np.random.RandomState(seed)

    G = 6.674e-11
    M_sun = 1.989e30
    c = 2.998e8
    a = 5.791e10
    e = 0.2056

    n_points = 200
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    v = np.sqrt(G * M_sun * (2.0 / r - 1.0 / a))
    vc2 = (v / c) ** 2

    y_classical = np.zeros(n_points)

    delta_phi_per_orbit = 6.0 * np.pi * G * M_sun / (c**2 * a * (1 - e**2))
    y_true = delta_phi_per_orbit * vc2

    noise = rng.normal(0, 0.001 * np.mean(np.abs(y_true)), size=n_points)
    y_obs = y_true + noise
    residual = y_obs - y_classical

    X = {"vc2": vc2, "r": r, "v": v, "theta": theta}
    return X, y_obs, y_classical, residual


def load_binary_pulsar_decay(
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Binary pulsar orbital period decay (Hulse-Taylor / PSR B1913+16 inspired).

    Classical Kepler: P = const with no secular change.
    GR quadrupole radiation predicts |dP/dt| via Peters & Matthews (1964).

    Key fix (v2.1): M, a, e held at Hulse-Taylor values; only P varies.
    Eliminating mass variation ensures dP/dt ∝ P^(-5/3) cleanly, matching
    the Peters formula in standard orbital-period form:

        dP/dt = -(192π/5) · [G·M/(2π)]^(5/3) / c^5 · (m1·m2/M) · f(e) · P^(-5/3)

    Synthetic-real hybrid: uses real CODATA/JPL constants + B1913+16 parameters.
    Correction type: additive (Newtonian dP/dt = 0).
    """
    rng = np.random.RandomState(seed)

    G = 6.674e-11        # gravitational constant [m³/kg/s²]
    c = 2.998e8          # speed of light [m/s]
    M_sun = 1.989e30     # solar mass [kg]

    # Hulse-Taylor binary PSR B1913+16 — fixed parameters
    m1 = 1.4408 * M_sun   # pulsar mass [kg]
    m2 = 1.3873 * M_sun   # companion mass [kg]
    M  = m1 + m2           # total mass [kg]
    mu = m1 * m2 / M       # reduced mass [kg]
    e  = 0.617             # eccentricity (constant)

    # Enhancement factor f(e) — Peters & Matthews (1964)
    f_e = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) / (1.0 - e**2) ** (7.0 / 2.0)

    # Peters formula prefactor (constant for fixed M, mu, e):
    # dP/dt = -A · P^(-5/3)   where A > 0
    A = (192.0 * np.pi / 5.0) * (G * M / (2.0 * np.pi)) ** (5.0 / 3.0) / c**5 * (mu / M) * f_e

    # Vary P directly over an inspiral range (7–9 hours, in seconds)
    n_points = 60
    P = np.linspace(7.0 * 3600.0, 9.0 * 3600.0, n_points)   # [s]

    # Ground truth: |dP/dt| = A · P^(-5/3)
    y_true = A * P ** (-5.0 / 3.0)
    y_classical = np.zeros(n_points)

    # 0.2% observational noise (pulsar timing is highly precise)
    noise = rng.normal(0.0, 0.002 * np.mean(y_true), size=n_points)
    y_obs = y_true + noise
    residual = y_obs - y_classical

    # Only P is the free variable; M, a, e folded into constant prefactor A
    X = {"P": P}
    return X, y_obs, y_classical, residual


def load_hydrogen_lamb_shift(
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Hydrogen Lamb shift data.
    
    The Dirac equation predicts degenerate 2S₁/₂ and 2P₁/₂ levels.
    QED radiative corrections (Lamb shift) lift this degeneracy.
    
    ΔE ∝ α⁵·mₑ·c²·ln(1/α²) / n³
    
    Returns 18 data points for n = 2..19.
    Correction type: additive.
    """
    rng = np.random.RandomState(seed)
    
    # Physical constants
    alpha = 7.297e-3     # fine structure constant
    m_e = 9.109e-31      # electron mass [kg]
    c = 2.998e8          # speed of light [m/s]
    # h = 6.626e-34      # Planck constant [J·s] (unused, kept for reference)
    eV = 1.602e-19       # electron-volt [J]
    
    # Quantum numbers
    n_values = np.arange(2, 20, dtype=np.float64)  # n = 2 to 19
    n_points = len(n_values)
    
    # Dirac prediction (leading order): E_n = -13.6 / n² eV
    E_dirac = -13.6 / n_values**2  # [eV]
    y_classical = E_dirac
    
    # Lamb shift: ΔE ∝ α⁵ · m_e · c² · ln(1/α²) / n³
    # Convert to eV
    delta_E_joules = alpha**5 * m_e * c**2 * np.log(1.0 / alpha**2) / n_values**3
    delta_E_eV = delta_E_joules / eV
    
    y_true = y_classical + delta_E_eV
    
    # Add measurement noise: 0.1% of mean |delta|
    noise_scale = 0.001 * np.mean(np.abs(delta_E_eV))
    noise = rng.normal(0, noise_scale, size=n_points)
    y_obs = y_true + noise
    
    # Additive residual
    residual = y_obs - y_classical
    
    X = {"n": n_values}
    return X, y_obs, y_classical, residual


def load_blackbody_radiation(
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Blackbody radiation: Rayleigh-Jeans (classical) vs Planck (quantum).
    
    At low frequencies, Rayleigh-Jeans ≈ Planck.
    At high frequencies, the classical law diverges (ultraviolet catastrophe).
    
    T = 5778 K (Sun surface temperature), f ∈ [1e11, 2e15] Hz.
    Correction type: multiplicative.
    """
    rng = np.random.RandomState(seed)
    
    # Physical constants
    k_B = 1.381e-23     # Boltzmann constant [J/K]
    h = 6.626e-34        # Planck constant [J·s]
    c = 2.998e8          # speed of light [m/s]
    T = 5778.0           # Sun surface temperature [K]
    
    n_points = 200
    
    # Frequency range (log-spaced to cover UV catastrophe regime)
    f = np.geomspace(1e11, 2e15, n_points)
    T_arr = np.full_like(f, T)
    
    # Rayleigh-Jeans classical prediction
    I_rj = 2.0 * k_B * T * f**2 / c**2
    y_classical = I_rj
    
    # Planck quantum prediction
    x = h * f / (k_B * T)
    # Use numerically stable computation: avoid overflow in exp(x)
    I_planck = np.where(
        x < 500,
        (2.0 * h * f**3 / c**2) / (np.exp(x) - 1.0),
        0.0  # exponentially suppressed for very high frequencies
    )
    y_true = I_planck
    
    # Add 0.5% multiplicative noise
    noise = rng.normal(0, 0.005, size=n_points)
    y_obs = y_true * (1.0 + noise)
    
    # Multiplicative residual
    residual = y_obs / y_classical - 1.0
    
    X = {"f": f, "T": T_arr}
    return X, y_obs, y_classical, residual


def load_muon_g2(
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Muon anomalous magnetic moment (g-2).
    
    Classical Dirac: g = 2 exactly, so anomaly a_μ = (g-2)/2 = 0.
    QED corrections: a = α/(2π) + 0.765857·(α/π)² + 24.05·(α/π)³ + ...
    
    Scan α_eff from 0.5α to 2.0α (100 points) to map the perturbative landscape.
    Correction type: additive (classical prediction is zero).
    """
    rng = np.random.RandomState(seed)
    
    # Physical constants
    alpha = 7.297e-3     # fine structure constant
    
    n_points = 100
    
    # Scan effective fine structure constant
    alpha_eff = np.linspace(alpha * 0.5, alpha * 2.0, n_points)
    
    # Classical Dirac prediction: g = 2, a_μ = 0
    y_classical = np.zeros(n_points)
    
    # QED perturbative expansion (Schwinger + higher orders)
    a_over_pi = alpha_eff / np.pi
    a_qed = (
        alpha_eff / (2.0 * np.pi)           # Schwinger term
        + 0.765857 * a_over_pi**2            # 2-loop
        + 24.05 * a_over_pi**3               # 3-loop
    )
    y_true = a_qed
    
    # Add noise ~ 0.5 ppm of mean value
    noise_scale = 0.5e-6 * np.mean(np.abs(a_qed))
    noise = rng.normal(0, noise_scale, size=n_points)
    y_obs = y_true + noise
    
    # Additive residual
    residual = y_obs - y_classical
    
    X = {"alpha": alpha_eff}
    return X, y_obs, y_classical, residual


# ── Convenience function ──────────────────────────────────────────────────────

def load_all_real_data() -> Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    """Load all real experiment datasets, keyed by scenario name."""
    return {
        "Real: Mercury Perihelion": load_mercury_perihelion(),
        "Real: Hydrogen Lamb Shift": load_hydrogen_lamb_shift(),
        "Real: Blackbody Radiation": load_blackbody_radiation(),
        "Real: Muon g-2": load_muon_g2(),
        "Real: Binary Pulsar Decay": load_binary_pulsar_decay(),
    }
