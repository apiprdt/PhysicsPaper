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

from adcd.constants import G, C as c, M_SUN as M_sun, K_B as k_B, H as h, M_E as m_e, E as eV


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


def _binary_pulsar_constants():
    """Shared Hulse-Taylor / CODATA parameters for binary pulsar loaders.

    G, c and M_sun come from the centralized :mod:`adcd.constants` module so
    that every loader shares identical CODATA values.
    """
    m1 = 1.4408 * M_sun
    m2 = 1.3873 * M_sun
    M = m1 + m2
    mu = m1 * m2 / M
    a = 1.95e9
    return G, c, M_sun, m1, m2, M, mu, a


def binary_pulsar_prefactor(M=None, mu=None, e: float = 0.617) -> float:
    """Peters & Matthews (1964) prefactor A in |dP/dt| = A · P^(-5/3)."""
    G, c, _, _, _, M_def, mu_def, _ = _binary_pulsar_constants()
    M = M_def if M is None else M
    mu = mu_def if mu is None else mu
    f_e = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) / (1.0 - e**2) ** (7.0 / 2.0)
    return (
        (192.0 * np.pi / 5.0)
        * (G * M / (2.0 * np.pi)) ** (5.0 / 3.0)
        / c**5
        * (mu / M)
        * f_e
    )


def load_binary_pulsar_decay(
    seed: int = 42,
    variant: str = "P_only",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Binary pulsar orbital period decay (Hulse-Taylor / PSR B1913+16 inspired).

    Variants (for sensitivity analysis):
      P_only   — fixed M, a, e; only P varies (clean P^(-5/3), default benchmark)
      P_e      — vary P and eccentricity e
      P_e_M    — vary P, e, and total mass M
      full     — legacy multi-parameter scan (P, M, a, e all vary)
    """
    rng = np.random.RandomState(seed)
    G, c, M_sun, _, _, M_ht, mu_ht, a_ht = _binary_pulsar_constants()
    n_points = 60

    if variant == "full":
        M_total = np.linspace(2.4, 3.2, n_points) * M_sun
        a = np.full(n_points, a_ht)
        e = np.full(n_points, 0.617)
        mu = M_total / 2.0
        P = 2.0 * np.pi * np.sqrt(a**3 / (G * M_total))
        f_e = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) / (1.0 - e**2) ** (7.0 / 2.0)
        dP_dt = (
            -(192.0 * np.pi / 5.0 / c**5)
            * (G ** (5.0 / 2.0))
            * (mu * M_total**2)
            / a ** (5.0 / 2.0)
            * f_e
            * P ** (-5.0 / 3.0)
        )
        y_true = np.abs(dP_dt)
        X = {"P": P, "M": M_total, "a": a, "e": e}
    elif variant == "P_e_M":
        P = np.linspace(7.0 * 3600.0, 9.0 * 3600.0, n_points)
        e = np.linspace(0.55, 0.68, n_points)
        M = np.linspace(2.6, 3.0, n_points) * M_sun
        mu = M / 2.0
        y_true = np.array([
            binary_pulsar_prefactor(M=M[i], mu=mu[i], e=e[i]) * P[i] ** (-5.0 / 3.0)
            for i in range(n_points)
        ])
        X = {"P": P, "e": e, "M": M}
    elif variant == "P_e":
        P = np.linspace(7.0 * 3600.0, 9.0 * 3600.0, n_points)
        e = np.linspace(0.55, 0.68, n_points)
        y_true = np.array([
            binary_pulsar_prefactor(e=e[i]) * P[i] ** (-5.0 / 3.0)
            for i in range(n_points)
        ])
        X = {"P": P, "e": e}
    else:  # P_only (v2.1 default)
        e = 0.617
        A = binary_pulsar_prefactor(e=e)
        P = np.linspace(7.0 * 3600.0, 9.0 * 3600.0, n_points)
        y_true = A * P ** (-5.0 / 3.0)
        X = {"P": P}

    y_classical = np.zeros(n_points)
    noise = rng.normal(0.0, 0.002 * np.mean(y_true), size=n_points)
    y_obs = y_true + noise
    residual = y_obs - y_classical
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

    # Physical constants (m_e, c, eV via adcd.constants)
    alpha = 7.297e-3     # fine structure constant
    
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

    # Physical constants (routed through adcd.constants for consistency)
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
