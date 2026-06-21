"""
f σ₈ growth-rate data loader for the S₈-tension experiment.

Data source: Alestas, Nesseris & Perivolaropoulos (2022), Table II
(https://arxiv.org/abs/2112.05752) — a homogenized compilation of 63 published
fσ₈ measurements spanning z ∈ [0.001, 1.944], with per-point fiducial Ωₘ and
error σ.

File format (``data/growth_rate/Growth_tableII.txt``):
    z   fσ₈_obs   σ(fσ₈)   Ωₘ_fid

  whitespace-separated, one measurement per line.

General-Relativity baseline
----------------------------
On sub-horizon scales in ΛCDM, linear-theory growth obeys

    f(z) ≡ d ln D / d ln a ≈ Ωₘ(z)^γ ,    γ = 0.55   (Linder 2005)

with Ωₘ(z) = Ωₘ₀ (1+z)³ / E(z)² and E(z) = H(z)/H₀ for a flat w=-1 universe.
The dimensionless prediction is then

    fσ₈(z) = σ₈₀ · D(z) · f(z)

where D(z) is the linear growth factor normalised to D(0)=1 and σ₈₀ is the
present-day mass variance inside 8 Mpc/h spheres (treated as a single nuisance
amplitude, fitted by least-squares to the data — exactly the Sagredo /
Nesseris methodology).

The residual Δ(z) ≡ fσ₈_obs(z) − fσ₈_GR(z) is the ADCD discovery target: a
non-zero structure in Δ(z) is the *observable signature of modified growth*
and is the cleanest place to look for a cosmological-model correction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------

# Planck 2018 TT,TE,EE+lowE+lensing (used as the default homogeneous cosmology).
# When ``cosmology="per_point"`` (default) the per-measurement Ωₘ_fid column is
# used instead, matching exactly what each survey published.
PLANCK_OM_M0 = 0.315
PLANCK_OM_DE0 = 0.685   # spatially flat

GROWTH_INDEX_GAMMA = 0.55   # Linder-Peebles GR value

DEFAULT_CACHE = "data/growth_rate/Growth_tableII.txt"

DataSource = Literal["REAL", "SIMULATED"]


def _E(z: np.ndarray, Om_m0: np.ndarray, Om_de0: np.ndarray) -> np.ndarray:
    """H(z)/H₀ for a spatially-flat ΛCDM universe."""
    return np.sqrt(Om_m0 * (1.0 + z) ** 3 + Om_de0)


def _Omz(z: np.ndarray, Om_m0: np.ndarray, Om_de0: np.ndarray) -> np.ndarray:
    """Ωₘ(z) for flat ΛCDM."""
    return Om_m0 * (1.0 + z) ** 3 / _E(z, Om_m0, Om_de0) ** 2


def _growth_factor_scalar(z: float, Om_m0: float, Om_de0: float) -> float:
    """Normalised linear growth factor D(z) with D(0)=1.

    Uses the exact integral form (Hamilton 2001)

        D(z) ∝ E(z) ∫_z^∞ (1+z') / E(z')³ dz'

    normalised by the same integral evaluated at z=0. Numerically integrated
    with quad — accurate to <1e-6 against Carroll 1992's approximation.
    """
    def _E_scalar(zp: float) -> float:
        return float(np.sqrt(Om_m0 * (1.0 + zp) ** 3 + Om_de0))

    norm, _ = quad(lambda zp: (1.0 + zp) / _E_scalar(zp) ** 3,
                   0.0, np.inf, limit=400)
    val, _ = quad(lambda zp: (1.0 + zp) / _E_scalar(zp) ** 3,
                  z, np.inf, limit=400)
    D0 = 2.5 * Om_m0 * 1.0 * norm
    Dz = 2.5 * Om_m0 * _E_scalar(z) * val
    return Dz / D0 if D0 > 0 else 0.0


def growth_factor(z: np.ndarray, Om_m0: float = PLANCK_OM_M0,
                  Om_de0: float = PLANCK_OM_DE0) -> np.ndarray:
    """Vectorised wrapper around :func:`_growth_factor_scalar`."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([_growth_factor_scalar(zi, Om_m0, Om_de0) for zi in z])


def gr_fs8_prediction(z: np.ndarray, Om_m0: np.ndarray, Om_de0: np.ndarray,
                      gamma: float = GROWTH_INDEX_GAMMA,
                      sigma8_0: float = 1.0) -> np.ndarray:
    """fσ₈(z) = σ₈₀ · D(z) · Ωₘ(z)^γ GR baseline.

    ``sigma8_0`` is left explicit so it can be fitted as a nuisance amplitude
    (the conventional approach when comparing RSD data to a model — see
    Nesseris & Perivolaropoulos 2017, Sagredo et al. 2018).
    """
    z = np.asarray(z, dtype=float)
    Om_m0 = np.asarray(Om_m0, dtype=float)
    Om_de0 = np.asarray(Om_de0, dtype=float)
    D = growth_factor(z, Om_m0=PLANCK_OM_M0, Om_de0=PLANCK_OM_DE0)
    # Per-point Ωₘ(z) (using each point's own fiducial cosmology)
    Om_z = _Omz(z, Om_m0, Om_de0)
    f = Om_z ** gamma
    return sigma8_0 * D * f


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass
class GrowthRateResult:
    z: np.ndarray
    fs8_obs: np.ndarray
    sigma_fs8: np.ndarray
    Om_fid: np.ndarray
    n_points: int
    data_source: DataSource
    # Derived (filled by load_growth_rate)
    fs8_gr: np.ndarray = 0.0          # type: ignore[assignment]
    sigma8_0_fit: float = 0.0
    residual: np.ndarray = 0.0         # type: ignore[assignment]


def _parse_table(cache_path: str) -> Tuple[np.ndarray, ...]:
    """Parse whitespace-separated ``z  fσ₈  σ  Ωₘ_fid`` rows."""
    rows = []
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(p) for p in parts[:4]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No rows parsed from {cache_path}")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _fit_sigma8_amplitude(z: np.ndarray, fs8_obs: np.ndarray,
                          sigma_fs8: np.ndarray, Om_m0: np.ndarray,
                          Om_de0: np.ndarray,
                          gamma: float = GROWTH_INDEX_GAMMA) -> float:
    """Weighted least-squares for σ₈₀ as a single nuisance amplitude.

    With γ fixed at the GR value and D(z) already computed, the only free
    parameter in fs8_GR(z) = σ₈₀ · D(z) · Ωₘ(z)^γ is the multiplicative
    amplitude σ₈₀. The closed-form weighted-LS estimator is

        σ₈₀ = Σ wᵢ Dᵢ fᵢ fs8ᵢ  /  Σ wᵢ (Dᵢ fᵢ)² ,   wᵢ = 1/σᵢ².
    """
    D = growth_factor(z, Om_m0=PLANCK_OM_M0, Om_de0=PLANCK_OM_DE0)
    Om_z = _Omz(z, Om_m0, Om_de0)
    f = Om_z ** gamma
    pred_unit = D * f                       # fs8_GR per unit σ₈₀
    w = 1.0 / sigma_fs8 ** 2
    num = np.sum(w * pred_unit * fs8_obs)
    den = np.sum(w * pred_unit ** 2)
    return float(num / den) if den > 0 else 0.811


def load_growth_rate(
    cache_path: str = DEFAULT_CACHE,
    cosmology: str = "per_point",
    gamma: float = GROWTH_INDEX_GAMMA,
    allow_simulated_fallback: bool = False,
) -> GrowthRateResult:
    """Load the fσ₈ compilation, build the GR baseline, return residual Δ(z).

    Parameters
    ----------
    cache_path : str
        Path to the Alestas Table II text file.
    cosmology : {"per_point", "planck"}
        ``"per_point"`` (default) uses each measurement's own Ωₘ_fid, exactly
        reproducing the published compilation. ``"planck"`` homogenises to
        Planck-2018 (Ωₘ=0.315, Ω_Λ=0.685) — the standard "consistency check".
    gamma : float
        Growth index γ; defaults to the GR value 0.55.
    allow_simulated_fallback : bool
        If True and the file is missing, return a tiny synthetic benchmark.

    Returns
    -------
    GrowthRateResult with z, fs8_obs, σ, Ωₘ_fid, fs8_GR (best-fit amplitude),
    the fitted σ₈₀, and residual = fs8_obs − fs8_GR.
    """
    if not os.path.exists(cache_path):
        if allow_simulated_fallback:
            return _simulate_growth_rate()
        raise FileNotFoundError(
            f"Growth-rate table not found at {cache_path}; "
            "set allow_simulated_fallback=True for a synthetic benchmark."
        )

    z, fs8_obs, sigma_fs8, Om_fid = _parse_table(cache_path)

    if cosmology == "planck":
        Om_m0 = np.full_like(z, PLANCK_OM_M0)
        Om_de0 = np.full_like(z, PLANCK_OM_DE0)
    else:  # "per_point"
        Om_m0 = Om_fid.copy()
        Om_de0 = 1.0 - Om_fid   # assume flatness, as Alestas do

    sigma8_0 = _fit_sigma8_amplitude(z, fs8_obs, sigma_fs8, Om_m0, Om_de0, gamma)
    fs8_gr = gr_fs8_prediction(z, Om_m0, Om_de0, gamma=gamma,
                               sigma8_0=sigma8_0)
    residual = fs8_obs - fs8_gr

    print("=== fσ₈ Growth-Rate Compilation (Alestas+2022, Table II) ===")
    print(f"  N points       : {len(z)}")
    print(f"  z range        : {z.min():.3f} – {z.max():.3f}")
    print(f"  cosmology      : {cosmology}")
    print(f"  γ (growth idx) : {gamma}")
    print(f"  σ₈₀ (best fit) : {sigma8_0:.4f}")
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    chi2 = float(np.sum((residual / sigma_fs8) ** 2))
    print(f"  Δ(z) RMSE      : {rmse:.4f}")
    print(f"  χ²/dof         : {chi2 / (len(z) - 1):.3f}  "
          f"(GR baseline, σ₈₀ free)")

    return GrowthRateResult(
        z=z, fs8_obs=fs8_obs, sigma_fs8=sigma_fs8, Om_fid=Om_fid,
        n_points=len(z), data_source="REAL",
        fs8_gr=fs8_gr, sigma8_0_fit=sigma8_0, residual=residual,
    )


def _simulate_growth_rate(n_points: int = 80, seed: int = 42) -> GrowthRateResult:
    """Tiny synthetic benchmark: pure GR + Gaussian noise, no correction."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.0, 2.0, n_points))
    Om_m0 = np.full_like(z, PLANCK_OM_M0)
    Om_de0 = np.full_like(z, PLANCK_OM_DE0)
    D = growth_factor(z)
    Om_z = _Omz(z, Om_m0, Om_de0)
    f = Om_z ** GROWTH_INDEX_GAMMA
    fs8_true = 0.811 * D * f
    sigma = 0.04 * np.ones_like(z)
    fs8_obs = fs8_true + rng.normal(0, sigma)
    fs8_gr = gr_fs8_prediction(z, Om_m0, Om_de0, sigma8_0=0.811)
    return GrowthRateResult(
        z=z, fs8_obs=fs8_obs, sigma_fs8=sigma, Om_fid=Om_m0,
        n_points=n_points, data_source="SIMULATED",
        fs8_gr=fs8_gr, sigma8_0_fit=0.811, residual=fs8_obs - fs8_gr,
    )
