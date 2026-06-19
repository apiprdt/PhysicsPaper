"""
SPARC rotation-curve loading and dimensionless stacking for MOND experiments.

Data source: Lelli, McGaugh & Schombert 2016 (AJ 152, 157)
File: MassModels_Lelli2016c.mrt — combined rotation curves for 175 LTGs.

Stacking (McGaugh RAR convention):
  x = g_bar / a_0        with g_bar = V_bar^2 / r  (SI units)
  nu = (V_obs / V_bar)^2
  Classical Newtonian prediction: nu = 1  =>  target correction Δν = nu - 1
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

# Primary + mirror URLs (astroweb migrated case.edu; legacy cwru often down)
SPARC_URLS = [
    "https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt",
    "http://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt",
]
DEFAULT_CACHE = "data/sparc/MassModels_Lelli2016c.mrt"
G_DAGGER = 1.2e-10  # m/s^2 — Milgrom acceleration scale
KPC_TO_M = 3.085677581e19
KM_TO_M = 1000.0

DataSource = Literal["REAL", "SIMULATED"]

SPARC_COLUMNS = [
    "galaxy", "distance_mpc", "radius_kpc", "v_obs", "e_v_obs",
    "v_gas", "v_disk", "v_bulge", "sb_disk", "sb_bulge",
]


@dataclass
class SparcStackResult:
    x: np.ndarray
    nu_obs: np.ndarray
    nu_classical: np.ndarray
    sigma_nu: np.ndarray
    n_galaxies: int
    n_points: int
    data_source: DataSource


def download_sparc(cache_path: str = DEFAULT_CACHE, timeout: int = 30) -> bool:
    """Try each mirror URL; return True if file cached successfully."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        return True
    for url in SPARC_URLS:
        try:
            print(f"Trying SPARC download: {url}")
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
            with open(cache_path, "wb") as fh:
                fh.write(data)
            if os.path.getsize(cache_path) > 1000:
                print("Download complete.")
                return True
        except Exception as exc:
            print(f"  failed: {exc}")
    return False


def parse_sparc_mrt(cache_path: str) -> pd.DataFrame:
    """Parse combined MassModels MRT table (whitespace-separated)."""
    rows = []
    with open(cache_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("Byte"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                _ = float(parts[1])
                _ = float(parts[2])
            except ValueError:
                continue
            rows.append({
                "galaxy": parts[0],
                "distance_mpc": float(parts[1]),
                "radius_kpc": float(parts[2]),
                "v_obs": float(parts[3]),
                "e_v_obs": float(parts[4]),
                "v_gas": float(parts[5]),
                "v_disk": float(parts[6]),
                "v_bulge": float(parts[7]),
                "sb_disk": float(parts[8]),
                "sb_bulge": float(parts[9]) if len(parts) > 9 else 0.0,
            })
    if not rows:
        raise ValueError(f"No SPARC rows parsed from {cache_path}")
    return pd.DataFrame(rows)


def _simulate_sparc_stack(n_samples: int = 2000, seed: int = 42) -> SparcStackResult:
    """High-fidelity Simple-MOND synthetic benchmark — explicitly labeled."""
    rng = np.random.default_rng(seed)
    g_bar = 10 ** rng.uniform(-12, -9, n_samples)
    x = g_bar / G_DAGGER
    nu_true = (1.0 + np.sqrt(1.0 + 4.0 / x)) / 2.0
    sigma = 0.05 * nu_true
    nu_obs = np.maximum(nu_true + rng.normal(0.0, sigma), 1.01)
    return SparcStackResult(
        x=x, nu_obs=nu_obs, nu_classical=np.ones_like(x),
        sigma_nu=sigma, n_galaxies=0, n_points=n_samples, data_source="SIMULATED",
    )


def stack_sparc_galaxies(
    df: pd.DataFrame,
    min_v_bar: float = 5.0,
    max_nu: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Stack galaxies into dimensionless (x, nu) arrays.

    V_bar = sqrt(V_gas^2 + V_disk^2 + V_bulge^2)  [km/s, Lelli+2016 convention]
    """
    x_all, nu_all, sigma_all = [], [], []
    galaxies_used = 0

    for _, grp in df.groupby("galaxy"):
        r_kpc = grp["radius_kpc"].values
        v_obs = grp["v_obs"].values
        e_v = grp["e_v_obs"].values
        v_bar = np.sqrt(grp["v_gas"].values ** 2 + grp["v_disk"].values ** 2 + grp["v_bulge"].values ** 2)

        r_m = r_kpc * KPC_TO_M
        v_bar_ms = v_bar * KM_TO_M
        v_obs_ms = v_obs * KM_TO_M

        valid = (
            (r_kpc > 0)
            & (v_bar > min_v_bar)
            & np.isfinite(v_obs)
            & np.isfinite(v_bar)
        )
        if valid.sum() < 5:
            continue

        r_m, v_bar_ms, v_obs_ms, e_v = r_m[valid], v_bar_ms[valid], v_obs_ms[valid], e_v[valid]
        g_bar = v_bar_ms ** 2 / r_m
        x = g_bar / G_DAGGER
        nu = (v_obs_ms / v_bar_ms) ** 2

        good = (nu > 0) & (nu < max_nu) & np.isfinite(nu) & np.isfinite(x) & (x > 0)
        if good.sum() < 3:
            continue

        x_g, nu_g = x[good], nu[good]
        e_v_g = e_v[good]
        v_b_g = v_bar_ms[good]
        # Propagate velocity uncertainty: sigma_nu ≈ 2*(Vobs/Vbar^2)*sigma_Vobs
        sigma_nu = 2.0 * v_obs_ms[good] * (e_v_g * KM_TO_M) / (v_b_g ** 2)

        x_all.extend(x_g)
        nu_all.extend(nu_g)
        sigma_all.extend(sigma_nu)
        galaxies_used += 1

    return (
        np.asarray(x_all), np.asarray(nu_all), np.asarray(sigma_all),
        np.ones(len(x_all)), galaxies_used,
    )


def load_sparc_stack(
    cache_path: str = DEFAULT_CACHE,
    allow_simulated_fallback: bool = True,
) -> SparcStackResult:
    """
    Load real SPARC stack or fall back to simulated benchmark.

    Returns SparcStackResult with explicit data_source label.
    """
    if download_sparc(cache_path):
        try:
            df = parse_sparc_mrt(cache_path)
            x, nu, sigma, nu_classical, n_gal = stack_sparc_galaxies(df)
            if len(x) >= 100:
                print("=== REAL DATA ===")
                print(f"Stacked {n_gal} galaxies, {len(x)} points from {cache_path}")
                return SparcStackResult(
                    x=x, nu_obs=nu, nu_classical=nu_classical, sigma_nu=sigma,
                    n_galaxies=n_gal, n_points=len(x), data_source="REAL",
                )
            print(f"REAL file parsed but only {len(x)} valid points — check filters.")
        except Exception as exc:
            print(f"SPARC parse/stack failed: {exc}")

    if not allow_simulated_fallback:
        raise RuntimeError("SPARC real data unavailable and simulated fallback disabled.")

    print("=== SIMULATED DATA ===")
    print("Synthetic Simple-MOND stack — benchmark only, NOT a real SPARC discovery.")
    return _simulate_sparc_stack()
