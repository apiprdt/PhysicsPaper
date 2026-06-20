"""Centralized physical constants for ADCD.

Single source of truth for all physical constants used across the package
(data generation, evaluation, dimensional registries, plotting).

All values are CODATA / IAU recommended values. Importing from this module
instead of redefining constants locally prevents the silent inconsistencies
that previously existed (e.g. ``G`` was 6.674e-11 in some loaders and
6.6743e-11 in others; ``k_B`` appeared as 1.38e-23, 1.380649e-23, and
1.381e-23 across different files).

Reference: CODATA 2018 recommended values.
"""

from __future__ import annotations

# ── CODATA 2018 recommended values ───────────────────────────────────────────

# Newtonian gravitational constant  [m^3 kg^-1 s^-2]
G = 6.6743e-11

# Speed of light in vacuum  [m s^-1]  (exact)
C = 2.99792458e8

# Solar mass  [kg]  (IAU 2015 nominal solar mass)
M_SUN = 1.98892e30

# Boltzmann constant  [J K^-1]  (exact, CODATA 2018)
K_B = 1.380649e-23

# Planck constant  [J s]  (exact, CODATA 2018)
H = 6.62607015e-34

# Reduced Planck constant  [J s]
HBAR = 1.054571817e-34

# Elementary charge  [C]  (exact)
E = 1.602176634e-19

# Electron mass  [kg]
M_E = 9.1093837015e-31

# Vacuum permittivity  [F m^-1]  (exact, derived)
EPSILON_0 = 8.8541878128e-12

# Coulomb constant 1/(4*pi*epsilon_0)  [N m^2 C^-2]
K_E = 8.9875517923e9

# ── Astrophysics ─────────────────────────────────────────────────────────────

# Milgrom acceleration scale (MOND)  [m s^-2]
# Canonical value used throughout the SPARC stacking experiments.
G_DAGGER = 1.2e-10

# Length conversions
KPC_TO_M = 3.085677581e19     # kiloparsec -> metre (1 kpc = 1000 * 3.085677581e16 m)
KMS_TO_MS = 1000.0            # km/s -> m/s

# ── Common aliases (match the names historically used in the codebase) ──────
# These keep existing call sites readable while routing through this module.

# Many legacy call sites use the lowercase single-letter name as the dict key
# (e.g. ``classical_constants={"G": ..., "c": ..., "M": ...}``). Exposing both
# the canonical upper-case and the legacy aliases avoids a confusing rename.
c = C
M = M_SUN          # legacy key "M" used in coarse_evaluator DEFAULT_CONSTANTS
a_0 = G_DAGGER     # MOND literature convention
A0 = G_DAGGER


# Convenience bundle for consumers that build a constant-substitution dict.
# Mirrors the historical DEFAULT_CONSTANTS in coarse_evaluator.
DEFAULT_CONSTANTS = {
    "c": C,
    "G": G,
    "M": M_SUN,
}


__all__ = [
    "G", "C", "M_SUN", "K_B", "H", "HBAR", "E", "M_E",
    "EPSILON_0", "K_E",
    "G_DAGGER", "KPC_TO_M", "KMS_TO_MS",
    # aliases
    "c", "M", "a_0", "A0",
    "DEFAULT_CONSTANTS",
]
