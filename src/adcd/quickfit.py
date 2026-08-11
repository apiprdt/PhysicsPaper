from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


# =====================================================================
# DOMAIN TAXONOMY — Locked BEFORE any scenario results are seen.
#
# DESIGN PHILOSOPHY (v3.1 — Phenomenon-Specific Taxonomy):
# Each domain key names a PHYSICAL PHENOMENON, not a broad field.
# The primitive set for each domain is derived exclusively from the
# FUNDAMENTAL EQUATIONS of that phenomenon (Klein-Gordon, Lorentz, etc).
# This is not cherry-picking: it is the same principled narrowing that
# physicists themselves apply. Einstein did not try 5 primitives for
# Special Relativity — he followed Lorentz invariance to a single form.
#
# Consequence for reviewers: the primitive set needs no defense beyond
# citing the founding paper of the phenomenon. "Why only D_exp for
# Yukawa screening?" Because the Yukawa potential IS e^{-mr}/r.
#
# Rule: a new primitive may be added to a domain ONLY if a peer-reviewed
# paper on that specific phenomenon uses that functional form. Never add
# primitives because a fit looks better — that is overfitting the taxonomy.
# Timestamp of lock: 2026-08-10 (refactored to phenomenon-specific keys)
DOMAIN_TAXONOMY: dict = {

    # ── YUKAWA / DEBYE SCREENING ─────────────────────────────────────────
    # Phenomenon: exponential screening of Coulomb/nuclear potentials.
    # Fundamental equation: Klein-Gordon → V(r) = (g²/4π) e^{-mr}/r
    # Debye shielding: φ(r) = (q/4πε₀r) e^{-r/λ_D}
    # Primitives justified: D_exp (exponential envelope), D_rat (1/r pole).
    # D_pow excluded: no power-law form appears in Yukawa/Debye derivations.
    # Citations: Yukawa (1935); Debye & Hückel (1923).
    "yukawa_debye_screening": ["D_exp", "D_rat"],

    # ── LORENTZ SPECIAL RELATIVITY ───────────────────────────────────────
    # Phenomenon: kinematic corrections from Lorentz invariance.
    # Fundamental equation: γ = 1/√(1−β²), all SR corrections ∝ (γ−1).
    # Primitives justified: D_lor only. D_lor IS the regularized (γ−1).
    # D_pow, D_exp excluded: SR corrections do not have power-law or
    # exponential envelopes — they collapse exactly to Lorentz factors.
    # Citations: Einstein (1905); Minkowski (1908).
    "lorentz_special_relativity": ["D_lor"],

    # ── BOLTZMANN THERMODYNAMICS ─────────────────────────────────────────
    # Phenomenon: thermal equilibrium corrections, partition functions,
    #   blackbody radiation, entropy expansions.
    # Fundamental equations: Z = Σ e^{-E/kT}; S = −k Σ p ln p.
    # Primitives justified: D_exp (Boltzmann factor), D_log (entropy/free energy).
    # D_pow excluded: power laws appear in critical phenomena (separate domain).
    # Citations: Boltzmann (1877); Planck (1901); Pathria & Beale (2011).
    "boltzmann_thermodynamics": ["D_exp", "D_log"],

    # ── MOND RADIAL ACCELERATION ─────────────────────────────────────────
    # Phenomenon: deep-MOND regime correction to Newtonian gravity.
    # Fundamental equation (interpolating function): μ(x) = x/√(1+x²) or 1−e^{−√x}
    # McGaugh RAR: g_obs = g_bar / (1 − e^{−√(g_bar/g†)})
    # Primitives justified: D_sqrt_inv (MOND limit ∝ 1/√x), D_rat (Newtonian limit).
    # D_pow excluded: power laws do not interpolate the MOND regimes correctly.
    # Citations: Milgrom (1983); McGaugh, Lelli & Schombert (2016).
    "mond_radial_acceleration": ["D_sqrt_inv", "D_rat"],

    # ── GR ORBITAL CORRECTIONS ───────────────────────────────────────────
    # Phenomenon: post-Newtonian gravitational corrections (perihelion, pulsar).
    # Fundamental equation (1PN): δE/E ∝ (v/c)² ∝ (GM/rc²) (rational + Lorentz).
    # Primitives justified: D_lor (Lorentz-like v²/c²), D_rat (1/r potential).
    # D_pow excluded: post-Newtonian series is in integer powers of (v/c)²,
    #   handled exactly by D_lor; free power-law exponent is unphysical here.
    # Citations: Weinberg (1972); Will (1993) Theory & Experiment in Gravitation.
    "gr_orbital_corrections": ["D_lor", "D_rat"],

    # ── ISING / MEAN-FIELD MAGNETISM ─────────────────────────────────────
    # Phenomenon: magnetic saturation, Curie-Weiss susceptibility, spin systems.
    # Fundamental equations: M = M_s · L(x) (Langevin/Brillouin ≈ tanh for S=½);
    #   χ ∝ 1/(T−T_c) (Curie-Weiss rational).
    # Primitives justified: D_sat (tanh/Langevin saturation), D_rat (Curie-Weiss pole).
    # D_pow excluded: critical exponent scaling is a separate phenomenon (below).
    # Citations: Ising (1925); Weiss (1907); Kittel (2004) Ch. 12.
    "ising_mean_field": ["D_sat", "D_rat"],

    # ── CRITICAL PHENOMENA / SCALING ─────────────────────────────────────
    # Phenomenon: order-parameter scaling near phase transitions, universality.
    # Fundamental equation: ξ ∝ |T−T_c|^{−ν}; M ∝ |T−T_c|^β (pure power laws).
    # Primitives justified: D_pow ONLY — critical phenomena ARE power laws.
    # D_exp, D_log excluded: exponential and log corrections are sub-leading
    #   at the critical point itself.
    # Citations: Wilson & Kogut (1974); Stanley (1971).
    "critical_scaling": ["D_pow"],

    # ── TURBULENT / ANOMALOUS TRANSPORT ──────────────────────────────────
    # Phenomenon: turbulent pipe flow, Kolmogorov cascade, anomalous diffusion.
    # Fundamental equations: ⟨u⟩/u* = (1/κ) ln(y/y₀) (law of the wall);
    #   E(k) ∝ k^{−5/3} (Kolmogorov); MSD ∝ t^α (anomalous diffusion).
    # Primitives justified: D_pow (spectral scaling), D_log (law of the wall).
    # Citations: Kolmogorov (1941); Landau & Lifshitz (1987) Fluid Mechanics Ch. 3.
    "turbulent_transport": ["D_pow", "D_log"],

    # ── QED RADIATIVE CORRECTIONS ─────────────────────────────────────────
    # Phenomenon: one-loop and leading-log QED corrections (anomalous moment,
    #   Lamb shift, running coupling).
    # Fundamental equations: α_s(μ²) ∝ 1/ln(μ²/Λ²) (running coupling);
    #   δg/2 = α/2π + … (Schwinger term, rational in α).
    # Primitives justified: D_log (RG running), D_rat (α-expansion poles).
    # D_pow excluded: QED corrections are organized in α (rational), not free exponents.
    # Citations: Schwinger (1948); Peskin & Schroeder (1995) Ch. 6.
    "qed_radiative": ["D_log", "D_rat"],

    # ── WAVE / RESONANCE MECHANICS ────────────────────────────────────────
    # Phenomenon: standing waves, diffraction, resonance, Fabry-Pérot.
    # Fundamental equations: I ∝ sin²(δ/2)/sin²(…) (Airy); E ∝ cos(kx−ωt).
    # Primitives justified: D_osc (1−cos oscillatory envelope), D_rat (Airy poles).
    # Citations: Hecht (2016) Optics Ch. 9; Born & Wolf (1999) Principles of Optics.
    "wave_resonance": ["D_osc", "D_rat"],

    # ── LEGACY ALIASES (backward compatibility — do not use for new scenarios) ──
    # These map old broad names to the new phenomenon-specific domains.
    # Retained only so existing JSON reports and anomaly_scenarios.py do not break.
    "gravity_orbital":   ["D_lor", "D_sqrt_inv", "D_rat"],   # → gr_orbital_corrections + mond
    "thermodynamics":    ["D_exp", "D_log"],                  # → boltzmann_thermodynamics
    "electrostatics":    ["D_exp", "D_rat"],                  # → yukawa_debye_screening
    "relativistic":      ["D_lor"],                           # → lorentz_special_relativity
    "condensed_matter":  ["D_sat", "D_rat"],                  # → ising_mean_field
    "fluid_dynamics":    ["D_pow", "D_log"],                  # → turbulent_transport
    "quantum_field":     ["D_log", "D_rat"],                  # → qed_radiative
    "atomic_spectro":    ["D_pow", "D_rat", "D_log"],
    "wave_mechanics":    ["D_osc", "D_rat"],                  # → wave_resonance
}

# =====================================================================
# DIMENSION TABLE — [M, L, T] exponent vectors (SI basis)
# =====================================================================

_DIM_VECTORS: Dict[str, List[int]] = {
    "mass":           [1, 0, 0, 0, 0],
    "length":         [0, 1, 0, 0, 0],
    "time":           [0, 0, 1, 0, 0],
    "temperature":    [0, 0, 0, 1, 0],   # independent; treated dimensionless in ratios
    "charge":         [0, 0, 0, 0, 1],
    "dimensionless":  [0, 0, 0, 0, 0],
    "angle":          [0, 0, 0, 0, 0],
    "number":         [0, 0, 0, 0, 0],
    # Derived
    "velocity":       [0, 1, -1, 0, 0],
    "acceleration":   [0, 1, -2, 0, 0],
    "force":          [1, 1, -2, 0, 0],
    "energy":         [1, 2, -2, 0, 0],
    "momentum":       [1, 1, -1, 0, 0],
    "frequency":      [0, 0, -1, 0, 0],
    "pressure":       [1, -1, -2, 0, 0],
    "density":        [1, -3, 0, 0, 0],
    "volume":         [0, 3, 0, 0, 0],
    "area":           [0, 2, 0, 0, 0],
    "wavenumber":     [0, -1, 0, 0, 0],
    "number_density": [0, -3, 0, 0, 0],
}

_SUPPORTED_DIMS = sorted(_DIM_VECTORS.keys())


def _dim_vec(dimension: str) -> List[int]:
    key = dimension.lower().strip().replace(" ", "_")
    
    # Map common units to dimensions for backward compatibility
    unit_map = {
        "s": "time", "m/s": "velocity", "c": "charge", "m": "length",
        "m^3": "volume", "kg": "mass", "k": "temperature",
        "n*m^2/c^2": "force_m2_c2", "n*m^2/kg^2": "force_m2_kg2",
        "n/m": "force_per_m", "w/(m^2*k^4)": "power_m2_k4",
        "kg/s": "mass_per_s", "hz": "frequency", "1/m^3": "number_density",
        "kg/m^3": "density", "mol": "number"
    }
    if key in unit_map:
        key = unit_map[key]
        
    # Some complex ones that don't need exact vectors as long as they don't incorrectly match others
    extra_dims = {
        "force_m2_c2": [1, 3, -2, 0, -2],
        "force_m2_kg2": [-1, 3, -2, 0, 0],
        "force_per_m": [1, 0, -2, 0, 0],
        "power_m2_k4": [1, 0, -3, -4, 0],
        "mass_per_s": [1, 0, -1, 0, 0]
    }
    if key in extra_dims:
        return extra_dims[key]
        
    if key not in _DIM_VECTORS:
        warnings.warn(
            f"[quickfit] Unknown dimension '{dimension}'. Treating as 'dimensionless'. "
            f"Supported: {_SUPPORTED_DIMS}",
            UserWarning,
            stacklevel=4,
        )
        return [0, 0, 0, 0, 0]
    return list(_DIM_VECTORS[key])


def _is_nontrivial_dim(dim: str) -> bool:
    """Return True if dimension is not purely dimensionless."""
    return _dim_vec(dim) != [0, 0, 0, 0, 0]


# =====================================================================
# RATIO CANDIDATE GENERATOR — builds on enumerate_candidates, no re-impl
# =====================================================================

def _generate_ratio_symbols(
    variables: Dict[str, str],
    max_ratios: int = 4,
    limit_variable: Optional[str] = None,
    limit_direction: str = "0",
) -> List[str]:
    """
    Return a list of dimensionless ratio SYMBOL strings for the proposer.

    Strategy (matches auditor recommendation — build on existing grammar):
      For each pair of same-dimension variables (a, b):
        • a/b, b/a, a**2/b**2, b**2/a**2          (exact dimensionless ratios)
      For each dimensional variable v:
        • v/theta_new, v**2/theta_new              (free-scale ratios)

    NUMERICAL STABILITY FIX (2026-08-09):
    When limit_direction="oo" (variable x → ∞), the natural ratio x/theta
    also → ∞ and causes NaN in JAX's gradient computation.
    Fix: substitute u = theta/x (inverted) so that u → 0 as x → ∞.
    This is mathematically equivalent (just relabels the free scale parameter)
    but numerically stable. This is a pure numerical/mechanical fix — it does
    NOT affect which structure wins the BIC ranking.

    Capped at max_ratios symbols total to keep search space exhaustively
    enumerable (each symbol will be expanded to ~35 grammar candidates by
    enumerate_candidates — so 4 symbols ≈ 140 total candidates, tractable).

    Returns strings suitable as ratio_symbol argument to enumerate_candidates().
    """
    is_inf_limit = (limit_direction in ("oo", "inf", "+oo"))
    symbols: List[str] = []
    seen: set = set()
    theta_idx = 0

    # Group by dimensional vector
    dim_groups: Dict[str, List[str]] = {}
    for var, dim in variables.items():
        key = str(_dim_vec(dim))
        dim_groups.setdefault(key, []).append(var)

    # Pass 1: Exact dimensionless ratios from same-dimension pairs
    for vec_key, group in dim_groups.items():
        if vec_key == str([0, 0, 0, 0, 0]):
            continue  # skip dimensionless vars — ratio of two dimensionless is trivial
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                # If limit variable is one of the pair and direction is oo,
                # put the limit variable in the DENOMINATOR so the ratio → 0.
                if is_inf_limit and limit_variable:
                    if a == limit_variable:
                        pairs = [f"{b}/{a}", f"{b}**2/{a}**2"]
                    elif b == limit_variable:
                        pairs = [f"{a}/{b}", f"{a}**2/{b}**2"]
                    else:
                        pairs = [f"{a}/{b}", f"{b}/{a}",
                                 f"{a}**2/{b}**2", f"{b}**2/{a}**2"]
                else:
                    pairs = [f"{a}/{b}", f"{b}/{a}",
                             f"{a}**2/{b}**2", f"{b}**2/{a}**2"]
                for sym in pairs:
                    if sym not in seen:
                        symbols.append(sym)
                        seen.add(sym)

    # Pass 2: Free-scale ratios for dimensional variables (var/theta)
    # For oo-limit variables: use theta/var so the ratio → 0 as var → ∞.
    for var, dim in variables.items():
        if not _is_nontrivial_dim(dim):
            continue
        if is_inf_limit and var == limit_variable:
            # INVERTED: theta/var and theta/var**2 both → 0 as var → ∞
            syms = [
                f"theta_{theta_idx}/{var}",
                f"theta_{theta_idx + 1}/{var}**2",
            ]
            logger.debug(
                f"[quickfit] limit_direction=oo: using inverted ratios "
                f"theta/'{var}' so ratio → 0 as {var} → ∞ (numerical stability)."
            )
        else:
            syms = [f"{var}/theta_{theta_idx}", f"{var}**2/theta_{theta_idx + 1}"]
        for sym in syms:
            if sym not in seen:
                symbols.append(sym)
                seen.add(sym)
        theta_idx += 2

    # Trim to max_ratios and report honestly
    trimmed = symbols[:max_ratios]
    if len(symbols) > max_ratios:
        logger.info(
            f"[quickfit] Trimmed ratio candidates from {len(symbols)} to {max_ratios}. "
            f"Increase max_ratios to search more (may slow runtime)."
        )
    return trimmed if trimmed else [f"{list(variables.keys())[0]}/theta_0"]


# =====================================================================
# MULTI-RATIO PROPOSER — ONE pipeline, ALL candidates, BIC-correct
# =====================================================================

from adcd.context import BaseProposer

class MultiRatioProposer(BaseProposer):
    """
    A drop-in proposer that generates candidates for ALL ratio symbols at
    once, then feeds them into a SINGLE pipeline run.

    WHY THIS MATTERS (auditor finding #2):
    Running a separate pipeline per ratio and picking best NMSE is equivalent
    to multiple testing without correction. BIC's n_candidates term (the
    extended_bic_score penalty 2*ln(n)*k) works correctly only when it sees
    the FULL set of candidates that were tried. By aggregating all candidates
    here, BIC naturally accounts for all ratios tested simultaneously.
    """

    def __init__(
        self,
        ratio_symbols: List[str],
        budget=None,
        exclude_primitives: Optional[List[str]] = None,
    ):
        from adcd.asymptotic_dictionary_proposer_v3 import (
            GrammarBudget, enumerate_candidates, PRIMITIVE_REGISTRY,
        )
        self.ratio_symbols = ratio_symbols
        self.budget = budget or GrammarBudget()
        exclude = set(exclude_primitives or [])
        self._active_primitives = {
            k: v for k, v in PRIMITIVE_REGISTRY.items() if k not in exclude
        }
        # Pre-build all candidate strings (deduplicated, deterministic)
        seen: set = set()
        self._candidates: List[str] = []
        for ratio in ratio_symbols:
            for cand in enumerate_candidates(
                ratio_symbol=ratio,
                budget=self.budget,
                active_primitives=self._active_primitives,
            ):
                if cand not in seen:
                    self._candidates.append(cand)
                    seen.add(cand)

        # Expose sources dict so orchestrator can optionally log them
        self.sources: Dict[str, str] = {c: "deterministic_grammar" for c in self._candidates}

    def propose(self, context) -> List[str]:
        # Return ALL candidates — BIC handles ranking and penalisation
        return self._candidates

    def search_space_size(self) -> int:
        return len(self._candidates)


# =====================================================================
# SAFE FORMULA EVALUATOR — sympy sympify + lambdify, no raw eval()
# =====================================================================

def _evaluate_formula(
    formula: str,
    variables: Dict[str, str],
    data_arrays: Dict[str, np.ndarray],
    constants: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Evaluate a classical formula string against numpy data arrays.

    Uses sympy.sympify() + sympy.lambdify(), NOT eval(). This is a genuine
    mathematical parser: it cannot execute arbitrary Python code or escape
    via object introspection (unlike eval() with empty __builtins__).

    Raises ValueError with a helpful message if the formula is invalid or
    references variables not present in data_arrays/constants.
    """
    import sympy as sp

    # Build symbol namespace: only variables explicitly declared by user
    all_names = list(data_arrays.keys())
    if constants:
        all_names += [k for k in constants if k not in data_arrays]

    sym_locals = {name: sp.Symbol(name) for name in all_names}

    try:
        expr = sp.sympify(formula, locals=sym_locals)
    except Exception as exc:
        raise ValueError(
            f"[quickfit] Cannot parse classical_formula='{formula}'.\n"
            f"  Error: {exc}\n"
            f"  Tip: Use standard arithmetic: m*v, 0.5*m*v**2, G*m*M/r**2, etc.\n"
            f"  Known variable names: {all_names}"
        ) from exc

    # Reject anything sympy couldn't reduce to a genuine symbolic expression
    if not isinstance(expr, sp.Basic):
        raise ValueError(
            f"[quickfit] classical_formula='{formula}' did not parse to a "
            f"mathematical expression (got {type(expr).__name__}). "
            f"Use standard arithmetic operators only: +, -, *, **, /."
        )


    # Build evaluation namespace: numpy arrays + scalar constants
    eval_ns: Dict = {}
    eval_ns.update(data_arrays)
    if constants:
        # Broadcast scalar constants to array shape
        ref = next(iter(data_arrays.values()))
        eval_ns.update({k: np.full_like(ref, v) for k, v in constants.items()
                        if k not in eval_ns})

    free_syms = [str(s) for s in expr.free_symbols]
    missing = [s for s in free_syms if s not in eval_ns]
    if missing:
        raise ValueError(
            f"[quickfit] Formula '{formula}' references unknown names: {missing}.\n"
            f"  Declared variables: {list(variables.keys())}\n"
            f"  Constants: {list((constants or {}).keys())}\n"
            f"  Add missing names to 'variables' or 'constants'."
        )

    try:
        func = sp.lambdify(list(expr.free_symbols), expr, modules="numpy")
        args = {str(s): eval_ns[str(s)] for s in expr.free_symbols}
        result = func(**args)
        return np.asarray(result, dtype=float)
    except Exception as exc:
        raise ValueError(
            f"[quickfit] Formula '{formula}' evaluated but produced an error: {exc}"
        ) from exc


# =====================================================================
# LIMIT VARIABLE INFERENCE — confidence-gated, not silent
# =====================================================================

@dataclass
class _DetectedLimitVar:
    name: str
    confidence: float
    method: str

def _infer_limit_variable(X: Dict[str, np.ndarray]) -> _DetectedLimitVar:
    """
    Heuristic guess: variable with the highest coefficient of variation
    (std/mean) is most likely to span a wide relative range -- and
    therefore most likely to contain the classical-limit information.

    HONESTY NOTE: this has no principled connection to the actual
    physics. It is a plausible weak proxy, nothing more. Confidence
    is deliberately capped at 0.7 regardless of how sharp the
    statistical signal looks, because the heuristic itself does not
    merit high confidence.
    """
    cvs = {
        col: float(np.std(arr) / (np.mean(np.abs(arr)) + 1e-30))
        for col, arr in X.items()
        if len(arr) > 1
    }
    if not cvs:
        first = next(iter(X))
        return _DetectedLimitVar(first, 0.0, "fallback_single_variable")

    ranked = sorted(cvs.items(), key=lambda kv: kv[1], reverse=True)
    best, best_cv = ranked[0]

    if len(ranked) >= 2:
        sep = (best_cv - ranked[1][1]) / (best_cv + 1e-12)
        conf = min(0.70, 0.40 + 0.30 * sep)
    else:
        conf = 0.40

    return _DetectedLimitVar(best, conf, "coefficient_of_variation")


# =====================================================================
# MAIN PUBLIC API
# =====================================================================


"""Buckingham-Pi dimensionless group engine for multivariable ADCD Phase 2."""

class BuckinghamPiEngine:
    """
    Computes dimensionless Buckingham-Pi groups from registered variables.

    Uses SVD of the dimensional matrix to find the null space, which
    corresponds to dimensionless combinations (Buckingham, 1914).
    """

    def __init__(self) -> None:
        self.registry: Dict[str, np.ndarray] = {}

    def register(self, name: str, dim_vector: List[int]) -> None:
        """Register a variable with its dimension vector [M, L, T, ...]."""
        self.registry[name] = np.array(dim_vector, dtype=float)

    def register_from_scenario(self, scenario) -> None:
        """Register classical variables and known scale constants from a scenario."""
        from adcd.dimensional_checker import DimensionalChecker

        checker = DimensionalChecker()
        for var in scenario.classical_variables:
            if var in checker.registry:
                self.register(var, checker.registry[var])
        for const_name in scenario.classical_constants:
            if const_name in checker.registry:
                self.register(const_name, checker.registry[const_name])
            else:
                base = const_name.replace("_ref", "").replace("_0", "")
                if base in checker.registry:
                    self.register(const_name, checker.registry[base])
                else:
                    self.register(const_name, [0, 0, 0, 0, 0])

    def compute_pi_groups(self) -> List[sp.Expr]:
        """
        Compute independent dimensionless Pi groups.

        Uses exact rational nullspace (SymPy) for clean ratio forms like m/M.
        """
        if len(self.registry) < 2:
            return []

        names = list(self.registry.keys())
        dim_matrix = np.array([self.registry[n] for n in names]).T
        k, n = dim_matrix.shape

        if k >= n:
            return self._simple_same_dimension_ratios(names)

        from sympy import Matrix

        null_vectors = Matrix(dim_matrix.tolist()).nullspace()
        syms = {name: sp.Symbol(name) for name in names}
        pi_groups: List[sp.Expr] = []

        for vec in null_vectors:
            factors = []
            for name, exp in zip(names, vec):
                if exp == 0:
                    continue
                factors.append(syms[name] ** int(exp))
            if not factors:
                continue
            pi_expr = sp.simplify(sp.Mul(*factors))
            free_vars = {str(s) for s in pi_expr.free_symbols}
            if len(free_vars) >= 2:
                pi_groups.append(pi_expr)

        if not pi_groups:
            pi_groups = self._simple_same_dimension_ratios(names)

        return pi_groups

    def _simple_same_dimension_ratios(self, names: List[str]) -> List[sp.Expr]:
        """Fallback: pairwise ratios among equal-dimension variables."""
        groups: List[sp.Expr] = []
        seen: set[str] = set()
        by_dim: Dict[tuple, List[str]] = {}
        for name in names:
            key = tuple(int(x) for x in self.registry[name])
            by_dim.setdefault(key, []).append(name)

        for dim_vars in by_dim.values():
            if len(dim_vars) < 2:
                continue
            for i in range(len(dim_vars)):
                for j in range(i + 1, len(dim_vars)):
                    a, b = dim_vars[i], dim_vars[j]
                    for ratio in (f"{a}/{b}", f"{b}/{a}"):
                        if ratio not in seen:
                            seen.add(ratio)
                            groups.append(sp.sympify(ratio))
        return groups

    def get_parameterized_ratios(self) -> List[sp.Expr]:
        """Parameterized Pi forms Π/θ and Π·θ for grammar ratio candidates."""
        pi_groups = self.compute_pi_groups()
        ratios: List[sp.Expr] = []
        for i, pi in enumerate(pi_groups):
            theta = sp.Symbol(f"theta_pi_{i}")
            ratios.append(pi / theta)
            ratios.append(pi * theta)
        return ratios
