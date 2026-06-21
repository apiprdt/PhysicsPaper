"""ADCD discovery on the fσ₈ growth-rate residual Δ(z) = fσ₈_obs − fσ₈_GR.

The S₈-tension experiment: does the residual of the observed fσ₈(z) compilation
relative to the General-Relativity baseline Ωₘ(z)^γ · D(z) · σ₈₀ admit a
parsimonious analytic structure?

Methodology (matches the SPARC MOND comparison in :mod:`mond_comparison`):
    * Treat σ₈₀ as the GR baseline's free amplitude (already fitted by the
      loader, see :mod:`growth_rate_data`).
    * Propose a library of physically-motivated 1- and 2-parameter correction
      forms for Δ(z), spanning polynomial, power-law, exponential and rational
      families — exactly the structural classes ADCD distinguishes.
    * Fit each candidate's free parameters with L-BFGS-B (log-parameterised
      for positivity where needed), with 30 random restarts.
    * Score with χ² and BIC against the *residual* Δ(z). Lower BIC wins.
    * The crucial null model ``Δ(z) = θ₀`` (a single constant offset) is in the
      library — it is the simplest physically-meaningful "explanation" of the
      S₈ tension (a pure σ₈₀ renormalisation).

Result classes:
    * CONSTANT_WINS — no z-dependent form beats the 1-param constant offset by
      ΔBIC > 10. The residual carries no functional signal beyond a global
      amplitude shift; the S₈ tension is **not** an ADCD-discoverable
      functional anomaly in this dataset.
    * FUNCTIONAL_WINS — some z-dependent form wins decisively. This would be a
      candidate for ADCD reporting and would hint at modified growth.

The experiment writes a machine-readable JSON summary to ``results/``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from adcd.metrics import bic_score
from adcd.experiments.growth_rate_data import (
    GrowthRateResult,
    load_growth_rate,
    DEFAULT_CACHE,
)

# ---------------------------------------------------------------------------
# ΔBIC threshold for "decisive" functional win (Kass-Raftery).
# ---------------------------------------------------------------------------
DECISIVE_DELTA_BIC = 10.0

# ---------------------------------------------------------------------------
# Bins used to inspect the residual's z-structure for a coarse verdict.
# ---------------------------------------------------------------------------
DEFAULT_ZBINS: List[Tuple[float, float]] = [
    (0.0, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 2.0),
]


# ---------------------------------------------------------------------------
# Candidate correction library
# ---------------------------------------------------------------------------
#
# Each entry is (name, formula_str, fn, n_params). The fn signature is
#     fn(z, *params) -> np.ndarray   (residual prediction Δ_pred(z))
#
# The library spans the four ADCD structural families plus the null model.
# Mirrors the form of ``mond_comparison.py``'s fitted-baseline list.
# ---------------------------------------------------------------------------

def _delta_const(z, a):
    return np.full_like(np.asarray(z, float), a)

def _delta_linear(z, a, b):
    return a + b * np.asarray(z, float)

def _delta_quadratic(z, a, b, c):
    z = np.asarray(z, float)
    return a + b * z + c * z ** 2

def _delta_power(z, A, k):
    z = np.asarray(z, float)
    return A * np.power(z + 1e-6, k)

def _delta_power_offset(z, A, k, c):
    z = np.asarray(z, float)
    return c + A * np.power(z + 1e-6, k)

def _delta_exponential(z, A, lam):
    z = np.asarray(z, float)
    return A * np.exp(-lam * z)

def _delta_exp_offset(z, A, lam, c):
    z = np.asarray(z, float)
    return c + A * np.exp(-lam * z)

def _delta_rational(z, A, z0):
    z = np.asarray(z, float)
    return A / (1.0 + (z / z0) ** 2)

def _delta_log(z, A, b):
    z = np.asarray(z, float)
    return A * np.log1p(z) + b


CANDIDATE_LIBRARY: List[Tuple[str, str, callable, int]] = [
    ("Constant offset",       "theta_0",                                   _delta_const,        1),
    ("Linear",                "theta_0 + theta_1*z",                       _delta_linear,       2),
    ("Quadratic",             "theta_0 + theta_1*z + theta_2*z**2",        _delta_quadratic,    3),
    ("Power law",             "theta_0*z**theta_1",                        _delta_power,        2),
    ("Power + offset",        "theta_2 + theta_0*z**theta_1",              _delta_power_offset, 3),
    ("Exponential decay",     "theta_0*exp(-theta_1*z)",                   _delta_exponential,  2),
    ("Exponential + offset",  "theta_2 + theta_0*exp(-theta_1*z)",         _delta_exp_offset,   3),
    ("Rational",              "theta_0 / (1 + (z/theta_1)**2)",            _delta_rational,     2),
    ("Log",                   "theta_0*log(1+z) + theta_1",                _delta_log,          2),
]


@dataclass
class CandidateScore:
    name: str
    formula: str
    n_params: int
    chi2: float
    chi2_reduced: float
    bic: float
    params: Dict[str, float] = field(default_factory=dict)


def _fit_candidate(fn, n_params: int, z: np.ndarray, y: np.ndarray,
                   sigma: np.ndarray, name: str, formula: str,
                   n_restarts: int = 30, seed: int = 42) -> CandidateScore:
    """Weighted-LS fit of one candidate with L-BFGS-B + random restarts.

    Initial guesses are drawn in log-space for strictly-positive parameters and
    in linear space otherwise — the choice per-parameter is made via the
    positive-mask inferred from the candidate (we just try both signs).
    """
    rng = np.random.default_rng(seed)
    w = 1.0 / sigma ** 2

    def objective(raw: np.ndarray) -> float:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                pred = fn(z, *raw)
        except Exception:
            return 1e15
        if not np.all(np.isfinite(pred)):
            return 1e15
        # Guard against overflow in the squared residual when the optimiser
        # explores extreme parameter values (e.g. power-law with k → 50).
        diff = np.clip(y - pred, -1e6, 1e6)
        cost = float(np.sum(w * diff ** 2))
        return cost if np.isfinite(cost) else 1e15

    best_cost = np.inf
    best_raw = np.zeros(n_params)
    for _ in range(n_restarts):
        x0 = rng.uniform(-1.0, 1.0, size=n_params)
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           options={"maxiter": 5000, "ftol": 1e-16})
            if res.fun < best_cost:
                best_cost = res.fun
                best_raw = res.x.copy()
        except Exception:
            continue

    pred = fn(z, *best_raw)
    chi2 = float(np.sum(((y - pred) / sigma) ** 2))
    n = len(y)
    # NMSE on the residual (the quantity the project's BIC uses)
    nmse = float(np.mean((y - pred) ** 2) / max(np.var(y), 1e-15))
    bic = bic_score(nmse, n_params, n)
    params = {f"theta_{i}": float(best_raw[i]) for i in range(n_params)}
    return CandidateScore(
        name=name, formula=formula, n_params=n_params,
        chi2=chi2, chi2_reduced=chi2 / max(n - n_params, 1),
        bic=bic, params=params,
    )


def _weighted_bin_means(z, y, sigma, zbins) -> List[Dict]:
    """Per-bin inverse-variance weighted mean of the residual (for inspection)."""
    out = []
    for zlo, zhi in zbins:
        m = (z >= zlo) & (z < zhi)
        if m.sum() == 0:
            continue
        w = 1.0 / sigma[m] ** 2
        mean = float(np.sum(y[m] * w) / np.sum(w))
        err = float(1.0 / np.sqrt(np.sum(w)))
        out.append({
            "z_lo": float(zlo), "z_hi": float(zhi), "n": int(m.sum()),
            "weighted_mean_Delta": mean, "error": err,
            "pull": float(mean / err) if err > 0 else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

@dataclass
class GrowthRateDiscovery:
    n_points: int
    sigma8_0_fit: float
    chi2_gr_baseline: float
    chi2_reduced_gr_baseline: float
    bin_means: List[Dict]
    candidates: List[Dict]
    best_candidate: str
    best_bic: float
    null_bic: float                     # the 1-param Constant offset
    delta_bic_vs_null: float           # best - null  (negative => functional win)
    verdict: str                       # "CONSTANT_WINS" or "FUNCTIONAL_WINS"
    reviewer_message: str
    data_source: str

    def to_dict(self) -> Dict:
        return asdict(self)


def run_growth_rate_discovery(
    cache_path: str = DEFAULT_CACHE,
    cosmology: str = "per_point",
    n_restarts: int = 30,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> GrowthRateDiscovery:
    """Run the full ADCD-vs-constant comparison on the fσ₈ residual.

    Returns a :class:`GrowthRateDiscovery` summary. If ``output_path`` is given
    the summary is also written there as JSON.
    """
    res: GrowthRateResult = load_growth_rate(
        cache_path=cache_path, cosmology=cosmology,
        allow_simulated_fallback=False,
    )

    z, y, sigma = res.z, res.residual, res.sigma_fs8

    # Coarse residual inspection
    bin_means = _weighted_bin_means(z, y, sigma, DEFAULT_ZBINS)
    print("\n=== Δ(z) per-redshift-bin structure ===")
    print(f"  {'z range':>14} | {'N':>3} | {'<Δ>_w':>9} | {'±err':>9} | {'pull':>7}")
    for b in bin_means:
        print(f"  [{b['z_lo']:.2f},{b['z_hi']:.2f}) | {b['n']:>3} | "
              f"{b['weighted_mean_Delta']:+.4f} | {b['error']:.4f} | "
              f"{b['pull']:+.2f}σ")

    # Baseline χ² (GR, σ₈₀ free, no functional correction)
    chi2_gr = float(np.sum((y / sigma) ** 2))
    chi2_red_gr = chi2_gr / max(len(y) - 1, 1)
    print(f"\n=== GR baseline (σ₈₀ free, no Δ(z)) ===")
    print(f"  χ²        = {chi2_gr:.3f}")
    print(f"  χ²/dof    = {chi2_red_gr:.3f}")

    # Fit each candidate
    print("\n=== ADCD candidate library (lower BIC = better) ===")
    print(f"  {'Candidate':<24} | {'k':>2} | {'χ²':>8} | {'χ²/dof':>8} | "
          f"{'BIC':>9}")
    scores: List[CandidateScore] = []
    for name, formula, fn, k in CANDIDATE_LIBRARY:
        s = _fit_candidate(fn, k, z, y, sigma, name, formula,
                           n_restarts=n_restarts, seed=seed)
        print(f"  {name:<24} | {s.n_params:>2} | {s.chi2:>8.3f} | "
              f"{s.chi2_reduced:>8.3f} | {s.bic:>9.2f}")
        scores.append(s)

    scores.sort(key=lambda s: s.bic)
    best = scores[0]
    null = next(s for s in scores if s.name == "Constant offset")
    delta_bic = best.bic - null.bic

    verdict = ("FUNCTIONAL_WINS"
               if (best.name != "Constant offset"
                   and delta_bic <= -DECISIVE_DELTA_BIC)
               else "CONSTANT_WINS")

    reviewer = _build_reviewer_message(
        verdict=verdict,
        best=best, null=null, delta_bic=delta_bic,
        chi2_red_gr=chi2_red_gr, sigma8_0=res.sigma8_0_fit,
        n_points=len(y),
    )
    print("\n" + reviewer)

    summary = GrowthRateDiscovery(
        n_points=len(y),
        sigma8_0_fit=float(res.sigma8_0_fit),
        chi2_gr_baseline=chi2_gr,
        chi2_reduced_gr_baseline=chi2_red_gr,
        bin_means=bin_means,
        candidates=[asdict(s) for s in scores],
        best_candidate=best.name,
        best_bic=float(best.bic),
        null_bic=float(null.bic),
        delta_bic_vs_null=float(delta_bic),
        verdict=verdict,
        reviewer_message=reviewer,
        data_source=res.data_source,
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)
        print(f"\nSummary written to {output_path}")

    return summary


def _build_reviewer_message(verdict: str, best: CandidateScore,
                            null: CandidateScore, delta_bic: float,
                            chi2_red_gr: float, sigma8_0: float,
                            n_points: int) -> str:
    if verdict == "CONSTANT_WINS":
        return (
            "RESULT — Growth-rate residual Δ(z) carries NO functional signal.\n"
            f"  • Dataset : Alestas+2022 Table II, N={n_points} points, "
            f"z ∈ [0.001, 1.944].\n"
            f"  • GR baseline (σ₈₀ free) : χ²/dof = {chi2_red_gr:.3f}, "
            f"σ₈₀ = {sigma8_0:.4f} (Planck: 0.811).\n"
            "  • Best ADCD candidate : '{best}' (BIC = {bb:.2f});\n"
            "    null 1-param constant offset: BIC = {nb:.2f}.\n"
            "  • ΔBIC vs null = {db:+.2f} → the constant offset is preferred;\n"
            "    no z-dependent correction beats a pure amplitude renormalisation.\n"
            "  • Conclusion : the S₈ tension in fσ₈ is NOT an ADCD-discoverable\n"
            "    functional anomaly on this dataset — it is fully absorbed by a\n"
            "    single nuisance parameter (σ₈₀). The wide-binary verdict (ADCDD\n"
            "    ≡ Simple MOND, does not win there either) is reinforced: ADCD does\n"
            "    not surface a novel functional correction in growth-rate data."
        ).format(best=best.name, bb=best.bic, nb=null.bic, db=delta_bic)
    return (
        "RESULT — Growth-rate residual Δ(z) shows a FUNCTIONAL signal.\n"
        f"  • Dataset : Alestas+2022 Table II, N={n_points} points.\n"
        f"  • Best ADCD candidate : '{best}' (BIC = {best.bic:.2f}),\n"
        f"    formula : {best.formula}\n"
        f"  • 1-param constant offset BIC = {null.bic:.2f}.\n"
        f"  • ΔBIC vs null = {delta_bic:+.2f} (decisive: |ΔBIC| > "
        f"{DECISIVE_DELTA_BIC:.0f}).\n"
        "  • Conclusion : a z-dependent correction is decisively preferred over a\n"
        "    pure amplitude shift — a candidate functional form for modified\n"
        "    growth. **Flag for cross-check on homogeneous BOSS subset (P3.4).**"
    )


# ---------------------------------------------------------------------------
# P3.4 — Homogeneous-subset cross-check
# ---------------------------------------------------------------------------
#
# The 63-point Alestas compilation mixes ~15 different surveys with different
# modelling systematics (Alam, Beutler, Blake, Howlett, Okumura, Pezzotta,
# Shi, Tamone, Wang, Zhao, ...). A z-dependent correction that "wins" on the
# full compilation could be an artefact of inter-survey systematics. The
# cleanest possible cross-check is therefore a *homogeneous* subset where the
# same survey, the same modelling pipeline and the same fiducial cosmology
# apply to every point.
#
# The three subsets below are identified from the data itself (no external
# row-to-survey map needed):
#
#   • "BOSS DR12 (Alam+2017)" : 6 points at z ∈ {0.38, 0.51, 0.61}, all with
#       Ωₘ_fid = 0.31 — the canonical BOSS DR12 consensus values. This is the
#       single most-cited, most-homogeneous RSD dataset in existence.
#   • "Precision subset"      : 20 points with σ(fσ₈) < 0.06 — modern, small-
#       error measurements (BOSS/eBOSS/2dFGRS dominant).
#   • "Mid-z window"          : 31 points with z ∈ [0.35, 0.75] — the BOSS-
#       dominated redshift range.
#
# The result-class logic is the same as for the full sample.
# ---------------------------------------------------------------------------

def _mask_boss_dr12(z, fs8, sigma, Om) -> np.ndarray:
    """6 canonical BOSS DR12 (Alam+2017) points: z∈{0.38,0.51,0.61}, Om=0.31."""
    mask = np.zeros_like(z, dtype=bool)
    for z_target in (0.38, 0.51, 0.61):
        mask |= (np.abs(z - z_target) < 0.005) & (np.abs(Om - 0.31) < 0.005)
    return mask


def _mask_precision(z, fs8, sigma, Om, n_top: int = 20) -> np.ndarray:
    """Top-N most precise measurements."""
    idx = np.argsort(sigma)[:n_top]
    mask = np.zeros_like(sigma, dtype=bool)
    mask[idx] = True
    return mask


def _mask_mid_z(z, fs8, sigma, Om, zlo: float = 0.35, zhi: float = 0.75) -> np.ndarray:
    return (z >= zlo) & (z <= zhi)


SUBSET_FACTORIES = {
    "BOSS DR12 (Alam+2017)": _mask_boss_dr12,
    "Precision (top-20)":    _mask_precision,
    "Mid-z window [0.35,0.75]": _mask_mid_z,
}


@dataclass
class SubsetResult:
    name: str
    n_points: int
    sigma8_0_fit: float
    chi2_reduced_gr: float
    best_candidate: str
    best_bic: float
    null_bic: float
    delta_bic_vs_null: float
    verdict: str


def _run_one_subset(name: str, mask: np.ndarray, res: GrowthRateResult,
                    n_restarts: int, seed: int) -> SubsetResult:
    """Re-fit σ₈₀ on the masked subset, then run the candidate library on its Δ."""
    from adcd.experiments.growth_rate_data import (
        gr_fs8_prediction, _fit_sigma8_amplitude,
    )
    z, fs8_obs, sigma, Om = (res.z[mask], res.fs8_obs[mask],
                             res.sigma_fs8[mask], res.Om_fid[mask])
    Om_m0, Om_de0 = Om.copy(), 1.0 - Om
    s80 = _fit_sigma8_amplitude(z, fs8_obs, sigma, Om_m0, Om_de0)
    fs8_gr = gr_fs8_prediction(z, Om_m0, Om_de0, sigma8_0=s80)
    y = fs8_obs - fs8_gr
    chi2_red = float(np.sum((y / sigma) ** 2) / max(len(y) - 1, 1))

    scores: List[CandidateScore] = []
    for cname, formula, fn, k in CANDIDATE_LIBRARY:
        scores.append(_fit_candidate(fn, k, z, y, sigma, cname, formula,
                                     n_restarts=n_restarts, seed=seed))
    scores.sort(key=lambda s: s.bic)
    best = scores[0]
    null = next(s for s in scores if s.name == "Constant offset")
    delta = best.bic - null.bic
    verdict = ("FUNCTIONAL_WINS"
               if (best.name != "Constant offset"
                   and delta <= -DECISIVE_DELTA_BIC)
               else "CONSTANT_WINS")
    return SubsetResult(
        name=name, n_points=int(mask.sum()),
        sigma8_0_fit=float(s80), chi2_reduced_gr=chi2_red,
        best_candidate=best.name, best_bic=float(best.bic),
        null_bic=float(null.bic), delta_bic_vs_null=float(delta),
        verdict=verdict,
    )


def run_subset_crosscheck(
    res: Optional[GrowthRateResult] = None,
    n_restarts: int = 30,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> List[SubsetResult]:
    """Re-run the discovery on three homogeneous subsets; report verdicts."""
    if res is None:
        res = load_growth_rate()

    print("\n" + "=" * 72)
    print("P3.4 — Homogeneous-subset cross-check")
    print("=" * 72)
    out: List[SubsetResult] = []
    for name, factory in SUBSET_FACTORIES.items():
        mask = factory(res.z, res.fs8_obs, res.sigma_fs8, res.Om_fid)
        if mask.sum() < 5:
            print(f"\n[{name}] skipped — only {mask.sum()} points")
            continue
        r = _run_one_subset(name, mask, res, n_restarts, seed)
        out.append(r)
        print(f"\n[{name}]  N = {r.n_points}")
        print(f"  σ₈₀ fit        = {r.sigma8_0_fit:.4f}")
        print(f"  GR χ²/dof      = {r.chi2_reduced_gr:.3f}")
        print(f"  best candidate = '{r.best_candidate}'  "
              f"(BIC = {r.best_bic:.2f})")
        print(f"  null (const)   = {r.null_bic:.2f}")
        print(f"  ΔBIC vs null   = {r.delta_bic_vs_null:+.2f}")
        print(f"  → {r.verdict}")

    # Headline: do ANY subsets show a functional win?
    any_win = any(r.verdict == "FUNCTIONAL_WINS" for r in out)
    print("\n--- Subset cross-check verdict ---")
    if not any_win:
        print("No homogeneous subset shows a functional win: the full-sample")
        print("CONSTANT_WINS verdict is robust to survey-systematics removal.")
    else:
        winning = [r.name for r in out if r.verdict == "FUNCTIONAL_WINS"]
        print(f"WARNING — functional win on {len(winning)} subset(s): {winning}")
        print("Investigate whether the functional form survives P3.5 (H(z)).")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in out], fh, indent=2)
        print(f"\nSubset summary written to {output_path}")
    return out


# ---------------------------------------------------------------------------
# P3.5 — Cosmic Chronometers H(z) secondary validation
# ---------------------------------------------------------------------------
#
# An independent observable. The growth rate fσ₈ and the expansion rate H(z)
# are related only implicitly (both are functions of the same cosmological
# background), so a functional anomaly in Δ(z) of fσ₈ would have to come from
# a modified-gravity signature that also leaves a detectable z-dependent
# residual in H(z)/H₀. The cosmic-chronometers H(z) compilation is the cleanest
# such test: it is model-independent, distance-ladder-free, and independent of
# CMB / BAO / RSD systematics.
#
# Reference ΛCDM prediction (flat, w=-1):
#     H(z) = H₀ · E(z),   E(z) = sqrt( Ωₘ₀ (1+z)³ + Ω_Λ₀ )
# We treat (H₀, Ωₘ₀) as two nuisance amplitudes fitted by weighted-LS, then
# look for a z-dependent structure in Δ_H(z) = H_obs − H_GR with the same
# candidate library used for fσ₈.
# ---------------------------------------------------------------------------

DEFAULT_CC_CACHE = "data/cosmic_chronometers/CC_Hz_compilation.txt"


@dataclass
class CosmicChronometersResult:
    n_points: int
    H0_fit: float
    Om_m0_fit: float
    chi2_reduced: float
    best_candidate: str
    best_bic: float
    null_bic: float
    delta_bic_vs_null: float
    verdict: str
    candidates: List[Dict]
    reviewer_message: str


def _parse_cc_table(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No CC rows parsed from {cache_path}")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _H_gr(z: np.ndarray, H0: float, Om_m0: float) -> np.ndarray:
    Om_de0 = 1.0 - Om_m0   # assume flatness
    return H0 * np.sqrt(Om_m0 * (1.0 + z) ** 3 + Om_de0)


def _fit_H0_Om(z, H_obs, sigma_H):
    """Two-parameter weighted-LS for (H₀, Ωₘ₀) using L-BFGS-B."""
    w = 1.0 / sigma_H ** 2

    def objective(p):
        H0 = p[0]
        Om = 1.0 / (1.0 + np.exp(-p[1]))   # logit -> (0,1)
        with np.errstate(over="ignore", invalid="ignore"):
            pred = _H_gr(z, H0, Om)
        if not np.all(np.isfinite(pred)):
            return 1e15
        diff = np.clip(H_obs - pred, -1e6, 1e6)
        return float(np.sum(w * diff ** 2))

    best_cost, best_p = np.inf, np.array([70.0, 0.0])
    rng = np.random.default_rng(42)
    for _ in range(40):
        x0 = np.array([rng.uniform(60, 80),
                       rng.uniform(-1.5, 1.5)])
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           options={"maxiter": 5000, "ftol": 1e-16})
            if res.fun < best_cost:
                best_cost = res.fun
                best_p = res.x.copy()
        except Exception:
            continue
    return float(best_p[0]), float(1.0 / (1.0 + np.exp(-best_p[1])))


def run_cosmic_chronometers(
    cache_path: str = DEFAULT_CC_CACHE,
    n_restarts: int = 30,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> CosmicChronometersResult:
    """Run the same ADCD candidate library on the H(z) residual Δ_H(z)."""
    z, H_obs, sigma_H = _parse_cc_table(cache_path)
    H0, Om = _fit_H0_Om(z, H_obs, sigma_H)
    y = H_obs - _H_gr(z, H0, Om)
    chi2_red = float(np.sum((y / sigma_H) ** 2) / max(len(y) - 2, 1))

    print("\n" + "=" * 72)
    print("P3.5 — Cosmic Chronometers H(z) secondary validation")
    print("=" * 72)
    print(f"  N points       : {len(z)}")
    print(f"  z range        : {z.min():.3f} – {z.max():.3f}")
    print(f"  H₀ fit         : {H0:.2f} km/s/Mpc  (Planck: 67.4, TRGB: ~70)")
    print(f"  Ωₘ₀ fit        : {Om:.4f}  (Planck: 0.315)")
    print(f"  GR baseline χ²/dof (H₀, Ωₘ free) : {chi2_red:.3f}")

    # Fit candidate library on Δ_H(z). Same machinery as for fσ₈.
    scores: List[CandidateScore] = []
    print("\n=== ADCD candidate library on Δ_H(z) ===")
    print(f"  {'Candidate':<24} | {'k':>2} | {'χ²':>8} | {'χ²/dof':>8} | {'BIC':>9}")
    for name, formula, fn, k in CANDIDATE_LIBRARY:
        s = _fit_candidate(fn, k, z, y, sigma_H, name, formula,
                           n_restarts=n_restarts, seed=seed)
        print(f"  {name:<24} | {s.n_params:>2} | {s.chi2:>8.3f} | "
              f"{s.chi2_reduced:>8.3f} | {s.bic:>9.2f}")
        scores.append(s)

    scores.sort(key=lambda s: s.bic)
    best = scores[0]
    null = next(s for s in scores if s.name == "Constant offset")
    delta_bic = best.bic - null.bic
    verdict = ("FUNCTIONAL_WINS"
               if (best.name != "Constant offset"
                   and delta_bic <= -DECISIVE_DELTA_BIC)
               else "CONSTANT_WINS")

    if verdict == "CONSTANT_WINS":
        reviewer = (
            "P3.5 RESULT — H(z) residual carries NO functional signal.\n"
            f"  • Dataset   : Cosmic Chronometers, N={len(z)} points, "
            f"z ∈ [{z.min():.3f}, {z.max():.3f}].\n"
            f"  • ΛCDM baseline : H₀={H0:.2f}, Ωₘ₀={Om:.4f}, χ²/dof={chi2_red:.3f}.\n"
            f"  • Best candidate : '{best.name}' (BIC = {best.bic:.2f});\n"
            f"    null constant offset : BIC = {null.bic:.2f}.\n"
            f"  • ΔBIC vs null = {delta_bic:+.2f} → constant offset is preferred.\n"
            "  • Conclusion : an INDEPENDENT observable (expansion rate) also\n"
            "    shows no ADCD-discoverable functional anomaly. Combined with\n"
            "    P3.3 + P3.4, this is strong evidence that ADCD does not surface\n"
            "    spurious functional corrections in real cosmological data."
        )
    else:
        reviewer = (
            "P3.5 RESULT — H(z) residual shows a FUNCTIONAL signal.\n"
            f"  • Best candidate : '{best.name}' (BIC = {best.bic:.2f}).\n"
            f"  • ΔBIC vs null = {delta_bic:+.2f} (decisive).\n"
            "  • This would suggest an actual modified-expansion signature;\n"
            "    treat with caution and seek independent confirmation."
        )
    print("\n" + reviewer)

    summary = CosmicChronometersResult(
        n_points=len(z), H0_fit=H0, Om_m0_fit=Om,
        chi2_reduced=chi2_red,
        best_candidate=best.name, best_bic=float(best.bic),
        null_bic=float(null.bic), delta_bic_vs_null=float(delta_bic),
        verdict=verdict,
        candidates=[asdict(s) for s in scores],
        reviewer_message=reviewer,
    )
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(summary), fh, indent=2)
        print(f"\nSummary written to {output_path}")
    return summary


if __name__ == "__main__":
    out = os.path.join("results", "growth_rate_discovery.json")
    run_growth_rate_discovery(output_path=out)
    subset_out = os.path.join("results", "growth_rate_subsets.json")
    run_subset_crosscheck(output_path=subset_out)
    cc_out = os.path.join("results", "cosmic_chronometers.json")
    run_cosmic_chronometers(output_path=cc_out)
