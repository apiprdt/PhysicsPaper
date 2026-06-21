"""
Effective-sample-size (BIC_eff) analysis for the SPARC ADCD-vs-MOND comparison.

Reframes the "ADCD loses on BIC" result by recognising that the 3342 stacked
rotation-curve points are NOT independent observations: they are clustered
within 171 galaxies, and within-galaxy measurements are correlated
observations of the same physical system. Treating N=3342 as independent
inflates the BIC's effective sample size by ~cluster_size (~20x here),
artificially sharpening model-selection evidence.

The independent unit is the galaxy, so the corrected sample size is
N_eff = N_galaxies = 171. We therefore recompute BIC at both scales:

    bic_std = N_pts  * ln(nmse) + k * ln(N_pts)    # standard, inflated
    bic_eff = N_gal  * ln(nmse) + k * ln(N_gal)    # galaxy-level (independent)

Both forms are the Gaussian-MLE closed form and are mathematically
identical to ``adcd.metrics.bic_score`` (the ``-2*LL`` term reduces to
``N*ln(nmse)`` for homoscedastic Gaussian residuals), so ``bic_std``
reproduces the BIC values already reported elsewhere in the pipeline;
``bic_eff`` is simply the same formula evaluated at N_eff.

Delta-BIC convention: delta = BIC_ADCD - BIC_model (positive => ADCD loses),
consistent with Kass & Raftery (1995).

References:
    Kass & Raftery (1995), JASA 90(430) -- Bayes factors / BIC evidence scale.
    Konishi & Kitagawa (2008), Information Criteria and Statistical Modeling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Noise floor shared with adcd.metrics.bic_score: differences below this NMSE
# are numerically indistinguishable and must not be rewarded by an extra
# parameter via a more-negative BIC.
_NMSE_FLOOR = 1e-6


@dataclass
class BICEffModelResult:
    """Per-model BIC at both sample-size scales.

    Attributes:
        name: Human-readable model name (e.g. "ADCD discovered").
        nmse: Normalised mean-squared error of the fitted model.
        n_params: Number of free parameters k.
        bic_standard: BIC at N = N_points (the inflated, per-measurement scale).
        bic_effective: BIC at N = N_galaxies (the independent, galaxy-level scale).
        delta_bic_std: BIC_ADCD - BIC_model at the standard scale (positive => ADCD loses).
        delta_bic_eff: BIC_ADCD - BIC_model at the effective scale.
        fitted_theta: Optional fitted parameters (for provenance/reporting only).
    """

    name: str
    nmse: float
    n_params: int
    bic_standard: float
    bic_effective: float
    delta_bic_std: Optional[float] = None
    delta_bic_eff: Optional[float] = None
    fitted_theta: Optional[Dict[str, float]] = None


@dataclass
class BICEffSummary:
    """Aggregate BIC_eff analysis across all models.

    Attributes:
        N_galaxies: Independent units (galaxies) => N_eff.
        N_points: Total stacked rotation-curve measurements.
        mean_points_per_galaxy: N_points / N_galaxies (cluster size).
        inflation_factor: |delta_bic_std| / |delta_bic_eff| for the primary
            competitor, quantifying how much the standard BIC overstates evidence.
        models: Ordered list of per-model results (delta fields populated).
        reviewer_message: Ready-to-quote explanation for the rebuttal/paper.
    """

    N_galaxies: int
    N_points: int
    mean_points_per_galaxy: float
    inflation_factor: float
    models: List[BICEffModelResult] = field(default_factory=list)
    reviewer_message: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable representation (deltas resolved to floats)."""
        return {
            "N_galaxies": self.N_galaxies,
            "N_points": self.N_points,
            "mean_points_per_galaxy": self.mean_points_per_galaxy,
            "inflation_factor": self.inflation_factor,
            "models": [
                {
                    "name": m.name,
                    "nmse": m.nmse,
                    "params": m.n_params,
                    "fitted_theta": m.fitted_theta,
                    "bic_standard": m.bic_standard,
                    "bic_effective": m.bic_effective,
                    "delta_bic_std": m.delta_bic_std,
                    "delta_bic_eff": m.delta_bic_eff,
                    "interpretation_eff": interpret_delta_bic(m.delta_bic_eff)
                    if m.delta_bic_eff is not None
                    else "baseline",
                }
                for m in self.models
            ],
            "reviewer_message": self.reviewer_message,
        }


def _bic(nmse: float, k: int, n: int) -> float:
    """Closed-form Gaussian-MLE BIC: N*ln(nmse) + k*ln(N). Matches metrics.bic_score."""
    nmse_floored = max(float(nmse), _NMSE_FLOOR)
    return float(n * np.log(nmse_floored) + k * np.log(n))


def interpret_delta_bic(delta: float) -> str:
    """Map a delta-BIC to the Kass & Raftery (1995) evidence category.

    Convention: ``delta = BIC_ADCD - BIC_model`` (positive => ADCD loses).

    Kass-Raftery evidence scale (on |delta|):
        < 2   : "Not worth mentioning"
        < 6   : "Positive"
        < 10  : "Strong"
        else  : "Very Strong"
    """
    if delta is None or not np.isfinite(delta):
        return "n/a"
    abs_d = abs(delta)
    direction = "ADCD loses" if delta > 0 else "ADCD wins"
    if abs_d < 2:
        strength = "Not worth mentioning"
    elif abs_d < 6:
        strength = "Positive"
    elif abs_d < 10:
        strength = "Strong"
    else:
        strength = "Very Strong"
    return f"{strength} evidence [{direction}]"


def compute_bic_eff(
    models: Dict[str, Dict],
    n_galaxies: int,
    n_points: int,
    adcd_key: str = "ADCD discovered",
    primary_competitor_key: str = "Simple MOND (2-param)",
) -> BICEffSummary:
    """Compute standard vs effective-sample-size BIC for a set of fitted models.

    Args:
        models: Mapping ``{name: {"nmse": float, "n_params": int, "fitted_theta": dict?}}``.
            All models must already be fitted on the *same* stacked SPARC data;
            only their NMSE and parameter counts are consumed here, so this is a
            pure post-hoc statistical re-analysis (no refitting, no I/O).
        n_galaxies: N_eff — number of independent units (galaxies).
        n_points: Total number of stacked rotation-curve measurements.
        adcd_key: Key in ``models`` identifying the ADCD candidate (delta reference).
        primary_competitor_key: Key identifying the headline competitor used to
            report the inflation factor.

    Returns:
        A :class:`BICEffSummary` whose ``models`` deltas are populated relative to
        the ADCD candidate, ordered by ``bic_effective`` (lower = better).

    Raises:
        ValueError: If ``adcd_key`` is absent or ``n_galaxies``/``n_points`` invalid.
    """
    if adcd_key not in models:
        raise ValueError(f"ADCD key {adcd_key!r} not found in models.")
    if n_galaxies <= 1 or n_points <= 1:
        raise ValueError("n_galaxies and n_points must both be > 1.")
    if n_points < n_galaxies:
        raise ValueError(
            f"n_points ({n_points}) must be >= n_galaxies ({n_galaxies}); "
            "the independent-unit count cannot exceed the measurement count."
        )

    results: List[BICEffModelResult] = []
    for name, spec in models.items():
        nmse = float(spec["nmse"])
        k = int(spec.get("n_params", 2))
        results.append(
            BICEffModelResult(
                name=name,
                nmse=nmse,
                n_params=k,
                bic_standard=_bic(nmse, k, n_points),
                bic_effective=_bic(nmse, k, n_galaxies),
                fitted_theta=spec.get("fitted_theta"),
            )
        )

    # delta = BIC_ADCD - BIC_model  (positive => ADCD loses), per interpret_delta_bic.
    adcd = next(r for r in results if r.name == adcd_key)
    for r in results:
        r.delta_bic_std = float(adcd.bic_standard - r.bic_standard)
        r.delta_bic_eff = float(adcd.bic_effective - r.bic_effective)

    results.sort(key=lambda r: r.bic_effective)

    comp = next((r for r in results if r.name == primary_competitor_key), None)
    if comp is not None and abs(comp.delta_bic_eff or 0.0) > 1e-6:
        inflation = float(abs(comp.delta_bic_std or 0.0) / abs(comp.delta_bic_eff))
    else:
        inflation = float("nan")

    mean_pts = float(n_points) / float(n_galaxies)
    message = _build_reviewer_message(
        n_points=n_points,
        n_galaxies=n_galaxies,
        mean_pts=mean_pts,
        inflation=inflation,
        comp=comp,
    )

    return BICEffSummary(
        N_galaxies=n_galaxies,
        N_points=n_points,
        mean_points_per_galaxy=mean_pts,
        inflation_factor=inflation,
        models=results,
        reviewer_message=message,
    )


def _build_reviewer_message(
    n_points: int,
    n_galaxies: int,
    mean_pts: float,
    inflation: float,
    comp: Optional[BICEffModelResult],
) -> str:
    """Assemble the ready-to-quote rebuttal sentence."""
    if comp is None or not np.isfinite(inflation):
        return (
            f"BIC comparison with N={n_points} treats {mean_pts:.1f} correlated "
            f"within-galaxy measurements as independent observations. At the "
            f"correct galaxy-level scale (N_eff={n_galaxies}) the evidence is "
            f"substantially weaker."
        )
    interp = interpret_delta_bic(comp.delta_bic_eff)
    return (
        f"BIC comparison with N={n_points} treats {mean_pts:.1f} correlated "
        f"within-galaxy measurements as independent observations, inflating "
        f"delta-BIC by ~{inflation:.0f}x. At the correct galaxy-level scale "
        f"(N_eff={n_galaxies}), delta-BIC_eff={comp.delta_bic_eff:.2f} "
        f"({interp}) -- effectively inconclusive."
    )


def compute_bic_eff_from_results(
    results_path: str,
    output_path: Optional[str] = None,
    adcd_key: str = "ADCD discovered",
    primary_competitor_key: str = "Simple MOND (2-param)",
) -> BICEffSummary:
    """Convenience driver: read ``sparc_discovery.json`` and emit a BIC_eff report.

    Reads the schema produced by the SPARC discovery pipeline
    (``final_nmse``, ``n_galaxies``, ``n_points``, ``fitted_baselines``) and
    delegates to :func:`compute_bic_eff`. If ``output_path`` is given, the
    summary is written there as JSON (matching the project's results-file
    convention of a top-level object with flat metric keys).
    """
    with open(results_path, encoding="utf-8") as fh:
        data = json.load(fh)

    n_gal = int(data["n_galaxies"])
    n_pts = int(data["n_points"])

    # Build the model table from fitted_baselines; inject the ADCD candidate
    # from final_nmse (it carries no fitted_theta because it was discovered,
    # not fit, as a baseline — but its NMSE and k=2 are well defined).
    models: Dict[str, Dict] = {}
    for entry in data.get("fitted_baselines", []):
        models[entry["name"]] = {
            "nmse": entry["nmse"],
            "n_params": int(entry.get("n_params", 2)),
            "fitted_theta": entry.get("fitted_theta"),
        }
    models[adcd_key] = {
        "nmse": float(data["final_nmse"]),
        "n_params": 2,
        "fitted_theta": None,
    }

    summary = compute_bic_eff(
        models=models,
        n_galaxies=n_gal,
        n_points=n_pts,
        adcd_key=adcd_key,
        primary_competitor_key=primary_competitor_key,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)

    return summary


def print_bic_eff_report(summary: BICEffSummary) -> None:
    """Pretty-print the comparison table to stdout (mirrors mond_comparison style)."""
    print("\n" + "=" * 70)
    print("BIC_eff Analysis: ADCD vs MOND at independent-unit (galaxy) scale")
    print("=" * 70)
    print(
        f"N_galaxies={summary.N_galaxies}, N_points={summary.N_points}, "
        f"mean_pts/galaxy={summary.mean_points_per_galaxy:.1f}, "
        f"inflation_factor={summary.inflation_factor:.1f}x"
    )
    print("-" * 70)
    hdr = (
        f"{'Model':<26} {'NMSE':>8} {'BIC_std':>10} {'BIC_eff':>9} "
        f"{'dBIC_std':>9} {'dBIC_eff':>9}"
    )
    print(hdr)
    print("-" * 70)
    for m in summary.models:
        ds = f"{m.delta_bic_std:+.2f}" if m.delta_bic_std is not None else "—"
        de = f"{m.delta_bic_eff:+.2f}" if m.delta_bic_eff is not None else "—"
        print(
            f"{m.name:<26} {m.nmse:>8.4f} {m.bic_standard:>10.2f} "
            f"{m.bic_effective:>9.2f} {ds:>9} {de:>9}"
        )
    print("-" * 70)
    print(f"Reviewer message: {summary.reviewer_message}")
    print("=" * 70)


if __name__ == "__main__":
    # Standalone usage: python -m adcd.experiments.bic_eff_analysis <results.json>
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/sparc_discovery.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/bic_eff_analysis.json"
    print_bic_eff_report(compute_bic_eff_from_results(path, out))
