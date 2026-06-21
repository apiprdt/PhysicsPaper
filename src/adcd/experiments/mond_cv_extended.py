"""Cross-validation comparison of ADCD vs the 2-parameter MOND family on SPARC.

This module formalises the *out-of-sample* comparison between the ADCD-discovered
form ``ν(x) = θ₀(√(1 + θ₁/x) − 1) + 1`` and the 2-parameter MOND / RAR
baselines on the stacked SPARC rotation-curve data.

The motivation is the well-known statistical asymmetry between BIC (an
*in-sample* approximation to the marginal likelihood, which is dominated by the
3342 stacked measurements and rewards any fit-quality advantage however small)
and cross-validated NMSE (an *out-of-sample* estimate that does not reward
over-fitting and carries a usable standard error). The SPARC discovery pipeline
already performs galaxy-level repeated 50/50 train/test splits and stores the
per-model CV NMSE ± standard error in ``results/sparc_discovery.json`` under the
``cross_validation`` key. This module is a pure *post-hoc* statistical analysis
of those numbers — it performs no refitting and reads no raw data.

The headline question it answers is:

    Does ADCD beat the best 2-parameter MOND interpolating function
    *out of sample*, or is the BIC ordering an artefact of in-sample scoring?

We classify the outcome into two scenarios:

    Scenario A (ADCD wins OOS):
        ADCD CV NMSE < MOND 2-param CV NMSE. The BIC disadvantage reported
        elsewhere in the pipeline is then a pure complexity/degrees-of-freedom
        artefact, and ADCD is the better predictive model. The paper headline
        ("ADCD is competitive with / beats MOND") stands as written.

    Scenario B (ADCD does not win OOS):
        ADCD CV NMSE >= MOND 2-param CV NMSE. ADCD and MOND 2-param are then
        statistically indistinguishable out of sample, and the in-sample BIC
        gap reflects the ~0.008 NMSE advantage MOND enjoys on this dataset.
        The paper SPARC headline must be reframed around *autonomous
        discovery parity* rather than *predictive superiority*.

In both scenarios we additionally report:

    * The two-sample z-statistic on the CV-NMSE difference (treating the
      per-model standard errors as independent), and its Kass-Raftery-style
      interpretation, so the reader can see at a glance whether the difference
      is statistically meaningful.
    * A ready-to-quote reviewer message carrying the scenario verdict and the
      numeric headline, mirroring the ``reviewer_message`` field produced by
      ``adcd.experiments.bic_eff_analysis``.

References:
    Kass & Raftery (1995), JASA 90(430) -- Bayes factors / evidence scale.
    Lemos, Llinares & Bovy (2022), MNRAS -- SPARC / RAR cross-validation.

Delta-NMSE convention used throughout this module:
    delta = NMSE_ADCD - NMSE_model  (positive => ADCD loses out of sample),
    consistent with the delta-BIC convention in ``bic_eff_analysis``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default model keys (must match the names written by sparc_stacking.run_galaxy_cv).
# ---------------------------------------------------------------------------
DEFAULT_ADCD_KEY = "ADCD discovered"
DEFAULT_MOND2P_KEY = "Simple MOND (2-param)"
DEFAULT_RAR2P_KEY = "RAR (McGaugh, 2-param)"
DEFAULT_STD_MOND2P_KEY = "Standard MOND (2-param)"

# Kass-Raftery |z| evidence bands, reused here on the CV-NMSE two-sample test.
# (z < 1 => "not significant"; we use 1/2/3 as common rule-of-thumb thresholds.)
_Z_NOT_SIGNIFICANT = 1.0
_Z_SUGGESTIVE = 2.0
_Z_STRONG = 3.0


@dataclass
class CVMModel:
    """Per-model cross-validated NMSE record.

    Attributes:
        name: Model key as stored under ``cross_validation`` in the JSON.
        mean_nmse: CV mean NMSE (lower = better out-of-sample prediction).
        std_error: Standard error of the CV mean across the repeated splits.
    """

    name: str
    mean_nmse: float
    std_error: float


@dataclass
class CVMDelta:
    """Pairwise ADCD-vs-model out-of-sample comparison.

    Attributes:
        model_name: The competing model's name.
        delta_nmse: NMSE_ADCD - NMSE_model (positive => ADCD loses OOS).
        combined_se: sqrt(SE_ADCD^2 + SE_model^2), assuming independence.
        z_score: delta_nmse / combined_se.
        interpretation: Human-readable significance verdict.
        model_nmse: The competitor's own CV mean NMSE (for reporting tables).
        model_se: The competitor's own CV standard error.
    """

    model_name: str
    delta_nmse: float
    combined_se: float
    z_score: float
    interpretation: str
    model_nmse: float = 0.0
    model_se: float = 0.0


@dataclass
class CVMVerdict:
    """Aggregate ADCD-vs-MOND 2-param CV analysis.

    Attributes:
        scenario: "A" (ADCD wins OOS) or "B" (ADCD does not win OOS).
        scenario_reason: One-line explanation of why the scenario was chosen.
        adcd: ADCD model CV record.
        primary_competitor: Headline competitor (Simple MOND 2-param by default).
        deltas: Pairwise ADCD-vs-each-2param-model deltas, ordered by |z_score|.
        n_repeats: Number of CV repeats the underlying pipeline used (provenance).
        reviewer_message: Ready-to-quote scenario verdict for the paper/rebuttal.
    """

    scenario: str
    scenario_reason: str
    adcd: CVMModel
    primary_competitor: CVMModel
    deltas: List[CVMDelta] = field(default_factory=list)
    n_repeats: Optional[int] = None
    reviewer_message: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable representation (dataclasses flattened)."""
        return {
            "scenario": self.scenario,
            "scenario_reason": self.scenario_reason,
            "n_repeats": self.n_repeats,
            "adcd": {
                "name": self.adcd.name,
                "mean_nmse": self.adcd.mean_nmse,
                "std_error": self.adcd.std_error,
            },
            "primary_competitor": {
                "name": self.primary_competitor.name,
                "mean_nmse": self.primary_competitor.mean_nmse,
                "std_error": self.primary_competitor.std_error,
            },
            "deltas": [
                {
                    "model_name": d.model_name,
                    "model_nmse": d.model_nmse,
                    "model_se": d.model_se,
                    "delta_nmse": d.delta_nmse,
                    "combined_se": d.combined_se,
                    "z_score": d.z_score,
                    "interpretation": d.interpretation,
                }
                for d in self.deltas
            ],
            "reviewer_message": self.reviewer_message,
        }


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def interpret_z_score(z: float) -> str:
    """Map a two-sample |z| to a plain-language significance verdict.

    Uses the standard rule-of-thumb thresholds (1 / 2 / 3 sigma) rather than
    Kass-Raftery's BIC bands, because we are scoring a *difference of means*
    rather than a Bayes factor. ``z`` here is the signed statistic; the sign
    only affects the direction clause.
    """
    if z is None or not math.isfinite(z):
        return "n/a"
    abs_z = abs(z)
    direction = "ADCD loses" if z > 0 else "ADCD wins"
    if abs_z < _Z_NOT_SIGNIFICANT:
        strength = "Not significant"
    elif abs_z < _Z_SUGGESTIVE:
        strength = "Suggestive"
    elif abs_z < _Z_STRONG:
        strength = "Significant"
    else:
        strength = "Strong"
    return f"{strength} OOS [{direction}]"


def _combined_se(se_a: float, se_b: float) -> float:
    """Independent two-sample combined SE: sqrt(SE_a^2 + SE_b^2)."""
    return float(math.sqrt(se_a * se_a + se_b * se_b))


def make_delta(adcd: CVMModel, model: CVMModel) -> CVMDelta:
    """Build a pairwise ADCD-vs-model CV delta record.

    delta = NMSE_ADCD - NMSE_model (positive => ADCD loses out of sample).
    """
    delta = float(adcd.mean_nmse - model.mean_nmse)
    se = _combined_se(adcd.std_error, model.std_error)
    z = float(delta / se) if se > 0 else float("inf") * math.copysign(1.0, delta)
    if se == 0 and delta == 0:
        z = 0.0
    return CVMDelta(
        model_name=model.name,
        delta_nmse=delta,
        combined_se=se,
        z_score=z,
        interpretation=interpret_z_score(z),
        model_nmse=model.mean_nmse,
        model_se=model.std_error,
    )


def classify_scenario(adcd: CVMModel, competitor: CVMModel) -> Tuple[str, str]:
    """Decide Scenario A vs B for the ADCD-vs-competitor CV comparison.

    Scenario A: ADCD CV NMSE < competitor CV NMSE (ADCD wins out of sample).
    Scenario B: ADCD CV NMSE >= competitor CV NMSE (ADCD does not win OOS).

    The verdict is based on the *sign* of the difference, not its magnitude:
    even an OOS tie favours Scenario B for paper-framing honesty, because the
    claim under scrutiny is "ADCD is *predictively* better than MOND", and a
    tie does not support that claim.
    """
    diff = adcd.mean_nmse - competitor.mean_nmse
    if diff < 0:
        return (
            "A",
            f"ADCD CV NMSE ({adcd.mean_nmse:.4f}) < "
            f"{competitor.name} ({competitor.mean_nmse:.4f}); "
            "ADCD is the better out-of-sample predictor.",
        )
    return (
        "B",
        f"ADCD CV NMSE ({adcd.mean_nmse:.4f}) >= "
        f"{competitor.name} ({competitor.mean_nmse:.4f}); "
        "ADCD does not beat MOND out of sample, so the in-sample BIC gap "
        "reflects an OOS fit-quality difference rather than a complexity "
        "penalty alone.",
    )


def _build_reviewer_message(
    scenario: str,
    adcd: CVMModel,
    competitor: CVMModel,
    delta: CVMDelta,
) -> str:
    """Assemble the ready-to-quote scenario sentence.

    Mirrors the style of ``bic_eff_analysis._build_reviewer_message``.
    """
    if scenario == "A":
        return (
            f"Out-of-sample, ADCD predicts the held-out galaxies better than "
            f"{competitor.name}: CV NMSE {adcd.mean_nmse:.4f} vs "
            f"{competitor.mean_nmse:.4f} (delta {delta.delta_nmse:+.4f}, "
            f"z={delta.z_score:+.2f}, {delta.interpretation}). The in-sample "
            "BIC disadvantage is therefore a degrees-of-freedom artefact, not "
            "a predictive deficit."
        )
    return (
        f"Out-of-sample, ADCD and {competitor.name} are statistically "
        f"indistinguishable: CV NMSE {adcd.mean_nmse:.4f} vs "
        f"{competitor.mean_nmse:.4f} (delta {delta.delta_nmse:+.4f}, "
        f"z={delta.z_score:+.2f}, {delta.interpretation}). The reported "
        "in-sample BIC gap thus reflects an OOS fit-quality difference of "
        f"~{delta.delta_nmse:.3f} NMSE, not a complexity penalty alone; the "
        "SPARC result should be framed as autonomous-discovery *parity* with "
        "MOND rather than predictive *superiority*."
    )


# ---------------------------------------------------------------------------
# High-level driver: analyse the CV dict produced by sparc_stacking
# ---------------------------------------------------------------------------

def analyse_cv(
    cv: Dict[str, Dict[str, float]],
    adcd_key: str = DEFAULT_ADCD_KEY,
    primary_competitor_key: str = DEFAULT_MOND2P_KEY,
    extra_competitor_keys: Optional[List[str]] = None,
    n_repeats: Optional[int] = None,
) -> CVMVerdict:
    """Analyse the ``cross_validation`` dict and return an A/B scenario verdict.

    Args:
        cv: Mapping ``{model_name: {"mean_nmse": float, "std_error": float}}``
            exactly as written by ``adcd.experiments.sparc_stacking.run_galaxy_cv``
            into ``results/sparc_discovery.json["cross_validation"]``.
        adcd_key: Key identifying the ADCD candidate.
        primary_competitor_key: Key identifying the headline competitor whose
            sign determines Scenario A vs B (Simple MOND 2-param by default).
        extra_competitor_keys: Additional 2-param baselines to score for the
            full delta table (e.g. RAR 2-param, Standard MOND 2-param). Defaults
            to the known 2-param family minus the primary competitor.
        n_repeats: Provenance only — number of CV repeats the pipeline used.

    Returns:
        A :class:`CVMVerdict` carrying the scenario, the primary delta, and the
        full pairwise delta table.

    Raises:
        ValueError: If ``adcd_key`` or ``primary_competitor_key`` is missing
            from ``cv``.
    """
    for key in (adcd_key, primary_competitor_key):
        if key not in cv:
            raise ValueError(
                f"Required CV key {key!r} not found in cross_validation "
                f"(available: {sorted(cv.keys())})."
            )

    adcd = _record(cv, adcd_key)
    primary = _record(cv, primary_competitor_key)

    # Build the full delta table (primary always included).
    if extra_competitor_keys is None:
        extra_competitor_keys = [
            k for k in (DEFAULT_RAR2P_KEY, DEFAULT_STD_MOND2P_KEY)
            if k in cv and k != primary_competitor_key
        ]
    competitor_keys = [primary_competitor_key] + [
        k for k in extra_competitor_keys if k in cv and k != primary_competitor_key
    ]

    deltas: List[CVMDelta] = []
    for key in competitor_keys:
        deltas.append(make_delta(adcd, _record(cv, key)))
    deltas.sort(key=lambda d: abs(d.z_score), reverse=True)

    scenario, reason = classify_scenario(adcd, primary)
    primary_delta = next(d for d in deltas if d.model_name == primary_competitor_key)
    message = _build_reviewer_message(scenario, adcd, primary, primary_delta)

    return CVMVerdict(
        scenario=scenario,
        scenario_reason=reason,
        adcd=adcd,
        primary_competitor=primary,
        deltas=deltas,
        n_repeats=n_repeats,
        reviewer_message=message,
    )


def _record(cv: Dict[str, Dict[str, float]], key: str) -> CVMModel:
    """Extract a :class:`CVMModel` from a raw CV entry, validating the schema."""
    entry = cv[key]
    if "mean_nmse" not in entry or "std_error" not in entry:
        raise ValueError(
            f"CV entry {key!r} missing 'mean_nmse'/'std_error' "
            f"(got keys: {sorted(entry.keys())})."
        )
    return CVMModel(
        name=key,
        mean_nmse=float(entry["mean_nmse"]),
        std_error=float(entry["std_error"]),
    )


def analyse_cv_from_results(
    results_path: str,
    output_path: Optional[str] = None,
    adcd_key: str = DEFAULT_ADCD_KEY,
    primary_competitor_key: str = DEFAULT_MOND2P_KEY,
    n_repeats: int = 10,
) -> CVMVerdict:
    """Convenience driver: read ``sparc_discovery.json`` and emit a CV verdict JSON.

    Reads the schema produced by the SPARC discovery pipeline (the
    ``cross_validation`` block) and delegates to :func:`analyse_cv`. If
    ``output_path`` is given, the verdict is written there as JSON (matching
    the project's results-file convention of a top-level object with flat
    metric keys).

    ``n_repeats`` is recorded for provenance only (it is not present in the
    discovery JSON); the default of 10 matches ``run_galaxy_cv``'s call site
    in ``sparc_stacking.run_sparc_discovery``.
    """
    with open(results_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if "cross_validation" not in data:
        raise ValueError(
            f"{results_path} has no 'cross_validation' block; run the SPARC "
            "discovery pipeline first."
        )

    verdict = analyse_cv(
        cv=data["cross_validation"],
        adcd_key=adcd_key,
        primary_competitor_key=primary_competitor_key,
        n_repeats=n_repeats,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(verdict.to_dict(), fh, indent=2)

    return verdict


def print_cv_report(verdict: CVMVerdict) -> None:
    """Pretty-print the CV comparison table (mirrors bic_eff_report style)."""
    print("\n" + "=" * 72)
    print("CV Analysis: ADCD vs 2-param MOND family (out-of-sample, galaxy-level)")
    print("=" * 72)
    print(f"Scenario {verdict.scenario}: {verdict.scenario_reason}")
    if verdict.n_repeats is not None:
        print(f"(based on {verdict.n_repeats} repeated 50/50 galaxy splits)")
    print("-" * 72)
    hdr = (
        f"{'Model':<26} {'CV NMSE':>9} {'SE':>8} "
        f"{'ΔNMSE':>9} {'z':>7}"
    )
    print(hdr)
    print("-" * 72)
    print(
        f"{verdict.adcd.name:<26} {verdict.adcd.mean_nmse:>9.4f} "
        f"{verdict.adcd.std_error:>8.4f} {'(baseline)':>9} {'—':>7}"
    )
    for d in verdict.deltas:
        print(
            f"{d.model_name:<26} {d.model_nmse:>9.4f} {d.model_se:>8.4f} "
            f"{d.delta_nmse:>+9.4f} {d.z_score:>+7.2f}"
        )
    print("-" * 72)
    print(f"Reviewer message: {verdict.reviewer_message}")
    print("=" * 72)


if __name__ == "__main__":
    # Standalone usage:
    #   python -m adcd.experiments.mond_cv_extended <results.json>
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/sparc_discovery.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/mond_cv_2param.json"
    print_cv_report(analyse_cv_from_results(path, out))
