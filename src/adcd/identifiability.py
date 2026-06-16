"""
Identifiability analysis for discovered corrections.

Diagnoses WHY a correction cannot be uniquely identified from data,
converting "system failed" narratives into precise, scientifically
meaningful statements about data information limits.

Three failure modes diagnosed:
1. model_degeneracy:        top-2 candidates statistically indistinguishable
2. low_snr:                 correction signal buried in measurement noise
3. undetectable_magnitude:  correction too small relative to classical prediction

Reference:
    Fisher (1925) Statistical Methods for Research Workers -- identifiability
    Rissanen (1978) Modeling by shortest data description -- MDL
    Physics context: screened Coulomb vs Yukawa are indistinguishable below SNR threshold
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IdentifiabilityReport:
    """
    Identifiability analysis result for a discovered correction.

    Attributes:
        is_identifiable: True if correction can be reliably identified
        failure_mode: "model_degeneracy" | "low_snr" | "undetectable_magnitude" | None
        snr: Signal-to-noise ratio of the correction (correction_std / noise_std)
        weight_ratio: Posterior weight ratio of top-2 candidates (inf if only 1 candidate)
        relative_magnitude: Median of |delta| / |y_classical| across data points
        summary: Human-readable summary for paper reporting
    """
    is_identifiable: bool
    failure_mode: Optional[str]
    snr: float
    weight_ratio: float
    relative_magnitude: float
    summary: str


class IdentifiabilityAnalyzer:
    """
    Analyzes whether the discovered correction is identifiable from data.

    A correction is identifiable if:
    1. It is large enough to be detected above noise (SNR > SNR_THRESHOLD)
    2. Its relative magnitude is non-negligible (> MAGNITUDE_THRESHOLD)
    3. The posterior unambiguously favors one candidate over others
       (weight_ratio > WEIGHT_RATIO_THRESHOLD)

    Inspired by:
    - Fisher information / Cramer-Rao bound (parameter identifiability)
    - Rissanen's MDL (information-theoretic identifiability)
    - Physics: indistinguishability below SNR threshold
    """

    SNR_THRESHOLD = 1.0
    WEIGHT_RATIO_THRESHOLD = 3.0
    MAGNITUDE_THRESHOLD = 1e-10

    def analyze(
        self,
        bayesian_output,          # BayesianCorrectionOutput
        residual: np.ndarray,
        y_classical: np.ndarray,
        noise_level: float = 0.0,
    ) -> IdentifiabilityReport:
        """
        Analyze identifiability of a discovered correction.

        Args:
            bayesian_output: BayesianCorrectionOutput from BayesianReranker.rank()
            residual: Array of residual values (observed - classical)
            y_classical: Array of classical model predictions
            noise_level: Fractional noise level (e.g. 0.05 for 5% noise)

        Returns:
            IdentifiabilityReport with diagnosis and human-readable summary.
        """
        residual = np.asarray(residual, dtype=float)
        y_classical = np.asarray(y_classical, dtype=float)

        # --- 1. Compute SNR ---
        correction_magnitude = float(np.std(residual))
        noise_magnitude = float(noise_level * np.std(y_classical)) + 1e-15
        snr = correction_magnitude / noise_magnitude

        # --- 2. Compute relative magnitude ---
        relative_magnitude = float(
            np.median(np.abs(residual) / (np.abs(y_classical) + 1e-15))
        )

        # --- 3. Compute posterior weight ratio ---
        weights = bayesian_output.posterior_weights
        if len(weights) >= 2:
            weight_ratio = float(weights[0] / (weights[1] + 1e-15))
        else:
            weight_ratio = float("inf")

        # --- 4. Diagnose failure mode (ordered by severity) ---
        failure_mode: Optional[str] = None

        if relative_magnitude < self.MAGNITUDE_THRESHOLD:
            failure_mode = "undetectable_magnitude"
        elif snr < self.SNR_THRESHOLD:
            failure_mode = "low_snr"
        elif weight_ratio < self.WEIGHT_RATIO_THRESHOLD:
            failure_mode = "model_degeneracy"

        is_identifiable = failure_mode is None

        # --- 5. Generate human-readable summary ---
        summary = self._build_summary(
            is_identifiable, failure_mode, snr, weight_ratio, relative_magnitude
        )

        return IdentifiabilityReport(
            is_identifiable=is_identifiable,
            failure_mode=failure_mode,
            snr=snr,
            weight_ratio=weight_ratio,
            relative_magnitude=relative_magnitude,
            summary=summary,
        )

    def _build_summary(
        self,
        is_identifiable: bool,
        failure_mode: Optional[str],
        snr: float,
        weight_ratio: float,
        relative_magnitude: float,
    ) -> str:
        if is_identifiable:
            return (
                f"Correction is identifiable: SNR={snr:.2f}, "
                f"weight_ratio={weight_ratio:.1f}, "
                f"relative_magnitude={relative_magnitude:.2e}"
            )
        elif failure_mode == "undetectable_magnitude":
            return (
                f"Correction undetectable: relative magnitude {relative_magnitude:.2e} "
                f"is below threshold {self.MAGNITUDE_THRESHOLD:.0e}. "
                f"Classical model already explains data completely."
            )
        elif failure_mode == "low_snr":
            return (
                f"Correction not identifiable: SNR={snr:.2f} < {self.SNR_THRESHOLD} "
                f"(correction magnitude {relative_magnitude:.2e} relative to classical). "
                f"More precise measurements needed."
            )
        elif failure_mode == "model_degeneracy":
            wr_str = f"{weight_ratio:.1f}" if np.isfinite(weight_ratio) else "inf"
            return (
                f"Model degeneracy: posterior weight ratio={wr_str} < "
                f"{self.WEIGHT_RATIO_THRESHOLD} (ambiguous between top candidates). "
                f"SNR={snr:.2f} is sufficient but data cannot distinguish functional forms."
            )
        else:
            return "Unknown identifiability status."
