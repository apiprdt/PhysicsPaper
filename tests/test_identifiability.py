"""Unit tests for IdentifiabilityAnalyzer (Task P3-2)."""

import numpy as np
from unittest.mock import MagicMock
from adcd.identifiability import IdentifiabilityAnalyzer, IdentifiabilityReport


def _mock_bayesian_output(weights):
    """Helper to create a mock BayesianCorrectionOutput with given weights."""
    obj = MagicMock()
    obj.posterior_weights = weights
    return obj


def test_identifiable_case():
    """High SNR, clear winner -> identifiable."""
    y_classical = np.ones(100) * 10.0
    residual = np.sin(np.linspace(0, 2, 100)) * 5.0   # large correction
    bayesian = _mock_bayesian_output([0.95, 0.05])

    analyzer = IdentifiabilityAnalyzer()
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.01)

    assert report.is_identifiable
    assert report.failure_mode is None
    assert report.snr > IdentifiabilityAnalyzer.SNR_THRESHOLD
    assert report.weight_ratio > IdentifiabilityAnalyzer.WEIGHT_RATIO_THRESHOLD


def test_low_snr_failure():
    """Correction buried in noise -> low_snr failure."""
    # y_classical near 1.0 so noise_magnitude = noise_level * std(y_classical) ~ noise_level * 0
    # Use varying y_classical so std > 0, but make residual << noise
    y_classical = np.linspace(1.0, 2.0, 100)          # std ~ 0.29
    residual = np.ones(100) * 1e-6                     # correction std = 0, effectively 0
    bayesian = _mock_bayesian_output([0.9, 0.1])

    analyzer = IdentifiabilityAnalyzer()
    # noise_magnitude = 0.5 * std(y_classical) ~ 0.5 * 0.29 = 0.145
    # correction_magnitude = std(residual) = 0 -> SNR << 1
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.5)

    assert not report.is_identifiable
    assert report.failure_mode in ("low_snr", "undetectable_magnitude")
    assert report.snr < IdentifiabilityAnalyzer.SNR_THRESHOLD


def test_model_degeneracy_failure():
    """Good SNR but ambiguous posterior -> model_degeneracy."""
    y_classical = np.ones(100) * 10.0
    residual = np.linspace(0.1, 1.0, 100)   # strong signal
    bayesian = _mock_bayesian_output([0.51, 0.49])  # nearly equal weights

    analyzer = IdentifiabilityAnalyzer()
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.001)

    assert not report.is_identifiable
    assert report.failure_mode == "model_degeneracy"
    assert report.weight_ratio < IdentifiabilityAnalyzer.WEIGHT_RATIO_THRESHOLD


def test_undetectable_magnitude_failure():
    """Vanishingly small correction -> undetectable_magnitude."""
    y_classical = np.ones(100) * 1e6
    residual = np.ones(100) * 1e-20   # effectively zero
    bayesian = _mock_bayesian_output([0.99, 0.01])

    analyzer = IdentifiabilityAnalyzer()
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.0)

    assert not report.is_identifiable
    assert report.failure_mode == "undetectable_magnitude"
    assert report.relative_magnitude < IdentifiabilityAnalyzer.MAGNITUDE_THRESHOLD


def test_single_candidate_not_degenerate():
    """Single posterior candidate -> weight_ratio = inf, not model_degeneracy."""
    y_classical = np.ones(50) * 5.0
    residual = np.linspace(0.5, 2.0, 50)
    bayesian = _mock_bayesian_output([1.0])   # only one candidate

    analyzer = IdentifiabilityAnalyzer()
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.0)

    assert np.isinf(report.weight_ratio)
    # should not fail on model_degeneracy
    assert report.failure_mode != "model_degeneracy"


def test_report_is_dataclass():
    """Output is IdentifiabilityReport with correct field types."""
    y_classical = np.ones(50) * 10.0
    residual = np.ones(50) * 2.0
    bayesian = _mock_bayesian_output([0.8, 0.2])

    analyzer = IdentifiabilityAnalyzer()
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.05)

    assert isinstance(report, IdentifiabilityReport)
    assert isinstance(report.is_identifiable, bool)
    assert isinstance(report.snr, float)
    assert isinstance(report.weight_ratio, float)
    assert isinstance(report.relative_magnitude, float)
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0


def test_summary_contains_snr():
    """Summary string should mention SNR value."""
    y_classical = np.ones(50) * 10.0
    residual = np.ones(50) * 5.0
    bayesian = _mock_bayesian_output([0.9, 0.1])

    analyzer = IdentifiabilityAnalyzer()
    report = analyzer.analyze(bayesian, residual, y_classical, noise_level=0.01)
    assert "SNR" in report.summary
