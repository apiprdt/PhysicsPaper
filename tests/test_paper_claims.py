"""Tests mapping to defensible paper claims."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gate_stats_dataclass_exported():
    from adcd import GateStats
    g = GateStats(input_count=10, output_count=2)
    assert g.survival_rates()["overall"] == pytest.approx(0.2)


def test_pysr_profiles_defined():
    from run_pysr_baseline import PYSR_PROFILES
    assert "fair" in PYSR_PROFILES
    assert PYSR_PROFILES["fair"]["niterations"] >= 100


def test_mercury_loader_uses_vc2():
    from adcd.real_data_loader import load_mercury_perihelion
    X, y_obs, y_cls, residual = load_mercury_perihelion()
    assert "vc2" in X
    assert X["vc2"].min() > 0
    assert (X["vc2"] < 1).all()


def test_binary_pulsar_loader():
    from adcd.real_data_loader import load_binary_pulsar_decay
    X, y_obs, y_cls, residual = load_binary_pulsar_decay()
    assert len(y_obs) == 60
    assert (y_obs > 0).all()


def test_real_scenarios_count():
    from adcd.real_scenarios import get_real_scenarios
    assert len(get_real_scenarios()) == 5


def test_validate_results_no_stale_mercury_claim():
    report = ROOT / "experiment_results.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        assert "1.34e-28" not in text


def test_experiment_report_auto_generated_header():
    report = ROOT / "experiment_results.md"
    if report.exists():
        assert "AUTO-GENERATED" in report.read_text(encoding="utf-8")
