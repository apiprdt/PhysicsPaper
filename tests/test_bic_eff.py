"""Tests for BIC_eff (effective-sample-size) analysis.

Validates the independent-unit BIC correction that reframes the SPARC
ADCD-vs-MOND comparison. The headline regression asserts that the module
reproduces the published numbers (ADCD BIC_eff ~ -158, inflation ~20x),
so any drift in the closed-form formula is caught.
"""

import json

import pytest

from adcd.experiments.bic_eff_analysis import (
    BICEffSummary,
    compute_bic_eff,
    compute_bic_eff_from_results,
    interpret_delta_bic,
    print_bic_eff_report,
)


# Real SPARC pipeline values (results/sparc_discovery.json), k=2 throughout.
N_GALAXIES = 171
N_POINTS = 3342
ADCD_NMSE = 0.372875912348825
SIMPLE_MOND_2P_NMSE = 0.3674392022869592


def _toy_models():
    return {
        "ADCD discovered": {"nmse": ADCD_NMSE, "n_params": 2, "fitted_theta": None},
        "Simple MOND (2-param)": {
            "nmse": SIMPLE_MOND_2P_NMSE,
            "n_params": 2,
            "fitted_theta": {"theta_0": 0.4970, "theta_1": 0.5476},
        },
    }


def test_interpret_delta_bic_categories():
    """Kass-Raftery evidence bands + ADCD direction sign."""
    assert interpret_delta_bic(0.5) == "Not worth mentioning evidence [ADCD loses]"
    assert interpret_delta_bic(-1.5) == "Not worth mentioning evidence [ADCD wins]"
    assert interpret_delta_bic(3.0) == "Positive evidence [ADCD loses]"
    assert interpret_delta_bic(-7.0) == "Strong evidence [ADCD wins]"
    assert interpret_delta_bic(50.0) == "Very Strong evidence [ADCD loses]"
    assert interpret_delta_bic(None) == "n/a"


def test_bic_formula_matches_metrics():
    """bic_eff uses the same closed form as adcd.metrics.bic_score at N=N_galaxies."""
    from adcd.metrics import bic_score

    nmse, k = 0.4, 2
    eff_via_metrics = bic_score(nmse, k, N_GALAXIES)
    summary = compute_bic_eff(
        {"ADCD discovered": {"nmse": nmse, "n_params": k}},
        n_galaxies=N_GALAXIES,
        n_points=N_POINTS,
    )
    assert summary.models[0].bic_effective == pytest.approx(eff_via_metrics, abs=1e-6)


def test_reproduces_headline_numbers():
    """Regression: ADCD BIC_eff ~ -158.4, inflation ~20x (the published framing)."""
    summary = compute_bic_eff(
        _toy_models(),
        n_galaxies=N_GALAXIES,
        n_points=N_POINTS,
    )

    adcd = next(m for m in summary.models if m.name == "ADCD discovered")
    assert adcd.bic_standard == pytest.approx(-3280.69, abs=0.1)
    assert adcd.bic_effective == pytest.approx(-158.4, abs=0.5)

    comp = next(m for m in summary.models if m.name == "Simple MOND (2-param)")
    # delta = BIC_ADCD - BIC_model: competitor has lower BIC => delta positive (ADCD loses).
    assert comp.delta_bic_std == pytest.approx(49.09, abs=0.2)   # ADCD loses at std scale
    assert comp.delta_bic_eff == pytest.approx(2.51, abs=0.3)    # ~inconclusive at eff scale
    assert summary.inflation_factor == pytest.approx(19.5, abs=1.0)


def test_delta_sign_positive_means_adcd_loses():
    """A competitor with strictly lower NMSE must have positive delta (ADCD loses)."""
    models = {
        "ADCD discovered": {"nmse": 0.50, "n_params": 2},
        "Better": {"nmse": 0.40, "n_params": 2},
    }
    summary = compute_bic_eff(models, n_galaxies=N_GALAXIES, n_points=N_POINTS)
    better = next(m for m in summary.models if m.name == "Better")
    # delta = BIC_ADCD - BIC_model: better model has lower BIC => delta positive.
    assert better.delta_bic_eff > 0
    assert "ADCD loses" in interpret_delta_bic(better.delta_bic_eff)


def test_models_sorted_by_bic_effective():
    """Output is ordered by bic_effective ascending (lower = better)."""
    models = {
        "ADCD discovered": {"nmse": 0.40, "n_params": 2},
        "Worse": {"nmse": 0.80, "n_params": 2},
    }
    summary = compute_bic_eff(models, n_galaxies=N_GALAXIES, n_points=N_POINTS)
    effs = [m.bic_effective for m in summary.models]
    assert effs == sorted(effs)


def test_inflation_factor_clustering():
    """Clustered (high N_points/N_galaxies) data inflates delta-BIC, sparse does not."""
    models = _toy_models()
    clustered = compute_bic_eff(models, n_galaxies=N_GALAXIES, n_points=N_POINTS)
    sparse = compute_bic_eff(models, n_galaxies=N_POINTS, n_points=N_POINTS)
    # When every point is its own cluster, std == eff, so inflation == 1.
    assert sparse.inflation_factor == pytest.approx(1.0, abs=1e-6)
    assert clustered.inflation_factor > 10.0  # ~20x for SPARC


def test_adcd_key_missing_raises():
    with pytest.raises(ValueError, match="ADCD key"):
        compute_bic_eff({"X": {"nmse": 0.4, "n_params": 2}},
                        n_galaxies=N_GALAXIES, n_points=N_POINTS,
                        adcd_key="ADCD discovered")


def test_invalid_sample_sizes_raise():
    models = {"ADCD discovered": {"nmse": 0.4, "n_params": 2}}
    with pytest.raises(ValueError):
        compute_bic_eff(models, n_galaxies=0, n_points=100)
    with pytest.raises(ValueError):
        compute_bic_eff(models, n_galaxies=200, n_points=100)  # n_gal > n_pts


def test_to_dict_roundtrip_jsonable():
    """Summary serialises cleanly to JSON (deltas resolved, no dataclasses)."""
    summary = compute_bic_eff(_toy_models(), n_galaxies=N_GALAXIES, n_points=N_POINTS)
    d = summary.to_dict()
    # Must be JSON-serialisable end-to-end.
    s = json.dumps(d)
    back = json.loads(s)
    assert back["N_galaxies"] == N_GALAXIES
    assert "interpretation_eff" in back["models"][0]
    assert back["inflation_factor"] == pytest.approx(19.5, abs=1.0)


def test_compute_from_results_json(tmp_path):
    """End-to-end: read a synthetic sparc_discovery.json, write report JSON."""
    payload = {
        "n_galaxies": N_GALAXIES,
        "n_points": N_POINTS,
        "final_nmse": ADCD_NMSE,
        "fitted_baselines": [
            {
                "name": "Simple MOND (2-param)",
                "nmse": SIMPLE_MOND_2P_NMSE,
                "n_params": 2,
                "fitted_theta": {"theta_0": 0.497, "theta_1": 0.548},
            },
        ],
    }
    src = tmp_path / "sparc_discovery.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "bic_eff_analysis.json"

    summary = compute_bic_eff_from_results(str(src), str(out))
    assert isinstance(summary, BICEffSummary)
    assert summary.inflation_factor == pytest.approx(19.5, abs=1.0)
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["N_galaxies"] == N_GALAXIES
    assert "reviewer_message" in written


def test_print_report_smoke(capsys):
    """print_bic_eff_report emits a readable table without error."""
    summary = compute_bic_eff(_toy_models(), n_galaxies=N_GALAXIES, n_points=N_POINTS)
    print_bic_eff_report(summary)
    captured = capsys.readouterr().out
    assert "BIC_eff Analysis" in captured
    assert "inflation_factor" in captured
    assert "ADCD discovered" in captured


def test_effective_scale_is_more_conservative():
    """Core scientific claim: BIC_eff yields weaker evidence than BIC_std."""
    summary = compute_bic_eff(_toy_models(), n_galaxies=N_GALAXIES, n_points=N_POINTS)
    comp = next(m for m in summary.models if m.name == "Simple MOND (2-param)")
    # |delta_eff| must be strictly smaller than |delta_std| for clustered data.
    assert abs(comp.delta_bic_eff) < abs(comp.delta_bic_std)
