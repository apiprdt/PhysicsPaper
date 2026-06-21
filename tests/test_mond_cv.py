"""Tests for the extended ADCD-vs-MOND 2-param cross-validation analysis.

Validates the out-of-sample scenario verdict that re-frames the SPARC
ADCD-vs-MOND comparison. The headline regression asserts that the module
reproduces the published numbers (ADCD CV NMSE ~0.386, MOND 2-param CV NMSE
~0.377 => Scenario B), so any drift in the analysis is caught.
"""

import json

import pytest

from adcd.experiments.mond_cv_extended import (
    CVMVerdict,
    analyse_cv,
    analyse_cv_from_results,
    classify_scenario,
    interpret_z_score,
    make_delta,
    print_cv_report,
)
from adcd.experiments.mond_cv_extended import CVMModel


# Real SPARC pipeline values (results/sparc_discovery.json cross_validation),
# the same numbers the BIC_eff regression test pins.
ADCD_CV = 0.38563896471434633
ADCD_SE = 0.015509196346496975
MOND2P_CV = 0.3773742214187018
MOND2P_SE = 0.014671820816077381
RAR2P_CV = 0.377421484832549
RAR2P_SE = 0.014656533475602921
STD_MOND2P_CV = 0.4664070473738463
STD_MOND2P_SE = 0.008236723909594231


def _real_cv():
    """Reconstruct the cross_validation dict as written by sparc_stacking."""
    return {
        "Simple MOND": {"mean_nmse": 0.6725, "std_error": 0.0350},
        "Standard MOND": {"mean_nmse": 0.7016, "std_error": 0.0055},
        "RAR (McGaugh)": {"mean_nmse": 0.6601, "std_error": 0.0346},
        "ADCD discovered": {"mean_nmse": ADCD_CV, "std_error": ADCD_SE},
        "Simple MOND (2-param)": {"mean_nmse": MOND2P_CV, "std_error": MOND2P_SE},
        "Standard MOND (2-param)": {"mean_nmse": STD_MOND2P_CV, "std_error": STD_MOND2P_SE},
        "RAR (McGaugh, 2-param)": {"mean_nmse": RAR2P_CV, "std_error": RAR2P_SE},
    }


# ---------------------------------------------------------------------------
# interpret_z_score
# ---------------------------------------------------------------------------

def test_interpret_z_score_bands():
    """Rule-of-thumb significance bands + ADCD direction sign."""
    assert interpret_z_score(0.5) == "Not significant OOS [ADCD loses]"
    assert interpret_z_score(-1.5) == "Suggestive OOS [ADCD wins]"
    assert interpret_z_score(2.5) == "Significant OOS [ADCD loses]"
    assert interpret_z_score(-4.0) == "Strong OOS [ADCD wins]"
    assert interpret_z_score(None) == "n/a"


# ---------------------------------------------------------------------------
# make_delta
# ---------------------------------------------------------------------------

def test_delta_sign_positive_means_adcd_loses():
    """A competitor with strictly lower NMSE must have positive delta."""
    adcd = CVMModel("ADCD discovered", 0.50, 0.01)
    better = CVMModel("Better", 0.40, 0.01)
    d = make_delta(adcd, better)
    assert d.delta_nmse > 0
    assert d.z_score > 0
    assert "ADCD loses" in d.interpretation


def test_delta_combined_se_independence():
    """Combined SE = sqrt(SE_a^2 + SE_b^2) for independent estimates."""
    adcd = CVMModel("ADCD", 0.4, 0.03)
    other = CVMModel("Other", 0.4, 0.04)
    d = make_delta(adcd, other)
    expected = (0.03 ** 2 + 0.04 ** 2) ** 0.5
    assert d.combined_se == pytest.approx(expected, abs=1e-9)
    # delta == 0 => z == 0 regardless of SE.
    assert d.z_score == pytest.approx(0.0, abs=1e-12)


def test_delta_zero_se_zero_delta_is_zero_z():
    """Degenerate edge: identical NMSE with zero SE => z == 0 (not inf/nan)."""
    adcd = CVMModel("ADCD", 0.4, 0.0)
    other = CVMModel("Other", 0.4, 0.0)
    d = make_delta(adcd, other)
    assert d.z_score == 0.0
    assert d.delta_nmse == 0.0


# ---------------------------------------------------------------------------
# classify_scenario
# ---------------------------------------------------------------------------

def test_classify_scenario_a_when_adcd_lower():
    """Scenario A: ADCD CV NMSE strictly below competitor."""
    adcd = CVMModel("ADCD discovered", 0.37, 0.01)
    comp = CVMModel("Simple MOND (2-param)", 0.40, 0.01)
    scenario, reason = classify_scenario(adcd, comp)
    assert scenario == "A"
    assert "better out-of-sample" in reason


def test_classify_scenario_b_when_adcd_equal():
    """Scenario B on an OOS tie (paper-framing honesty: a tie is not a win)."""
    adcd = CVMModel("ADCD discovered", 0.40, 0.01)
    comp = CVMModel("Simple MOND (2-param)", 0.40, 0.01)
    scenario, reason = classify_scenario(adcd, comp)
    assert scenario == "B"
    assert "does not beat" in reason


def test_classify_scenario_b_when_adcd_higher():
    """Scenario B: ADCD CV NMSE >= competitor (the real SPARC result)."""
    adcd = CVMModel("ADCD discovered", 0.39, 0.01)
    comp = CVMModel("Simple MOND (2-param)", 0.37, 0.01)
    scenario, _ = classify_scenario(adcd, comp)
    assert scenario == "B"


# ---------------------------------------------------------------------------
# analyse_cv — headline regression on real SPARC numbers
# ---------------------------------------------------------------------------

def test_analyse_cv_reproduces_scenario_b():
    """Regression: real SPARC CV => Scenario B, MOND 2-param wins OOS by ~0.4 sigma."""
    verdict = analyse_cv(_real_cv())

    assert verdict.scenario == "B"
    assert verdict.adcd.mean_nmse == pytest.approx(ADCD_CV, abs=1e-9)
    assert verdict.primary_competitor.mean_nmse == pytest.approx(MOND2P_CV, abs=1e-9)

    primary_delta = next(
        d for d in verdict.deltas if d.model_name == "Simple MOND (2-param)"
    )
    # delta = NMSE_ADCD - NMSE_MOND ~ +0.0083 (ADCD loses by a hair).
    assert primary_delta.delta_nmse == pytest.approx(ADCD_CV - MOND2P_CV, abs=1e-6)
    # z ~ +0.39 (well below 1 sigma => not significant).
    assert primary_delta.z_score == pytest.approx(0.387, abs=0.05)
    assert "Not significant" in primary_delta.interpretation


def test_analyse_cv_includes_extra_competitors():
    """RAR 2-param and Standard MOND 2-param are both present in the delta table."""
    verdict = analyse_cv(_real_cv())
    names = {d.model_name for d in verdict.deltas}
    assert "Simple MOND (2-param)" in names
    assert "RAR (McGaugh, 2-param)" in names
    assert "Standard MOND (2-param)" in names
    # Zero-param baselines must NOT leak into the 2-param comparison.
    assert "Simple MOND" not in names
    assert "Standard MOND" not in names


def test_analyse_cv_deltas_sorted_by_abs_z():
    """Delta table is ordered by |z| descending (most significant first)."""
    verdict = analyse_cv(_real_cv())
    zs = [abs(d.z_score) for d in verdict.deltas]
    assert zs == sorted(zs, reverse=True)


def test_analyse_cv_reviewer_message_mentions_scenario():
    """The ready-to-quote message names the scenario and the parity framing."""
    verdict = analyse_cv(_real_cv())
    assert "indistinguishable" in verdict.reviewer_message
    assert "parity" in verdict.reviewer_message


# ---------------------------------------------------------------------------
# Robustness to the framing-fragility we found in B1: zero-param entries
# must NOT be picked up as the primary competitor by accident.
# ---------------------------------------------------------------------------

def test_analyse_cv_ignores_zero_param_baseline_names():
    """Even if a zero-param key is requested as competitor, it must be present
    in cv to be used; otherwise the explicit error fires (no silent fallback
    to a zero-param model)."""
    cv = _real_cv()
    # 'Simple MOND' (zero-param) exists in the dict, but as a competitor it
    # would give a false Scenario A (its CV NMSE is huge). The module must
    # honour the caller's explicit key choice, not second-guess it.
    verdict = analyse_cv(cv, primary_competitor_key="Simple MOND")
    assert verdict.scenario == "A"  # ADCD beats the canonical zero-param form
    primary_delta = next(d for d in verdict.deltas if d.model_name == "Simple MOND")
    assert primary_delta.delta_nmse < 0  # ADCD wins


def test_analyse_cv_missing_adcd_key_raises():
    with pytest.raises(ValueError, match="Required CV key"):
        analyse_cv({"X": {"mean_nmse": 0.4, "std_error": 0.01}})


def test_analyse_cv_missing_competitor_key_raises():
    cv = {"ADCD discovered": {"mean_nmse": 0.4, "std_error": 0.01}}
    with pytest.raises(ValueError, match="Required CV key"):
        analyse_cv(cv, primary_competitor_key="Simple MOND (2-param)")


def test_analyse_cv_malformed_entry_raises():
    cv = {
        "ADCD discovered": {"mean_nmse": 0.4, "std_error": 0.01},
        "Simple MOND (2-param)": {"mean_nmse": 0.4},  # missing std_error
    }
    with pytest.raises(ValueError, match="missing 'mean_nmse'/'std_error'"):
        analyse_cv(cv)


# ---------------------------------------------------------------------------
# Serialisation + driver
# ---------------------------------------------------------------------------

def test_to_dict_roundtrip_jsonable():
    """Verdict serialises cleanly to JSON (no dataclasses leak through)."""
    verdict = analyse_cv(_real_cv())
    d = verdict.to_dict()
    s = json.dumps(d)
    back = json.loads(s)
    assert back["scenario"] == "B"
    assert back["adcd"]["mean_nmse"] == pytest.approx(ADCD_CV)
    assert any(item["model_name"] == "Simple MOND (2-param)" for item in back["deltas"])


def test_analyse_cv_from_results_json(tmp_path):
    """End-to-end: read a synthetic sparc_discovery.json, write verdict JSON."""
    payload = {"cross_validation": _real_cv()}
    src = tmp_path / "sparc_discovery.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "mond_cv_2param.json"

    verdict = analyse_cv_from_results(str(src), str(out))
    assert isinstance(verdict, CVMVerdict)
    assert verdict.scenario == "B"
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["scenario"] == "B"
    assert "reviewer_message" in written


def test_analyse_cv_from_results_missing_block_raises(tmp_path):
    """A discovery JSON with no cross_validation block is a clear error."""
    src = tmp_path / "no_cv.json"
    src.write_text(json.dumps({"final_nmse": 0.4}), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'cross_validation' block"):
        analyse_cv_from_results(str(src))


def test_print_report_smoke(capsys):
    """print_cv_report emits a readable table without error."""
    verdict = analyse_cv(_real_cv())
    print_cv_report(verdict)
    captured = capsys.readouterr().out
    assert "CV Analysis" in captured
    assert "Scenario B" in captured
    assert "ADCD discovered" in captured
    assert "Reviewer message" in captured
