"""
tests/test_real_world_sparc.py
==============================
Regression tests for Real-World SPARC Galaxy Rotation Curve Identifiability Audit.

Verifies:
1. Parsing and validation of official 2,693 astronomical data points from SPARC (RAR.mrt).
2. Detection gate: Newtonian baseline is decisively rejected (Delta-BIC > 10,000).
3. Structural validity: Crossing-pole diagnostic detects baseline crossing on real observational data.
4. Competitive model ranking: McGaugh RAR law decisively beats rival families (Delta-BIC > 50).
"""

from __future__ import annotations

import os
import pytest
from scripts.audit_sparc_rar import (
    DATA_PATH,
    load_sparc_rar_data,
    evaluate_all_galaxy_models,
)
from adcd.validation_v2 import run_three_tier_audit, ProtocolVerdict
from spike.crossover import check_convex_validity
import numpy as np


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="SPARC data file not downloaded")
def test_sparc_data_loading_and_properties() -> None:
    data = load_sparc_rar_data(DATA_PATH)
    n_pts = len(data["x"])
    assert n_pts == 2693, f"Expected 2693 SPARC data points, got {n_pts}"
    assert np.all(data["x"] > 0), "All acceleration ratios must be strictly positive"
    assert np.all(data["y"] > 0), "All boost factors must be strictly positive"
    assert np.median(data["sigma_dex"]) > 0.05, "Uncertainty scale must be non-trivial"


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="SPARC data file not downloaded")
def test_sparc_structural_convex_validity() -> None:
    data = load_sparc_rar_data(DATA_PATH)
    A = np.ones_like(data["x"])
    B = data["x"] ** (-0.5)
    val_report = check_convex_validity(data["x"], data["y"], A, B)
    assert not val_report.is_valid, "Convex crossover must be invalid on SPARC coordinates"
    assert val_report.has_crossing_pole, "Crossing pole must be flagged"
    assert val_report.w_req_min < -100.0, "Empirical weight must blow up into large negative values"


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="SPARC data file not downloaded")
def test_sparc_audit_verdict_and_ranks() -> None:
    data = load_sparc_rar_data(DATA_PATH)
    fits = evaluate_all_galaxy_models(data)

    null_fit = next(f for f in fits if f.family == "family_null_newton")
    candidates = [f for f in fits if f.family != "family_null_newton"]

    audit_res = run_three_tier_audit(
        candidates=candidates,
        null_bic=null_fit.bic,
        search_space_size=len(candidates),
        check_adequacy=False,
    )

    # 1. Detection gate: Null rejected decisively
    assert audit_res.delta_bic_vs_null > 10000.0, "Newtonian gravity must be rejected by massive BIC margin"

    # 2. Winner is McGaugh RAR
    assert audit_res.best_candidate is not None
    assert audit_res.best_candidate.family == "family_mcgaugh_rar"

    # 3. Delta-BIC vs closest rival (Simple MOND) > 50
    assert audit_res.delta_bic_vs_rival > 50.0
    assert audit_res.verdict == ProtocolVerdict.UNIQUELY_IDENTIFIABLE
