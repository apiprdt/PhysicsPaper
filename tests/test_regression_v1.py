"""
Regression tests for ADCD v1 paper baseline reproduction.
Verifies reference metrics for seed 42 and across 5 independent seeds (0, 42, 99, 256, 512).
"""

import json
import math
import os
import numpy as np
import pytest

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import run_scenario_protocol


def test_reference_five_seed_stability_table():
    """Verify golden multi-seed results match the published paper metrics."""
    results_path = os.path.join(
        os.path.dirname(__file__), "..", "rebuttal", "results", "multi_seed_stability.json"
    )
    assert os.path.exists(results_path), f"Missing golden dataset: {results_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_dbic = {
        "Screened Coulomb": [56.25, 30.65, 263.16, 11.02, 21.94],
        "Entropy Expansion": [9.50, 14.70, 5.12, 9.91, 10.34],
        "Time Dilation": [8.35, 13.79, 5.49, 8.01, 13.73],
    }

    expected_seed_42 = {
        "Screened Coulomb": {
            "nmse": 2.77e-4,
            "delta_bic": 30.65,
            "cand_dom": 60,
            "cand_abl": 814,
            "k_dom": 2,
            "k_abl": 3,
            "all_passed": True,
        },
        "Entropy Expansion": {
            "nmse": 1.59e-2,
            "delta_bic": 14.70,
            "cand_dom": 40,
            "cand_abl": 534,
            "k_dom": 1,
            "k_abl": 3,
            "all_passed": True,
        },
        "Time Dilation": {
            "nmse": 0.412,
            "delta_bic": 13.79,
            "cand_dom": 10,
            "cand_abl": 700,
            "k_dom": 1,
            "k_abl": 2,
            "all_passed": False,
        },
    }

    for scenario_name, exp_seeds in expected_dbic.items():
        assert scenario_name in data, f"Scenario {scenario_name} missing from multi_seed_stability.json"
        per_seed = data[scenario_name]["per_seed"]
        actual_seeds = [s["seed"] for s in per_seed]
        assert actual_seeds == [0, 42, 99, 256, 512]

        actual_dbics = [round(s["delta_bic"], 2) for s in per_seed]
        np.testing.assert_allclose(
            actual_dbics, exp_seeds, atol=0.05,
            err_msg=f"Mismatch in 5-seed ΔBIC for {scenario_name}"
        )

        # Check Seed 42 details
        seed_42_entry = next(s for s in per_seed if s["seed"] == 42)
        exp_42 = expected_seed_42[scenario_name]
        assert np.isclose(seed_42_entry["nmse_best"], exp_42["nmse"], rtol=0.02)
        assert np.isclose(seed_42_entry["delta_bic"], exp_42["delta_bic"], atol=0.1)
        assert seed_42_entry["n_cand_domain"] == exp_42["cand_dom"]
        assert seed_42_entry["n_cand_ablated"] == exp_42["cand_abl"]
        assert seed_42_entry["k_domain"] == exp_42["k_dom"]
        assert (seed_42_entry["verdict"] == "IDENTIFIABLE") == exp_42["all_passed"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_live_pipeline_reproduction_screened_coulomb():
    """Execute live pipeline on Screened Coulomb seed 42 to verify live reproducibility."""
    scenarios = {s.name: s for s in get_all_scenarios()}
    sc = scenarios["Screened Coulomb"]
    
    result = run_scenario_protocol(sc, top_k_val=5, use_taxonomy_prior=True, seed=42)
    assert result.all_passed is True

    # Check primary search
    ps = result.checks["primary_search"]
    assert np.isclose(ps["nmse"], 2.77e-4, rtol=0.05)
    assert ps["discovered_class"] == "exponential"

    # Check search space budget
    bd = result.checks["budget_disclosure"]
    assert bd["search_space_size"] == 60

    # Check positive control
    pc = result.checks["positive_control"]
    assert pc["pass"] is True
    assert pc["nmse"] < 0.05

    # Check ablation control
    ac = result.checks["ablation_control"]
    assert ac["pass"] is True
    assert np.isclose(ac["bic_diff"], 30.65, atol=0.25)
