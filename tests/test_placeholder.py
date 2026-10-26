"""Core unit tests for ADCD primitives, ratio enumeration, physical gates, and determinism."""

import numpy as np
import sympy as sp

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget, PRIMITIVE_REGISTRY
from adcd.context import ProposalContext
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.mode_detection import detect_correction_mode
from adcd.pipeline import Stage1Pipeline
from adcd.quickfit import DOMAIN_TAXONOMY, _generate_ratio_symbols


def test_primitives_vanish_at_origin_and_d_lor_stability():
    """All 8 primitives satisfy D(0) = 0; D_lor avoids float64 cancellation at u = 1e-16."""
    assert len(PRIMITIVE_REGISTRY) == 8
    for name, prim in PRIMITIVE_REGISTRY.items():
        val0 = float(prim.numpy_form(np.array([0.0]))[0])
        assert abs(val0) < 1e-15, f"{name} did not vanish at u=0: {val0}"

    u_tiny = np.array([1e-16], dtype=np.float64)
    d_lor_val = float(PRIMITIVE_REGISTRY["D_lor"].numpy_form(u_tiny)[0])
    d_sqrt_val = float(PRIMITIVE_REGISTRY["D_sqrt_inv"].numpy_form(u_tiny)[0])
    d_osc_val = float(PRIMITIVE_REGISTRY["D_osc"].numpy_form(np.array([1e-8], dtype=np.float64))[0])
    assert np.isclose(d_lor_val, 5.0e-17, rtol=1e-2)
    assert np.isclose(d_sqrt_val, -5.0e-17, rtol=1e-2)
    assert np.isclose(d_osc_val, 5.0e-17, rtol=1e-2)


def test_ratio_enumeration_and_search_space_sizes():
    """Verify exact |C_domain| = 60/40/10, |C_ablated| = 814/534/700, and r/theta_4 index."""
    scenarios = {s.name: s for s in get_all_scenarios()}

    sc = scenarios["Screened Coulomb"]
    sc_ratios = _generate_ratio_symbols(sc.variables_with_units, max_ratios=12)
    assert len(sc_ratios) == 12
    assert sc_ratios[8] == "r/theta_4"

    expected = {
        "Screened Coulomb": (60, 814, "D_exp"),
        "Entropy Expansion": (40, 534, "D_log"),
        "Time Dilation": (10, 700, "D_lor"),
    }

    for name, (exp_dom, exp_abl, true_prim) in expected.items():
        scen = scenarios[name]
        ctx = ProposalContext(
            variable_names=scen.classical_variables,
            target_name="residual",
            data_statistics={},
            n_candidates=1000,
            constants=scen.classical_constants,
            variables_with_units=scen.variables_with_units,
        )
        allowed = set(DOMAIN_TAXONOMY[scen.domain])
        dom_exclude = [p for p in PRIMITIVE_REGISTRY if p not in allowed]
        prop_dom = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=dom_exclude,
        )
        assert prop_dom.search_space_size(ctx) == exp_dom

        prop_abl = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=[true_prim],
        )
        assert prop_abl.search_space_size(ctx) == exp_abl


def test_stage1_gate_cascade_and_determinism():
    """Verify Stage 1 gate cascade filters invalid expressions and is deterministic."""
    checker = DimensionalChecker()
    validator = ASTValidator(max_depth=12, max_tokens=50)
    scorer = ARCScorer(regimes=build_arc_regimes("r", "0"))
    pipeline = Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)

    cands = [
        ("theta_0 * (exp(-r/theta_4) - 1.0)", True),
        ("exp(r)", False),  # fails transcendental dimension check (r has length dimension)
    ]
    res1 = pipeline.execute(cands, target_dimension_key=None)
    res2 = pipeline.execute(cands, target_dimension_key=None)
    assert res1 == res2
    assert len(res1) == 1
    assert res1[0][0] == "theta_0 * (exp(-r/theta_4) - 1.0)"


def test_mode_detection_determinism_and_constant_baseline_fallback():
    """Verify detect_correction_mode is deterministic and falls back to additive on constant baseline."""
    y_cl_const = np.full(50, 15.0)
    y_obs = y_cl_const + 8.314 * np.log1p(np.linspace(0.01, 1.0, 50))
    mode1, conf1 = detect_correction_mode(y_obs, y_cl_const)
    mode2, conf2 = detect_correction_mode(y_obs, y_cl_const)
    assert mode1 == mode2 == "additive"
    assert conf1 == conf2 == 0.5
