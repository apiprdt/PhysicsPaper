"""Tests for multi-variable ARC regime construction."""

import sympy as sp

from adcd.arc_scorer import build_arc_regimes, ARCScorer


def test_build_arc_regimes_single_variable():
    regimes = build_arc_regimes("v", "0")
    assert len(regimes) == 1
    assert regimes[0].variable == sp.Symbol("v")
    assert regimes[0].limit_target == 0


def test_build_arc_regimes_multi_variable():
    regimes = build_arc_regimes("x, y", "0, oo")
    assert len(regimes) == 2
    assert regimes[0].variable == sp.Symbol("x")
    assert regimes[0].limit_target == 0
    assert regimes[1].variable == sp.Symbol("y")
    assert regimes[1].limit_target == sp.oo


def test_arc_scorer_multi_variable_correction():
    """Δ = x/y vanishes when x→0 and when y→∞."""
    regimes = build_arc_regimes("x, y", "0, oo")
    scorer = ARCScorer(regimes=regimes)
    score = scorer.score("x / y")
    assert score == 1.0


def test_discover_correction_auto_mode():
    import adcd

    scenarios = adcd.get_all_scenarios()
    scenario = [s for s in scenarios if s.name == "Relativistic KE"][0]
    result = adcd.discover_correction(
        scenario,
        max_iterations=1,
        proposer="mock",
        correction_mode="auto",
        seed=42,
        verbose=False,
    )
    assert result.scenario.correction_type in ("additive", "multiplicative")
