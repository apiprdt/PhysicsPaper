"""
Unit tests for ARCGate module.
"""

import pytest
import sympy as sp
from adcd.arc_gate import ARCGate, ARCCheckResult


def test_arc_passes_for_valid_decay_correction():
    gate = ARCGate(target_limit=0.0, tolerance=1e-3)
    # delta(x) = exp(-x) -> lim_{x->oo} delta(x) = 0
    res = gate.check(
        candidate_expr="exp(-x1)",
        limit_vars=["x1"],
        limit_points=[sp.oo]
    )
    assert res.passed is True
    assert res.rejection_reason is None


def test_arc_rejects_nonzero_limit():
    gate = ARCGate(target_limit=0.0, tolerance=1e-3)
    # delta(x) = 1 + exp(-x) -> lim_{x->oo} delta(x) = 1 != 0
    res = gate.check(
        candidate_expr="1 + exp(-x1)",
        limit_vars=["x1"],
        limit_points=[sp.oo]
    )
    assert res.passed is False
    assert "ARC violation" in res.rejection_reason


def test_arc_passes_at_zero_limit():
    gate = ARCGate(target_limit=0.0, tolerance=1e-3)
    # delta(x) = x1^2 -> lim_{x->0} delta(x) = 0
    res = gate.check(
        candidate_expr="x1**2",
        limit_vars=["x1"],
        limit_points=[0]
    )
    assert res.passed is True
    assert res.limit_value == 0.0


def test_arc_handles_invalid_sympy_string():
    gate = ARCGate()
    res = gate.check(
        candidate_expr="x1 +++ / invalid",
        limit_vars=["x1"]
    )
    assert res.passed is False
    assert "SymPy parse error" in res.rejection_reason


def test_arc_handles_parameters():
    gate = ARCGate(target_limit=0.0, tolerance=1e-3)
    # theta_0 * exp(-theta_1 * x1) -> limit is 0 as x1 -> oo
    res = gate.check(
        candidate_expr="theta_0 * exp(-theta_1 * x1)",
        limit_vars=["x1"],
        limit_points=[sp.oo]
    )
    assert res.passed is True
