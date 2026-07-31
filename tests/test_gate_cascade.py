"""
Unit tests for PhysicsGateCascade module.
"""

import pytest
import sympy as sp
from adcd.gate_cascade import PhysicsGateCascade, GateResult


def test_cascade_stage0_parse_error():
    cascade = PhysicsGateCascade()
    res = cascade.check("invalid ((( syntax")
    assert res.passed is False
    assert res.rejected_stage == 0


def test_cascade_stage1_depth_violation():
    cascade = PhysicsGateCascade(max_depth=3)
    # Deep nested expression: exp(exp(exp(exp(x1)))) -> depth > 3
    res = cascade.check("exp(exp(exp(exp(x1))))")
    assert res.passed is False
    assert res.rejected_stage == 1
    assert "Depth Violation" in res.rejection_reason


def test_cascade_stage1_token_violation():
    cascade = PhysicsGateCascade(max_tokens=5)
    # Token count > 5
    res = cascade.check("x1 + x2 + x3 + x4 + x5 + x6")
    assert res.passed is False
    assert res.rejected_stage == 1
    assert "Token Violation" in res.rejection_reason


def test_cascade_stage3_arc_violation():
    cascade = PhysicsGateCascade(
        max_depth=7,
        max_tokens=20,
        enable_dimensional_check=False,
        enable_arc_check=True
    )
    # Candidate with non-zero limit
    res = cascade.check(
        candidate_expr="5.0 + x1",
        limit_vars=["x1"],
        limit_points=[0]
    )
    assert res.passed is False
    assert res.rejected_stage == 3
    assert "ARC Violation" in res.rejection_reason


def test_cascade_all_stages_passed():
    cascade = PhysicsGateCascade(
        max_depth=7,
        max_tokens=20,
        enable_dimensional_check=False,
        enable_arc_check=True
    )
    # Valid decay candidate
    res = cascade.check(
        candidate_expr="x1**2",
        limit_vars=["x1"],
        limit_points=[0]
    )
    assert res.passed is True
    assert res.rejected_stage is None
    assert res.rejection_reason is None
