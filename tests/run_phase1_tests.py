"""
Phase 1 Unit Test Runner for ARCGate and PhysicsGateCascade.
Uses standard Python unittest framework.
"""

import sys
import os
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
from adcd.arc_gate import ARCGate
from adcd.gate_cascade import PhysicsGateCascade


class TestARCGate(unittest.TestCase):

    def test_arc_passes_for_valid_decay_correction(self):
        gate = ARCGate(target_limit=0.0, tolerance=1e-3)
        res = gate.check(
            candidate_expr="exp(-x1)",
            limit_vars=["x1"],
            limit_points=[sp.oo]
        )
        self.assertTrue(res.passed)
        self.assertIsNone(res.rejection_reason)

    def test_arc_rejects_nonzero_limit(self):
        gate = ARCGate(target_limit=0.0, tolerance=1e-3)
        res = gate.check(
            candidate_expr="1 + exp(-x1)",
            limit_vars=["x1"],
            limit_points=[sp.oo]
        )
        self.assertFalse(res.passed)
        self.assertIn("ARC violation", res.rejection_reason)

    def test_arc_passes_at_zero_limit(self):
        gate = ARCGate(target_limit=0.0, tolerance=1e-3)
        res = gate.check(
            candidate_expr="x1**2",
            limit_vars=["x1"],
            limit_points=[0]
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.limit_value, 0.0)

    def test_arc_handles_invalid_sympy_string(self):
        gate = ARCGate()
        res = gate.check(
            candidate_expr="x1 +++ / invalid",
            limit_vars=["x1"]
        )
        self.assertFalse(res.passed)
        self.assertIn("SymPy parse error", res.rejection_reason)


class TestGateCascade(unittest.TestCase):

    def test_cascade_stage0_parse_error(self):
        cascade = PhysicsGateCascade()
        res = cascade.check("invalid ((( syntax")
        self.assertFalse(res.passed)
        self.assertEqual(res.rejected_stage, 0)

    def test_cascade_stage1_depth_violation(self):
        cascade = PhysicsGateCascade(max_depth=3)
        res = cascade.check("exp(exp(exp(exp(x1))))")
        self.assertFalse(res.passed)
        self.assertEqual(res.rejected_stage, 1)
        self.assertIn("Depth Violation", res.rejection_reason)

    def test_cascade_stage1_token_violation(self):
        cascade = PhysicsGateCascade(max_tokens=5)
        res = cascade.check("x1 + x2 + x3 + x4 + x5 + x6")
        self.assertFalse(res.passed)
        self.assertEqual(res.rejected_stage, 1)
        self.assertIn("Token Violation", res.rejection_reason)

    def test_cascade_stage3_arc_violation(self):
        cascade = PhysicsGateCascade(
            max_depth=7,
            max_tokens=20,
            enable_dimensional_check=False,
            enable_arc_check=True
        )
        res = cascade.check(
            candidate_expr="5.0 + x1",
            limit_vars=["x1"],
            limit_points=[0]
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.rejected_stage, 3)

    def test_cascade_all_stages_passed(self):
        cascade = PhysicsGateCascade(
            max_depth=7,
            max_tokens=20,
            enable_dimensional_check=False,
            enable_arc_check=True
        )
        res = cascade.check(
            candidate_expr="x1**2",
            limit_vars=["x1"],
            limit_points=[0]
        )
        self.assertTrue(res.passed)
        self.assertIsNone(res.rejected_stage)


if __name__ == "__main__":
    unittest.main()
