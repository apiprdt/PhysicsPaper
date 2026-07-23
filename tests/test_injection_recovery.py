"""Unit tests for InjectionRecoveryEvaluator."""

from adcd.injection_recovery import InjectionRecoveryEvaluator


def test_positive_injection_recovery():
    evaluator = InjectionRecoveryEvaluator()
    res = evaluator.run_positive_recovery(noise_level=0.02, n_points=100, seed=42)
    assert res.passed
    assert res.test_type == "positive_recovery"
    assert res.nmse < 0.15


def test_negative_control_injection_recovery():
    evaluator = InjectionRecoveryEvaluator()
    res = evaluator.run_negative_control(noise_level=0.05, n_points=100, seed=42)
    assert res.passed
    assert res.test_type == "negative_control"
