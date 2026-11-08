"""
Tests for Phase 1: ClaimCard schema, canonical hashing, audit API, and imported custom formulas.
"""

import numpy as np
import pytest

from adcd.claim_card import ClaimCard, NoiseSpec
from adcd.audit import audit, audit_custom_expressions, verify_classical_limit


def test_claim_card_json_roundtrip():
    """ClaimCard can be serialized to JSON and deserialized without data loss."""
    card = ClaimCard(
        name="Screened Coulomb Test",
        variables={"r": "m", "q1": "C", "q2": "C"},
        baseline_expr="k_e * q1 * q2 / r",
        baseline_constants={"k_e": 8.98755e9},
        limit_variable="r",
        limit_direction="0",
        window={"r": (0.2, 4.0)},
        mode="multiplicative",
        candidate_expressions=["exp(-r/theta_0) - 1.0", "r/(1.0 - r)"],
        noise_spec=NoiseSpec(type="relative", value=0.01),
        seed=42,
    )

    json_str = card.to_json()
    assert isinstance(json_str, str)

    card_loaded = ClaimCard.from_json(json_str)
    assert card_loaded.name == card.name
    assert card_loaded.variables == card.variables
    assert card_loaded.baseline_expr == card.baseline_expr
    assert card_loaded.baseline_constants == card.baseline_constants
    assert card_loaded.limit_variable == card.limit_variable
    assert card_loaded.limit_direction == card.limit_direction
    assert card_loaded.window == card.window
    assert card_loaded.mode == card.mode
    assert card_loaded.candidate_expressions == card.candidate_expressions
    assert card_loaded.noise_spec.value == 0.01
    assert card_loaded.seed == 42


def test_claim_card_canonical_hash_deterministic():
    """Identical claim cards produce byte-exact identical hashes regardless of field insertion order."""
    card1 = ClaimCard(
        name="Card A",
        variables={"q1": "C", "r": "m"},
        baseline_expr="k_e * q1 / r",
        limit_variable="r",
        limit_direction="0",
        seed=42,
    )

    card2 = ClaimCard(
        name="Card A",
        variables={"r": "m", "q1": "C"},
        baseline_expr="k_e * q1 / r",
        limit_variable="r",
        limit_direction="0",
        seed=42,
    )

    h1 = card1.canonical_hash()
    h2 = card2.canonical_hash()
    assert len(h1) == 64  # SHA-256
    assert h1 == h2, f"Hashes must be identical: {h1} != {h2}"


def test_audit_deterministic_report_hash():
    """audit(card, data) produces identical reports and report hashes across identical runs."""
    card = ClaimCard(
        name="Screened Coulomb Audit",
        variables={"r": "m"},
        baseline_expr="1.0 / r",
        limit_variable="r",
        limit_direction="0",
        window={"r": (0.2, 4.0)},
        mode="multiplicative",
        candidate_expressions=["exp(-r/1.5) - 1.0", "r/(1.0 - r)"],
        noise_spec=NoiseSpec(type="relative", value=0.01),
        seed=42,
    )

    # Synthetic dataset
    rng = np.random.RandomState(42)
    r = np.linspace(0.2, 4.0, 100)
    y_cl = 1.0 / r
    delta = np.exp(-r / 1.5) - 1.0
    y_obs = y_cl * (1.0 + delta) * (1.0 + rng.normal(0, 0.01, size=len(r)))
    data = {"r": r, "y_obs": y_obs, "y_cl": y_cl}

    rep1 = audit(card, data)
    rep2 = audit(card, data)

    assert rep1.card_hash == rep2.card_hash
    assert rep1.report_hash == rep2.report_hash
    assert rep1.verdict == rep2.verdict
    assert rep1.best_candidate == rep2.best_candidate


def test_audit_custom_expression_valid_limit():
    """audit_custom_expressions verifies asymptotic limit both symbolically and numerically."""
    # A valid correction: u/(1+u) -> 0 as u -> 0
    valid_expr = "u / (1.0 + u)"
    status = verify_classical_limit(valid_expr, limit_var="u", limit_dir="0")
    assert status["valid"] is True
    assert status["symbolic_verified"] is True
    assert status["numerical_verified"] is True
    assert "limit verified, not enforced by construction" in status["message"]


def test_audit_custom_expression_invalid_limit_rejected():
    """An imported formula that does not vanish at the classical limit is rejected with a clear message."""
    # Invalid: cos(u) -> 1 != 0 as u -> 0
    invalid_expr_1 = "cos(u)"
    status_1 = verify_classical_limit(invalid_expr_1, limit_var="u", limit_dir="0")
    assert status_1["valid"] is False
    assert "Classical limit violated" in status_1["message"]
    assert "does not vanish" in status_1["message"]

    # Invalid: u + 0.5 -> 0.5 != 0
    invalid_expr_2 = "u + 0.5"
    status_2 = verify_classical_limit(invalid_expr_2, limit_var="u", limit_dir="0")
    assert status_2["valid"] is False
    assert "Classical limit violated" in status_2["message"]

    # When auditing a card with an invalid imported candidate, it must be rejected
    card = ClaimCard(
        name="Invalid Candidate Test",
        variables={"x": "dimensionless"},
        baseline_expr="1.0",
        limit_variable="x",
        limit_direction="0",
        candidate_expressions=["cos(x)"],  # Does not vanish at x=0
        seed=42,
    )
    with pytest.raises(ValueError, match="Classical limit violated"):
        audit_custom_expressions(card, data={"x": np.array([0.1, 0.2]), "y_obs": np.array([1.0, 1.0])})


def test_plain_language_report_semantics():
    """Audit report contains explicit plain-language statements of what verdict means and does NOT mean."""
    card = ClaimCard(
        name="Semantics Check",
        variables={"x": "dimensionless"},
        baseline_expr="1.0",
        limit_variable="x",
        limit_direction="0",
        candidate_expressions=["exp(-x) - 1.0"],
        seed=42,
    )
    x = np.linspace(0.01, 1.0, 50)
    data = {"x": x, "y_obs": np.exp(-x), "y_cl": np.ones_like(x)}
    rep = audit(card, data)

    # Must contain plain language explanations
    summary = rep.plain_language_summary
    assert "WHAT THIS VERDICT MEANS:" in summary
    assert "WHAT THIS VERDICT DOES NOT MEAN:" in summary
    assert "claim card" in summary.lower() or "card hash" in summary.lower()
