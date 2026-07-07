"""
test_llm_rich_context.py
========================
Tests for LLM prompt template enrichment via _build_rich_context_strings.

Covers:
- Percentile statistics injection with unit context
- Leading exponent hint injection as standalone bullet (no leading newline)
- Graceful no-op when residual_data is None
- Graceful no-op when residual_features has no leading_exponent
- Both GeminiProposer and CorrectionGeminiProposer inject rich context
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from adcd.llm_proposer import (
    GeminiProposer,
    CorrectionGeminiProposer,
    ProposalContext,
    _build_rich_context_strings,
)


def _make_context(residual_data=None, leading_exponent=None, target_name="y"):
    """Helper to construct a ProposalContext with optional rich fields."""
    rf = None
    if leading_exponent is not None:
        rf = MagicMock()
        rf.leading_exponent = leading_exponent

    rng = np.random.RandomState(42)
    if residual_data is None:
        residual_data = None  # explicitly absent

    return ProposalContext(
        variable_names=["x"],
        target_name=target_name,
        data_statistics={},
        domain="electromagnetism",
        classical_expr="theta_0 * x",
        anomaly_description="non-linear scaling at high x",
        residual_data=residual_data,
        residual_features=rf,
    )


class TestBuildRichContextStrings:
    """Unit tests for the module-level helper function directly."""

    def test_percentile_str_with_data_and_units(self):
        """Percentile string must include 5 quantiles and target unit context."""
        data = 2.0 * np.linspace(0.1, 1.0, 100)
        ctx = _make_context(residual_data=data, target_name="velocity")
        pct_str, hint_str = _build_rich_context_strings(ctx)

        assert "5th:" in pct_str
        assert "25th:" in pct_str
        assert "50th:" in pct_str
        assert "75th:" in pct_str
        assert "95th:" in pct_str
        # Unit context should appear
        assert "[velocity units]" in pct_str

    def test_percentile_str_none_when_no_data(self):
        """Without residual_data, percentile_str should be the literal 'None'."""
        ctx = _make_context(residual_data=None)
        pct_str, _ = _build_rich_context_strings(ctx)
        assert pct_str == "None"

    def test_exponent_hint_str_present_and_no_leading_newline(self):
        """exponent_hint_str must NOT start with a newline (formatting regression test)."""
        ctx = _make_context(leading_exponent=2.14)
        _, hint_str = _build_rich_context_strings(ctx)

        assert hint_str != ""
        assert not hint_str.startswith("\n"), (
            "exponent_hint_str must not start with newline — "
            "callers are responsible for their own line separators."
        )
        assert "2.14" in hint_str
        assert "NAAE" in hint_str  # should reference the estimator

    def test_exponent_hint_str_empty_when_no_features(self):
        """Without residual_features, exponent_hint_str must be empty string."""
        ctx = _make_context(leading_exponent=None)
        _, hint_str = _build_rich_context_strings(ctx)
        assert hint_str == ""

    def test_exponent_hint_str_empty_when_exponent_is_none(self):
        """If leading_exponent attribute is None, exponent_hint_str must be ''."""
        rf = MagicMock()
        rf.leading_exponent = None
        ctx = _make_context()
        ctx = ProposalContext(
            variable_names=["x"],
            target_name="y",
            data_statistics={},
            residual_features=rf,
        )
        _, hint_str = _build_rich_context_strings(ctx)
        assert hint_str == ""


class TestGeminiProposerRichContext:
    """Integration tests — verify rich context appears in actual prompt output."""

    def test_full_rich_context_injected(self):
        """All rich context fields appear in GeminiProposer prompt with data."""
        data = 2.0 * np.linspace(0.1, 1.0, 100) + 0.01 * np.random.randn(100)
        ctx = _make_context(residual_data=data, leading_exponent=2.14, target_name="force")

        proposer = GeminiProposer(api_key="dummy_key")
        prompt = proposer.get_prompt_template(ctx)

        assert "Absolute residual magnitude percentiles:" in prompt
        assert "5th:" in prompt
        assert "95th:" in prompt
        assert "[force units]" in prompt
        assert "2.14" in prompt
        assert "NAAE" in prompt

    def test_no_hint_when_no_features(self):
        """Prompt should not contain NAAE hint when residual_features is None."""
        data = np.linspace(0.1, 1.0, 50)
        ctx = _make_context(residual_data=data, leading_exponent=None)

        proposer = GeminiProposer(api_key="dummy_key")
        prompt = proposer.get_prompt_template(ctx)

        assert "Absolute residual magnitude percentiles:" in prompt
        # No exponent hint should appear
        assert "NAAE" not in prompt
        assert "Leading scaling exponent hint" not in prompt


class TestCorrectionGeminiProposerRichContext:
    """Verify CorrectionGeminiProposer also injects rich context."""

    def test_percentile_and_hint_in_correction_prompt(self):
        """CorrectionGeminiProposer must include percentiles and exponent hint."""
        data = np.linspace(0.05, 0.5, 80)
        ctx = _make_context(residual_data=data, leading_exponent=1.75, target_name="omega")

        proposer = CorrectionGeminiProposer(api_key="dummy_key")
        prompt = proposer.get_prompt_template(ctx)

        assert "Absolute residual magnitude percentiles:" in prompt
        assert "1.75" in prompt
        assert "NAAE" in prompt
