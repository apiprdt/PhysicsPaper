import numpy as np
import pytest
from unittest.mock import MagicMock
from adcd.llm_proposer import GeminiProposer, ProposalContext

def test_gemini_proposer_rich_context():
    """Verify GeminiProposer injects percentile statistics and exponent hints into prompt template."""
    # Synthesize dummy data
    rng = np.random.RandomState(42)
    residual_data = 2.0 * np.linspace(0.1, 1.0, 100) + 0.01 * rng.randn(100)
    
    # Mock residual features with a leading exponent
    rf = MagicMock()
    rf.leading_exponent = 2.14
    
    context = ProposalContext(
        variable_names=["x"],
        target_name="y",
        data_statistics={},
        domain="electromagnetism",
        classical_expr="theta_0 * x",
        anomaly_description="non-linear scaling at high x",
        residual_data=residual_data,
        residual_features=rf,
    )
    
    proposer = GeminiProposer(api_key="dummy_key")
    prompt = proposer.get_prompt_template(context)
    
    # Assert rich context fields are injected in the prompt template
    assert "Absolute residual magnitude percentiles:" in prompt
    assert "5th:" in prompt
    assert "95th:" in prompt
    assert "Leading scaling exponent hint:" in prompt
    assert "2.14" in prompt
