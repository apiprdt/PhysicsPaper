"""Unit tests for ExtendedGrammarProposer."""

from adcd.extended_grammar import ExtendedGrammarProposer


def test_extended_grammar_pattern_count():
    proposer = ExtendedGrammarProposer()
    assert proposer.get_pattern_count() >= 6


def test_extended_grammar_candidate_generation():
    proposer = ExtendedGrammarProposer()
    candidates = proposer.propose_candidates(["r", "v"])
    assert len(candidates) >= 12
    # Verify each candidate has required fields
    for cand in candidates:
        assert "expr" in cand
        assert "justification" in cand
        assert "domain" in cand
        assert len(cand["justification"]) > 10


def test_multi_proposer_integration():
    from adcd.llm_proposer import MockProposer, ProposalContext
    from adcd.extended_grammar import MultiProposer

    multi = MultiProposer([MockProposer(), ExtendedGrammarProposer()])
    ctx = ProposalContext(variable_names=["r"], target_name="y", data_statistics={})
    cands = multi.propose(ctx)
    assert len(cands) > 6
    # Verify candidates contain expressions from both MockProposer and ExtendedGrammarProposer
    joined = " ".join(cands)
    assert "erf" in joined or "exp" in joined
