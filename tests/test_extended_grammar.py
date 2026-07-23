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
