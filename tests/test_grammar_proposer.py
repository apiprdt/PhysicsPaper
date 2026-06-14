import pytest
import sympy as sp
from adcd.grammar_proposer import GrammarProposer
from adcd.llm_proposer import ProposalContext

def test_grammar_proposer_basics():
    proposer = GrammarProposer(seed=123)
    
    context = ProposalContext(
        variable_names=["m", "v"],
        target_name="residual",
        data_statistics={
            "m": {"mean": 2.0, "std": 1.0, "min": 0.5, "max": 4.0},
            "v": {"mean": 3.0, "std": 1.5, "min": 0.5, "max": 6.0}
        },
        n_candidates=30,
        iteration=0,
        constants={"c": 3e8}
    )
    
    candidates = proposer.propose(context)
    
    assert len(candidates) > 0
    assert len(candidates) <= 30
    assert len(set(candidates)) == len(candidates)  # Deduplicated
    
    # Check that they parse in SymPy
    for cand in candidates:
        expr = sp.sympify(cand)
        assert isinstance(expr, sp.Expr)
        # All free parameters must be standardized (theta_0, theta_1, etc.)
        thetas = [str(sym) for sym in expr.free_symbols if str(sym).startswith("theta")]
        for th in thetas:
            assert th.startswith("theta_")
            idx = int(th.split("_")[1])
            assert idx >= 0

def test_grammar_proposer_different_ratios():
    proposer = GrammarProposer(seed=456)
    
    # With variables that can form a Buckingham-pi ratio (m and M)
    context = ProposalContext(
        variable_names=["m", "M", "r"],
        target_name="residual",
        data_statistics={},
        n_candidates=50,
        iteration=0,
        constants={}
    )
    
    candidates = proposer.propose(context)
    
    # Find candidates containing m/M or M/m
    ratio_found = False
    for cand in candidates:
        if "m" in cand and "M" in cand:
            ratio_found = True
            break
            
    assert ratio_found, "GrammarProposer should discover and use dimensionless ratio m/M or M/m"
