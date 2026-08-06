from typing import List, Optional, Dict
from adcd.context import BaseProposer, ProposalContext
from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget, enumerate_candidates, PRIMITIVE_REGISTRY
from adcd.quickfit import _generate_ratio_symbols

class GrammarProposerV3(BaseProposer):
    """
    Fully blind Grammar Proposer. Takes physical variables, derives Buckingham-Pi
    dimensionless ratios, and enumerates regularized primitive compositions over
    ALL those ratios simultaneously.
    """
    def __init__(
        self,
        budget: Optional[GrammarBudget] = None,
        exclude_primitives: Optional[List[str]] = None,
        dimensional_checker = None,
        variables_with_units: Optional[Dict[str, str]] = None,
    ):
        self.budget = budget or GrammarBudget()
        exclude = set(exclude_primitives or [])
        self._active_primitives = {
            k: v for k, v in PRIMITIVE_REGISTRY.items() if k not in exclude
        }
        self.dimensional_checker = dimensional_checker
        self.variables_with_units = variables_with_units or {}
        
        self._candidates = []
        self._built = False

    def _build_candidates(self, context: ProposalContext):
        if self._built:
            return
        
        # Merge context variables if available
        vars_with_units = self.variables_with_units
        if not vars_with_units and hasattr(context, 'variables_with_units') and context.variables_with_units:
            vars_with_units = context.variables_with_units
            
        # Fallback to variable names if no units
        if not vars_with_units:
            vars_with_units = {v: "dimensionless" for v in context.variable_names}
            
        # Get ratios
        max_ratios = getattr(self.budget, 'max_ratio_candidates', 8)
        ratio_syms = _generate_ratio_symbols(vars_with_units, max_ratios=max_ratios)
        
        seen = set()
        self._candidates = []
        for ratio in ratio_syms:
            for cand in enumerate_candidates(
                ratio_symbol=ratio,
                budget=self.budget,
                active_primitives=self._active_primitives,
            ):
                if cand not in seen:
                    self._candidates.append(cand)
                    seen.add(cand)
        self._built = True

    def propose(self, context: ProposalContext) -> List[str]:
        self._build_candidates(context)
        return self._candidates

    def search_space_size(self, context: Optional[ProposalContext] = None) -> int:
        if not self._built:
            if context:
                self._build_candidates(context)
            else:
                self._build_candidates(ProposalContext(variable_names=list(self.variables_with_units.keys()), target_name="", data_statistics={}))
        return len(self._candidates)
