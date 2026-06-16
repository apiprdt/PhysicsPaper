import sympy as sp
import numpy as np
import logging
import itertools
from typing import List, Optional
from adcd.llm_proposer import BaseProposer, ProposalContext
from adcd.dimensional_checker import DimensionalChecker, ASTValidator

logger = logging.getLogger(__name__)

def standardize_expression(expr: sp.Expr) -> str:
    """
    Standardizes the theta parameter names in a SymPy expression to theta_0, theta_1, ...
    based on their order of appearance in a preorder traversal.
    """
    symbols = []
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol and (str(node).startswith("theta") or "temp" in str(node)):
            if node not in symbols:
                symbols.append(node)
    
    rename_dict = {s: sp.Symbol(f"theta_{i}") for i, s in enumerate(symbols)}
    # Convert to string to avoid SymPy expression rebuilding errors
    return str(expr.subs(rename_dict))

class GrammarProposer(BaseProposer):
    """
    Generates candidates from a formal grammar using dimensionless ratios
    and parameter-scaled physical variables.
    """
    def __init__(self, checker: Optional[DimensionalChecker] = None, seed: int = 42):
        self.checker = checker if checker is not None else DimensionalChecker()
        self.validator = ASTValidator(max_depth=7, max_tokens=25)
        self.seed = seed

    def propose(self, context: ProposalContext) -> List[str]:
        rng = np.random.RandomState(self.seed + context.iteration)
        
        # Step 1: Enumerate base dimensionless ratios (RATIO seeds)
        symbols = list(context.variable_names)
        if context.constants:
            for c in context.constants:
                if c not in symbols and not c.startswith("theta_"):
                    symbols.append(c)

        # Dimension-guided ratios
        base_ratios = self.checker.enumerate_dimensionless_ratios(symbols, max_degree=2)
        
        # Parameter-scaled ratios (v / theta_k) for variables with non-zero dimensions
        # This allows building dimensionless terms when no Buckingham-pi ratio exists
        param_counter = 999  # Temporary high number to avoid clashes before standardizing
        for var in context.variable_names:
            if var in self.checker.registry:
                dim = self.checker.registry[var]
                if dim != [0, 0, 0]:
                    base_ratios.append(sp.Symbol(var) / sp.Symbol(f"theta_temp_{param_counter}"))
                    param_counter -= 1
            else:
                # If not in registry, treat it as having 1 physical symbol (adaptive relaxation)
                base_ratios.append(sp.Symbol(var) / sp.Symbol(f"theta_temp_{param_counter}"))
                param_counter -= 1

        # Deduplicate base ratios structurally
        unique_ratios = []
        seen_ratios = set()
        for r in base_ratios:
            r_str = str(r)
            if r_str not in seen_ratios:
                seen_ratios.add(r_str)
                unique_ratios.append(r)

        # Step 2: Generate Unary Forms
        unaries = []
        temp_id = 0
        for R in unique_ratios:
            # Generate unaries for this ratio
            t0 = sp.Symbol(f"theta_temp_{temp_id}_0")
            t1 = sp.Symbol(f"theta_temp_{temp_id}_1")
            
            unary_templates = [
                t0 * R,
                t0 * R**t1,
                t0 * R**(-t1),
                t0 * sp.exp(-R / t1),
                t0 * (sp.exp(-R / t1) - 1.0),
                t0 * sp.log(1.0 + R / t1),
                t0 * sp.sin(R / t1),
                t0 * sp.cos(R / t1),
                t0 * sp.tanh(t1 / R)**2,
                t0 * R**2,
                t0 * R**(-1),
                t0 * sp.sqrt(sp.Abs(R)),
                # Composite: product of linear/power and exponential
                t0 * R * sp.exp(-R / t1),
                t0 * R**2 * sp.exp(-R / t1),
                t0 * R**(-1) * sp.exp(-R / t1),
                # Rational forms in unary
                t0 / (1.0 + R**2 / t1**2),
                t0 * R**2 / (R**2 + t1**2),
                t0 * R / (R + t1),
                # New standard rational templates for physical limits
                t0 * R**2 / (1.0 - R**2 / t1**2),
                t0 * R / (1.0 - R / t1),
                t0 * R**2 / (1.0 + R**2 / t1**2),
                t0 * R / (1.0 + R / t1)
            ]
            
            for ut in unary_templates:
                unaries.append(ut)
                temp_id += 1

        # Step 3: Generate Binary Forms
        # Combine pairs of unaries using +, *, -, and /
        binaries = []
        max_combos = 600
        unary_pairs = list(itertools.combinations(unaries, 2))
        if len(unary_pairs) > max_combos:
            indices = rng.choice(len(unary_pairs), size=max_combos, replace=False)
            pairs_to_combine = [unary_pairs[idx] for idx in indices]
        else:
            pairs_to_combine = unary_pairs

        for U1, U2 in pairs_to_combine:
            binaries.append(U1 + U2)
            binaries.append(U1 * U2)
            binaries.append(U1 - U2)
            
            # Division / Rational combinations (with safeguards against division by zero)
            try:
                if U2 != 0:
                    binaries.append(U1 / U2)
                binaries.append(U1 / (1.0 + U2))
                binaries.append(U1 / (1.0 - U2))
            except Exception:
                pass

        # Step 4: Standardize and validate all candidates
        from adcd.metrics import classify_structure
        
        family_candidates = {
            "exponential": [],
            "trigonometric": [],
            "logarithmic": [],
            "rational": [],
            "power_law": [],
            "polynomial": []
        }
        
        seen_cands = set()
        
        # Prioritize unaries over binaries, and simpler over complex
        all_generated = [(expr, True) for expr in unaries] + [(expr, False) for expr in binaries]
        
        # Sort by complexity first (number of tokens/depth)
        def complexity(expr_info):
            expr = expr_info[0]
            try:
                tokens = len(list(sp.preorder_traversal(expr)))
                # Give a small bonus to unaries
                bonus = 0 if expr_info[1] else 5
                return tokens + bonus
            except Exception:
                return 999

        all_generated_sorted = sorted(all_generated, key=complexity)

        for expr, is_unary in all_generated_sorted:
            try:
                # Standardize theta naming
                cand_str = standardize_expression(expr)
                
                # Verify AST constraints
                if self.validator.verify(cand_str):
                    if cand_str not in seen_cands:
                        seen_cands.add(cand_str)
                        # Classify the structure family
                        parsed_expr = sp.sympify(cand_str)
                        fam = classify_structure(parsed_expr)
                        if fam in family_candidates:
                            family_candidates[fam].append(cand_str)
                        else:
                            family_candidates["polynomial"].append(cand_str)
            except Exception:
                continue

        # Distribute the budget context.n_candidates across the families
        # We target n_oversample = n_budget * 4 because ~75% of candidates will be rejected
        # by the numerical ARC pre-filter.
        selected_candidates = []
        n_budget = context.n_candidates
        n_oversample = n_budget * 4
        
        # Keep taking candidates round-robin from each family until oversample target is reached
        families = list(family_candidates.keys())
        family_indices = {fam: 0 for fam in families}
        
        added_in_round = True
        while len(selected_candidates) < n_oversample and added_in_round:
            added_in_round = False
            for fam in families:
                idx = family_indices[fam]
                if idx < len(family_candidates[fam]):
                    cand = family_candidates[fam][idx]
                    if cand not in selected_candidates:
                        selected_candidates.append(cand)
                        added_in_round = True
                    family_indices[fam] += 1
                if len(selected_candidates) >= n_oversample:
                    break

        logger.info(f"[GrammarProposer] Selected {len(selected_candidates)} diverse candidates for ARC pre-filtering. Target: {n_oversample}")
        
        # Apply numerical ARC pre-filter
        selected_candidates = self._numerical_arc_prefilter(selected_candidates, context)
        logger.info(f"[GrammarProposer] After ARC pre-filter: {len(selected_candidates)} candidates")
        return selected_candidates[:n_budget]

    def _numerical_arc_prefilter(
        self,
        candidates: List[str],
        context: ProposalContext,
        tol: float = 0.05,
        n_samples: int = 10,
    ) -> List[str]:
        """
        Numerical ARC pre-filter: reject candidates that don't vanish at classical limit.
        
        SymPy's symbolic limit fails on parameterized expressions like exp(-r/theta_0)
        because it can't determine the sign of theta_0. This numerical pre-filter
        evaluates the expression near the classical limit and rejects those that
        don't vanish, using random theta samples to handle parameterization.
        
        Replaces the need for 4x oversampling budget.
        """
        import numpy as np
        
        # Safely extract limit variable and direction from context metadata
        limit_var = None
        limit_dir = None
        
        # 1. Try known_limits list
        if hasattr(context, "known_limits") and context.known_limits:
            limit_var = context.known_limits[0].get("variable")
            limit_dir = context.known_limits[0].get("limit")
        # 2. Fallback to direct attributes if they exist
        elif hasattr(context, "classical_limit_variable") and context.classical_limit_variable:
            limit_var = context.classical_limit_variable
            limit_dir = getattr(context, "classical_limit_direction", None)
            
        if not limit_var or not limit_dir:
            return candidates
        
        # Handle both string and list formats
        if isinstance(limit_var, str):
            limit_vars = [limit_var]
            limit_dirs = [limit_dir]
        else:
            limit_vars = limit_var
            limit_dirs = limit_dir
        
        rng = np.random.default_rng(self.seed)
        passed = []
        
        for cand_str in candidates:
            try:
                expr = sp.sympify(cand_str)
                phys_syms = [s for s in expr.free_symbols 
                             if str(s) in context.variable_names]
                theta_syms = [s for s in expr.free_symbols 
                              if str(s).startswith("theta")]
                
                vanishes = True
                for limit_v, limit_d in zip(limit_vars, limit_dirs):
                    limit_sym = sp.Symbol(limit_v)
                    # Set limit value
                    if limit_d == "0":
                        limit_val = 1e-6
                    elif limit_d in ("oo", "inf", "+oo"):
                        limit_val = 1e8
                    else:
                        try:
                            limit_val = float(limit_d) * 0.01
                        except:
                            continue
                    
                    # Test with n_samples random theta values
                    any_vanish = False
                    for _ in range(n_samples):
                        subs = {}
                        for s in phys_syms:
                            if str(s) == limit_v:
                                subs[s] = limit_val
                            else:
                                # Use midpoint of variable range
                                subs[s] = 1.0
                        for s in theta_syms:
                            subs[s] = rng.uniform(0.1, 10.0)
                        
                        try:
                            val = float(complex(expr.subs(subs)).real)
                            if np.isfinite(val) and abs(val) < tol:
                                any_vanish = True
                                break
                        except Exception:
                            continue
                    
                    if not any_vanish:
                        vanishes = False
                        break
                
                if vanishes:
                    passed.append(cand_str)
                    
            except Exception:
                continue
        
        logger.info(
            f"[GrammarProposer] ARC pre-filter: "
            f"rejected {len(candidates)-len(passed)}/{len(candidates)} candidates"
        )
        return passed
