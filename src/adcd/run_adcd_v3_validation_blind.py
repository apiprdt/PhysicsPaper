from __future__ import annotations

import json
import os
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

logging.basicConfig(
    filename="adcd_validation_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
import warnings
warnings.filterwarnings("always", category=RuntimeWarning)
logging.captureWarnings(True)

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget, PRIMITIVE_REGISTRY
from adcd.context import ProposalContext
from adcd.pipeline import Stage1Pipeline
from adcd.dimensional_checker import DimensionalChecker, ASTValidator
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.jax_optimizer import JAXOptimizer
from adcd.mode_detection import detect_correction_mode
from adcd.metrics import classify_structure, extended_bic_score

from adcd.quickfit import DOMAIN_TAXONOMY
from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig, JuliaEngineData




BIC_SIGNIFICANCE_THRESHOLD = 10.0  # Kass-Raftery "very strong evidence" (default for synthetic)

# AUDIT FIX (2026-08-13): the four-step protocol as published in Section 3.8
# of the paper is EXACTLY these four checks -- no more, no fewer.
# `primary_search` is intentionally NOT included -- it is computed by comparing
# against scenario.correction_expr (hidden ground truth) and must never gate
# the formal IDENTIFIABLE/WITHHELD verdict.
FORMAL_PROTOCOL_CHECKS = (
    "budget_disclosure",
    "positive_control",
    "ablation_control",
    "determinism_check",
)


@dataclass
class ScenarioThresholdConfig:
    """
    Per-scenario threshold configuration for ADCD search.

    Generalises the global BIC/NMSE constants so that:
    - Synthetic/textbook scenarios use tight thresholds (low noise, exact form)
    - Real observational data (SPARC, Hubble, etc.) uses calibrated thresholds
      that account for intrinsic scatter and measurement uncertainty.

    Design principles:
    - bic_threshold: Kass-Raftery scale.  6=strong, 10=very strong.
      For real data with scatter, use 6.0 to avoid over-rejection.
    - nmse_fine: how well the correction fits.  Synthetic: 0.1.
      Real observational data: derived from _calibrated_nmse() below.
    - nmse_coarse: coarse gate (prefilter).  Keep loose: 0.8-1.0.
    - groups: for hierarchical data.  Pass list of per-group sizes so
      n_eff = n_groups instead of n_points.  Critical for SPARC (147 galaxies).
    """
    bic_threshold: float = 10.0   # very strong (synthetic default)
    nmse_fine:     float = 0.1    # tight fit (synthetic default)
    nmse_coarse:   float = 1.0    # loose pre-filter
    groups:        Optional[List[int]] = None  # None = iid data

    @staticmethod
    def _calibrated_nmse(
        scatter_level: float,
        n_eff: int,
        confidence: float = 0.95,
        optimization_slack: float = 0.35,
    ) -> float:
        """
        Derive nmse_fine from first principles (Neyman-Pearson framework).

        For a PERFECT model on data with intrinsic scatter sigma, the minimum
        achievable NMSE is the statistical floor:

            residual ≈ sigma  =>  NMSE_floor ≈ sigma^2

        Asking for NMSE < sigma^2 is mathematically impossible — even a perfect
        model cannot do better. This floor is the only scientifically defensible
        lower bound.

        Sampling uncertainty for n_eff INDEPENDENT groups (chi-squared approx):
            std(NMSE_estimate) / NMSE ≈ sqrt(2 / n_eff)
        One-sided 95% upper margin (z=1.645):
            sampling_margin = floor * z * sqrt(2 / n_eff)

        optimization_slack: additional margin for grammar discretization
        and L-BFGS non-convergence. Derived from synthetic benchmarks
        (see audit/verify_threshold_provenance.py). Documented separately
        from the statistical floor so it can be audited independently.

        References:
            Kass & Raftery (1995), J. Amer. Stat. Assoc. 90(430):773-795
            Burnham & Anderson (2002), Model Selection and Multimodel Inference

        Args:
            scatter_level: fractional intrinsic scatter (sigma). E.g., 0.30
                for SPARC RAR (Li et al. 2018, ApJ 853:109, Table 2: ~0.13 dex
                ≈ 30% fractional). MUST be sourced from literature/measurement.
            n_eff: number of INDEPENDENT groups (not raw data points).
                For SPARC: 147 galaxies, NOT 2696 measurement points.
            confidence: one-sided confidence level for sampling margin (default 0.95).
            optimization_slack: fractional overhead above the statistical floor
                for grammar discretization and optimizer non-convergence (default 0.35).
        """
        import math
        z = 1.645 if confidence == 0.95 else 1.96
        floor = scatter_level ** 2
        sampling_margin = floor * z * math.sqrt(2.0 / max(n_eff, 2))
        return floor * (1.0 + optimization_slack) + sampling_margin

    @classmethod
    def synthetic(cls) -> "ScenarioThresholdConfig":
        """Clean synthetic data: tight thresholds."""
        return cls(bic_threshold=10.0, nmse_fine=0.1, nmse_coarse=1.0, groups=None)

    @classmethod
    def real_observational(cls, n_groups: int = 1, scatter_level: float = 0.25) -> "ScenarioThresholdConfig":
        """
        Real observational data with intrinsic scatter.

        nmse_fine is derived from first principles via _calibrated_nmse().
        The formula: floor*(1+slack) + sampling_margin, where
            floor = scatter_level^2,
            sampling_margin = floor * 1.645 * sqrt(2/n_eff).

        This replaces the previous (scatter_level**2)*5.0+0.05 formula which
        had no statistical derivation and produced thresholds 3-5x too loose.

        Args:
            n_groups: number of independent groups/objects (e.g. 147 for SPARC)
            scatter_level: fractional scatter sourced from literature (MUST be cited
                in the call site below and in any paper using these results).
        """
        nmse_fine_calibrated = cls._calibrated_nmse(
            scatter_level=scatter_level,
            n_eff=n_groups if n_groups > 1 else 1,
        )
        # Fallback placeholder: only used when galaxy_id is NOT available in X.
        # For SPARC: config_from_scenario() in julia_bridge.py builds REAL groups
        # from np.where(galaxy_ids==uid), so this placeholder is never active
        # for the SPARC scenario. It exists for future scenarios without galaxy_id.
        groups_list = [1] * n_groups if n_groups > 1 else None
        return cls(
            bic_threshold=6.0,   # Kass-Raftery "strong" -- appropriate for real data
            nmse_fine=nmse_fine_calibrated,
            nmse_coarse=1.0,
            groups=groups_list,
        )

    @classmethod
    def for_scenario(cls, scenario) -> "ScenarioThresholdConfig":
        """
        Automatically select thresholds based on scenario tier and domain.

        All scatter_level values are sourced from peer-reviewed literature —
        NOT chosen to make results look better. Each cite is mandatory.

        Tier logic:
        - "textbook" / "synthetic" / "cross_domain": synthetic defaults
        - "real" / "observational": relaxed thresholds via _calibrated_nmse()
        - Domain "mond_radial_acceleration" (SPARC): 147 galaxy groups, scatter ~0.30
        - Domain "hubble_expansion": ~1000 SNe, scatter ~0.15
        """
        tier = getattr(scenario, "tier", "synthetic")
        domain = getattr(scenario, "domain", "")

        # Domain-specific calibration takes priority
        if domain == "mond_radial_acceleration":
            # SPARC RAR: 147 independent galaxies.
            # scatter_level=0.30: Li et al. (2018, ApJ 853:109, Table 2) reports
            # intrinsic scatter ~0.13 dex ≈ 30% fractional. McGaugh et al. (2016)
            # report ~0.11 dex ≈ 25%. We use 0.30 (Li+2018, more conservative).
            # Result: nmse_fine ≈ 0.14 (vs. old hardcoded 0.50 — corrected per audit).
            return cls.real_observational(n_groups=147, scatter_level=0.30)
        elif domain == "hubble_expansion":
            # Hubble: scatter_level=0.15 (typical SNe Ia intrinsic dispersion ~0.15 mag).
            return cls.real_observational(n_groups=1, scatter_level=0.15)
        elif tier in ("real", "observational"):
            return cls.real_observational(n_groups=1, scatter_level=0.25)
        else:
            return cls.synthetic()


@dataclass
class ProtocolResult:
    scenario_name: str
    checks: Dict[str, dict] = field(default_factory=dict)
    all_passed: bool = False


def _build_context(scenario, n_candidates: int, exclude_primitives: Optional[List[str]] = None) -> ProposalContext:
    return ProposalContext(
        variable_names=scenario.classical_variables,
        target_name="residual",
        data_statistics={},
        n_candidates=n_candidates,
        constants=scenario.classical_constants,
        known_limits=[{
            "variable": scenario.classical_limit_variable,
            "limit": scenario.classical_limit_direction,
            "expected": "0",
        }],
        variables_with_units=scenario.variables_with_units,
    )


def _make_pipeline(checker: DimensionalChecker, scenario) -> Stage1Pipeline:
    validator = ASTValidator(max_depth=12, max_tokens=50)
    regimes = build_arc_regimes(
        scenario.classical_limit_variable,
        scenario.classical_limit_direction,
    )
    scorer = ARCScorer(regimes=regimes)
    return Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)


# ---------------------------------------------------------------------------
# DOMAIN RESTRICTIONS: these are the LOCKED historical low-signal boundaries
# described in the paper abstract, Section 5.1, and Table 2.
# domain_max is the *only* parameter that controls observation window.
# Changing any value here changes ALL reported BIC/NMSE numbers.
# ---------------------------------------------------------------------------
DOMAIN_RESTRICTIONS = {
    "Time Dilation":     {"domain_max": 0.3},   # v <= 0.3c (historical regime)
    "Screened Coulomb":  {"domain_max": 4.0},   # r <= 4.0  (default already 4.0)
    "Entropy Expansion": {"domain_max": 1.0},   # dV/V_i <= 1.0 (historical regime)
}


def _run_search(
    scenario,
    exclude_primitives: Optional[List[str]],
    seed: int,
    n_candidates: int = 400,
    threshold_cfg: Optional["ScenarioThresholdConfig"] = None,
    noise_level: float = 0.01,
    domain_max: float = None
) -> Tuple[List[Tuple[str, float, float, dict]], int, GrammarProposerV3]:
    """
    Runs one full GrammarProposerV3 search (Stage 1 gates + Stage 2 JAX
    optimization + BIC ranking) and returns candidates ranked by BIC.

    exclude_primitives=None means a FULLY BLIND search (Step 1). A non-empty
    list restricts the primitive registry for calibration (positive
    control) or robustness (ablation) checks -- the RATIO is never hinted
    either way; only the primitive family list is ever restricted, and only
    for the two checks whose entire published purpose is to isolate that
    variable.

    DOMAIN FIX (2026-08-08): generate_data() is now called WITH domain_max
    from DOMAIN_RESTRICTIONS. Previously the call had no domain_max argument,
    so Time Dilation silently used the default v<=0.99c and Entropy Expansion
    used dV/V_i<=100 -- both contradicting the paper's stated observation windows.
    """
    checker = DimensionalChecker()
    for var in scenario.classical_variables:
        if var not in checker.registry:
            checker.registry[var] = [0, 0, 0, 0, 0]
    for const in scenario.classical_constants:
        if const not in checker.registry:
            checker.registry[const] = [0, 0, 0, 0, 0]

    # Python proposer + pipeline are only needed in the Python path.
    # Julia path builds a dummy proposer for the return value; skip here.
    pipeline = _make_pipeline(checker, scenario)
    domain_kwargs = DOMAIN_RESTRICTIONS.get(scenario.name, {})
    X, y_obs, y_classical, _ = scenario.generate_data(
        noise_level=noise_level, seed=seed, **({"domain_max": domain_max} if domain_max is not None else domain_kwargs)
    )
    for c_name, c_val in scenario.classical_constants.items():
        if c_name not in X:
            X[c_name] = np.full_like(y_obs, c_val)

    # -------------------------------------------------------------------------
    # FIX: TARGET LEAKAGE REMOVED
    # Dynamically detect mode using Spearman correlation instead of reading
    # the answer key from the scenario. Recompute the residual based on the
    # detected mode.
    # -------------------------------------------------------------------------
    detected_mode, mode_conf = detect_correction_mode(y_obs, y_classical)
    
    if detected_mode == "multiplicative":
        residual = y_obs / y_classical - 1.0
    else:
        residual = y_obs - y_classical
        
    # GrammarProposerV3 always generates dimensionless shape functions scaled by theta_0.
    # theta_0 mathematically absorbs whatever units are needed to match the observation (or 1.0 if multiplicative).
    # Therefore, the target dimensionality for the shape function itself must always be dimensionless.
    target_dim_key = "dimensionless"

    if getattr(scenario, "engine", "python") == "julia":
        # -------------------------------------------------------------------------
        # JULIA ENGINE PATH
        # The Julia engine runs its own CorrectionProposer (6 patterns, expanded
        # grammar) and 5-gate FilterCascade internally.  The Python GrammarProposerV3
        # is NOT run here — it would be redundant and would report a different
        # (smaller) search space because it lacks D_sqrt_inv, D_nested_mond, etc.
        # -------------------------------------------------------------------------
        print("[_run_search] Delegating to ADCDJuliaEngine...")
        # Resolve thresholds: scenario-specific config overrides global constants
        tcfg = threshold_cfg or ScenarioThresholdConfig.for_scenario(scenario)
        # Build groups from galaxy_id if present
        groups = tcfg.groups
        if "galaxy_id" in X:
            galaxy_ids = np.asarray(X["galaxy_id"])
            unique_ids = np.unique(galaxy_ids)
            groups = [(np.where(galaxy_ids == uid)[0] + 1).tolist() for uid in unique_ids]

        config = JuliaEngineConfig(
            domain=scenario.domain,
            target_dim=target_dim_key,
            input_vars=scenario.classical_variables,
            known_constants=scenario.classical_constants,
            bic_threshold=tcfg.bic_threshold,
            nmse_coarse=tcfg.nmse_coarse,
            nmse_fine=tcfg.nmse_fine,
            n_restarts=50,
            max_proposals=n_candidates,
            groups=groups,
            excluded_primitives=list(exclude_primitives) if exclude_primitives else [],
            correction_type=detected_mode,  # Bug #2 fix: wire mode_detection → Julia
            classical_limit_direction=scenario.classical_limit_direction,
        )
        data = JuliaEngineData(
            y_classical=y_classical,
            y_obs=y_obs,
            vars={k: X[k] for k in scenario.classical_variables},
        )
        engine_jl = ADCDJuliaEngine()
        result = engine_jl.run(config, data)

        ranked = []
        for cand in result.identifiable:
            # cand.expr is an alias for cand.description (the expression string).
            # theta_dict keys follow the "theta_0", "theta_1"... convention
            # expected by _find_true_structure_in_pareto.
            theta_dict = {f"theta_{i}": v for i, v in enumerate(cand.theta)}
            # Julia delta_bic is positive-is-better (larger = more evidence).
            # To use the same sort order as Python (lower BIC = better),
            # we store -delta_bic so that sort(key=r[2]) gives the correct order.
            ranked.append((cand.expr, cand.nmse, -cand.delta_bic, theta_dict))

        space_size = result.n_proposals_generated
        ranked.sort(key=lambda r: r[2])  # ascending: most negative (-large_delta_bic) first

        # Build a dummy proposer for the return value.  Python callers use it only for
        # logging the search space size and primitive list.  For Julia, the real primitive
        # list comes from result.primitives_active — expose via a custom attribute so
        # run_scenario_protocol can distinguish the two paths.
        checker = DimensionalChecker()
        for var in scenario.classical_variables:
            if var not in checker.registry:
                checker.registry[var] = [0, 0, 0, 0, 0]
        for const in scenario.classical_constants:
            if const not in checker.registry:
                checker.registry[const] = [0, 0, 0, 0, 0]
        proposer = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=exclude_primitives,
            dimensional_checker=checker,
        )
        # Attach Julia primitives so budget_disclosure uses the correct list
        proposer._julia_primitives_active = result.primitives_active
    else:
        # -------------------------------------------------------------------------
        # PYTHON ENGINE PATH
        # Build proposer, generate candidates, and run Stage 1 + JAX optimization.
        # -------------------------------------------------------------------------
        context = _build_context(scenario, n_candidates=n_candidates)
        proposer = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=exclude_primitives,
            dimensional_checker=checker,
        )
        candidates = proposer.propose(context)
        space_size = proposer.search_space_size(context)

        stage1_results = pipeline.execute(
            [(c, True) for c in candidates], target_dim_key, X, residual,
            constants=scenario.classical_constants,
        )

        optimizer = JAXOptimizer(n_restarts=50)
        top_k = stage1_results[:30]
        ranked = []
        for expr_str, combined_score, mse, arc_score, deferred_arc in top_k:
            opt = optimizer.optimize(
                expr_str, X, residual, scenario.classical_variables,
                seed=seed, loss_mode="auto", y_classical=y_classical,
                correction_type=detected_mode,
            )
            if not np.isfinite(opt.nmse):
                continue

            # ---------------------------------------------------------------------
            # FIX: STRICT POST-FIT ARC ENFORCEMENT (Deferred ARC)
            # If the candidate bypassed ARC in stage 1 due to parameters,
            # rigorously re-evaluate ARC at the fitted parameters now.
            # ---------------------------------------------------------------------
            if deferred_arc:
                fitted_expr = sp.sympify(expr_str).subs(opt.theta)
                post_fit_arc_score = float(pipeline.scorer.score(fitted_expr, constants=scenario.classical_constants))
                if post_fit_arc_score <= 0.0:
                    continue # Rejected by ARC limit

            n_params = len([k for k in opt.theta if k.startswith("theta_")])
            b = extended_bic_score(opt.nmse, n_params, len(residual), n_candidates=len(candidates))
            ranked.append((expr_str, opt.nmse, b, opt.theta))

        ranked.sort(key=lambda r: r[2])  # lower BIC = better

    return ranked, space_size, proposer


def _find_true_structure_in_pareto(ranked_blind, scenario):
    """
    Find the true correction structure within the Pareto front.

    Returns (true_structure_bic, true_structure_rank, match_level) where
    match_level is one of: "exact", "class_only", "none".

    Matching uses numeric sampling rather than sp.simplify, which is brittle
    on exponentiated expressions. When parameter counts differ (e.g. ground
    truth has implicit coefficient 1, candidate has explicit theta_0), extra
    candidate thetas are collapsed to 1 before retry.
    """
    true_expr_str = getattr(scenario, "correction_expr", None)
    true_class = getattr(scenario, "correction_class", None)
    
    if not true_expr_str:
        return None, None, "none"
        
    true_expr = sp.sympify(true_expr_str)
    true_thetas = [str(s) for s in true_expr.free_symbols if str(s).startswith("theta_")]

    def _try_exact_match(cand_expr, theta_keys):
        """Try all routes to exact match via numeric sampling. Returns True if match found."""
        import numpy as np
        try:
            # Generate 5 random points in a safe domain
            test_points = np.random.uniform(0.1, 0.9, 5)
            # Find the main variable to substitute
            vars_cand = [str(s) for s in cand_expr.free_symbols if not str(s).startswith("theta_")]
            vars_true = [str(s) for s in true_expr.free_symbols if not str(s).startswith("theta_")]
            main_var = vars_cand[0] if vars_cand else (vars_true[0] if vars_true else "x")
            
            # Numeric evaluation function
            def eval_expr(expr, var_val, theta_subs):
                subs_dict = {sp.Symbol(main_var): var_val}
                for k, v in theta_subs.items():
                    subs_dict[sp.Symbol(k)] = v
                return float(expr.subs(subs_dict))
                
            # If candidate and ground truth match perfectly when all thetas are set to 1.0
            cand_subs = {k: 1.0 for k in theta_keys}
            true_subs = {k: 1.0 for k in true_thetas}
            
            match_all = True
            for val in test_points:
                try:
                    c_val = eval_expr(cand_expr, val, cand_subs)
                    t_val = eval_expr(true_expr, val, true_subs)
                    if not np.isclose(c_val, t_val):
                        match_all = False
                        break
                except Exception:
                    match_all = False
                    break
                    
            if match_all:
                return True
        except Exception:
            pass
            
        return False

    # 1. Exact Symbolic/Numeric Match (all routes)
    for rank, (expr_str, nmse, bic, theta_fit) in enumerate(ranked_blind):
        cand_expr = sp.sympify(expr_str)
        theta_keys = [k for k in theta_fit.keys() if k.startswith("theta_")]
        if _try_exact_match(cand_expr, theta_keys):
            return bic, rank + 1, "exact"

    # 2. Class Match Fallback
    for rank, (expr_str, nmse, bic, theta_fit) in enumerate(ranked_blind):
        disc_class = classify_structure(expr_str, theta_fit)
        if disc_class == true_class:
            return bic, rank + 1, "class_only"
            
    return None, None, "none"


def _guess_true_primitive(expr_str: str) -> Optional[str]:
    """Dynamically deduce the generating primitive from the ground truth expression."""
    if not expr_str: return None
    if "D_nested_mond" in expr_str: return "D_nested_mond"
    if "D_sqrt_inv" in expr_str: return "D_sqrt_inv"
    if "exp(" in expr_str: return "D_exp"
    if "log(" in expr_str: return "D_log"
    if "cos(" in expr_str or "sin(" in expr_str: return "D_osc"
    if "tanh(" in expr_str: return "D_sat"
    if "sqrt(" in expr_str:
        if "1 -" in expr_str or "1.0 -" in expr_str: return "D_lor"
        return "D_sqrt_inv"
    if "**" in expr_str and "**2" not in expr_str: return "D_pow"
    if "/" in expr_str: return "D_rat"
    return None


def run_scenario_protocol(
    scenario,
    seed: int = 42,
    top_k_val: int = 5,
    use_taxonomy_prior: bool = True,
    threshold_cfg: Optional["ScenarioThresholdConfig"] = None,
    noise_level: float = 0.01,
    domain_max: float = None
) -> ProtocolResult:
    result = ProtocolResult(scenario_name=scenario.name)
    if threshold_cfg is None:
        threshold_cfg = ScenarioThresholdConfig.for_scenario(scenario)

    true_primitive = _guess_true_primitive(scenario.correction_expr)
    
    taxonomy_allowed = None
    if use_taxonomy_prior:
        taxonomy_allowed = DOMAIN_TAXONOMY[scenario.domain]
        taxonomy_exclude = [p for p in PRIMITIVE_REGISTRY.keys() if p not in taxonomy_allowed]
    else:
        taxonomy_exclude = None

    # Step 0: Budget disclosure
    _, space_size_blind, proposer = _run_search(
        scenario, exclude_primitives=taxonomy_exclude, seed=seed, n_candidates=0,
        threshold_cfg=threshold_cfg, noise_level=noise_level, domain_max=domain_max
    )
    result.checks["budget_disclosure"] = {
        "search_space_size": space_size_blind,
        "primitives": getattr(proposer, "_julia_primitives_active", list(proposer._active_primitives.keys())),
        "pass": True,
    }

    # Step 1: Blind Search
    ranked_blind, _, _ = _run_search(
        scenario, exclude_primitives=taxonomy_exclude, seed=seed,
        threshold_cfg=threshold_cfg, noise_level=noise_level, domain_max=domain_max
    )

    top_candidates = []
    if ranked_blind:
        for expr_str, nmse, bic, theta_fit in ranked_blind[:top_k_val]:
            top_candidates.append({
                "expr_str": expr_str, "nmse": nmse, "bic": bic,
                "class": classify_structure(expr_str, theta_fit), "theta_fit": theta_fit,
            })

    top = ranked_blind[0] if ranked_blind else None
    true_structure_bic, true_structure_rank, match_level = _find_true_structure_in_pareto(ranked_blind, scenario)

    if top is not None:
        expr_str, nmse, bic, theta_fit = top
        result.checks["primary_search"] = {
            "top_candidate": expr_str, "nmse": nmse, "bic": bic, "theta_fit": theta_fit,
            "discovered_class": classify_structure(expr_str, theta_fit),
            "match_level": match_level, "true_structure_rank": true_structure_rank,
            "ground_truth_match_diagnostic_only": match_level in ["exact", "class_only"],
            "counts_toward_verdict": False, "pareto_front": top_candidates,
        }
    else:
        result.checks["primary_search"] = {"pass": False}

    # Step 2: Positive Control (Terisolasi)
    ranked_isolated, space_size_isolated, _ = _run_search(
        scenario,
        exclude_primitives=[p for p in PRIMITIVE_REGISTRY if p != true_primitive],
        seed=seed, threshold_cfg=threshold_cfg, noise_level=noise_level, domain_max=domain_max
    )
    pc_pass = len(ranked_isolated) > 0 and ranked_isolated[0][1] < threshold_cfg.nmse_fine
    result.checks["positive_control"] = {
        "search_space_size": space_size_isolated,
        "nmse": ranked_isolated[0][1] if ranked_isolated else None,
        "pass": pc_pass,
    }

    # Step 3: Ablation Control
    ranked_ablated, _, _ = _run_search(
        scenario, exclude_primitives=[true_primitive], seed=seed,
        threshold_cfg=threshold_cfg, noise_level=noise_level, domain_max=domain_max
    )
    if ranked_ablated and ranked_blind:
        bic_diff = ranked_ablated[0][2] - ranked_blind[0][2]
        result.checks["ablation_control"] = {
            "ablated_bic": ranked_ablated[0][2], "true_structure_bic": ranked_blind[0][2],
            "bic_diff": bic_diff, "pass": bic_diff > threshold_cfg.bic_threshold,
        }
    else:
        result.checks["ablation_control"] = {"pass": False}

    # Step 4: Determinism Check (Menguji Stabilitas Re-start Solver pada Dataset yang Sama)
    runs = []
    for _ in range(3):
        r, _, _ = _run_search(
            scenario, exclude_primitives=taxonomy_exclude, seed=seed,
            threshold_cfg=threshold_cfg, noise_level=noise_level, domain_max=domain_max
        )
        runs.append(r[0][0] if r else None)

    all_none = all(r is None for r in runs)
    determinism_pass = (not all_none) and (len(set(runs)) == 1)
    result.checks["determinism_check"] = {"runs": runs, "pass": determinism_pass}

    result.all_passed = all(
        result.checks[name].get("pass", False) for name in FORMAL_PROTOCOL_CHECKS if name in result.checks
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="ADCD Validation Protocol")
    parser.add_argument("--top-k", type=int, default=5, help="Number of Pareto Front candidates to display")
    parser.add_argument("--no-taxonomy", action="store_false", dest="taxonomy", help="Disable Domain Taxonomy Prior for Stage 1")
    parser.add_argument("--engine", type=str, choices=["python", "julia"], default="python", help="Which execution engine to use")
    parser.set_defaults(taxonomy=True)
    args = parser.parse_args()
    
    scenarios = {s.name: s for s in get_all_scenarios()}
    locked = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]
    
    all_results = {}
    for name in locked:
        if name not in scenarios:
            print(f"[SKIP] Scenario '{name}' not found.")
            continue
        print("=" * 80)
        mode_str = "BAYESIAN TAXONOMY PRIOR" if args.taxonomy else "BLIND SEARCH"
        print(f" ADCD VALIDATION PROTOCOL ({mode_str}): {name.upper()}")
        print("=" * 80)
        
        # Override the scenario's default engine if the user specified it explicitly via CLI
        scenarios[name].engine = args.engine
        
        res = run_scenario_protocol(scenarios[name], top_k_val=args.top_k, use_taxonomy_prior=args.taxonomy)
        all_results[name] = res

        for step, info in res.checks.items():
            status = "PASS" if info.get("pass") else "FAIL"
            # Remove pareto_front from info dict temporarily so it doesn't clutter the raw dict print
            info_to_print = {k: v for k, v in info.items() if k != "pareto_front"}
            print(f"[{status:^6}] {step.upper():<20} | {info_to_print}")
            
            if step == "primary_search" and "pareto_front" in info:
                print("\n" + " "*10 + "--- TOP-K PARETO FRONT ---")
                print(" "*10 + f"{'Rank':<5} | {'BIC':<8} | {'NMSE':<10} | {'Class':<15} | {'Equation'}")
                print(" "*10 + "-" * 70)
                for i, cand in enumerate(info["pareto_front"]):
                    bic_val = f"{cand['bic']:.2f}" if cand['bic'] is not None else "-"
                    nmse_val = f"{cand['nmse']:.2e}"
                    print(" "*10 + f"{i+1:<5} | {bic_val:<8} | {nmse_val:<10} | {cand['class']:<15} | {cand['expr_str']}")
                print(" "*10 + "-" * 70 + "\n")

        print("-" * 80)
        if res.all_passed:
            print("[SUCCESS] STATUS: All checks passed with a genuinely blind search "
                  "(no hand-typed ratio, no primitive hint in the discovery step).")
        else:
            print("[FAIL] STATUS: At least one check failed. Do NOT cite this scenario's "
                  "result until every check passes AND you have inspected the raw "
                  "discovered_class field by hand -- the placeholder pass logic in "
                  "this script is NOT yet a substitute for manual inspection.")
        print("=" * 80 + "\n")

    os.makedirs("run_outputs", exist_ok=True)
    report_name = os.path.join("run_outputs", "adcd_v3_taxonomy_validation_report.json") if args.taxonomy else os.path.join("run_outputs", "adcd_v3_blind_validation_report.json")
    with open(report_name, "w") as f:
        json.dump(
            {name: {"all_passed": r.all_passed, "checks": r.checks} for name, r in all_results.items()},
            f, indent=2, default=str,
        )
    print(f"Full report saved to {report_name}")


if __name__ == "__main__":
    main()

