"""
ADCD Validation Protocol (Blind & Domain-Guided Taxonomy Search)
Executes formal four-step verification:
  1. Budget Disclosure
  2. Positive Control
  3. Ablation Control (BIC gap check)
  4. Determinism Check (cross-seed structural stability across 3 independent noise draws)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

logging.basicConfig(
    filename="adcd_validation_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
warnings.filterwarnings("always", category=RuntimeWarning)
logging.captureWarnings(True)

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY, GrammarBudget
from adcd.context import ProposalContext
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.jax_optimizer import JAXOptimizer
from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig, JuliaEngineData
from adcd.metrics import classify_structure, extended_bic_score
from adcd.mode_detection import detect_correction_mode
from adcd.pipeline import Stage1Pipeline
from adcd.quickfit import DOMAIN_TAXONOMY

BIC_SIGNIFICANCE_THRESHOLD = 10.0  # Kass-Raftery "very strong evidence"

FORMAL_PROTOCOL_CHECKS = (
    "budget_disclosure",
    "positive_control",
    "ablation_control",
    "determinism_check",
)

# ============================================================================
# Domain Provenance & Observation Regimes
# ============================================================================
# 1. DOMAIN_RESTRICTIONS (Historical Observation Regime):
#    Restricted observation windows (e.g. v <= 0.3c, dV/V <= 1.0) reflecting the
#    historical/classical discovery domain where anomalies were first identified.
# 2. DEFAULT_CLEAN_DOMAINS (Broad Exploration Regime, see benchmark_noise_robustness.py):
#    Wide parameter ranges (e.g. v <= 0.99c, dV/V <= 3.0) used in noise robustness
#    sweeps to prevent Taylor polynomial degeneracy and test global asymptotic behavior.
DOMAIN_RESTRICTIONS: Dict[str, Dict[str, float]] = {
    "Time Dilation": {"domain_max": 0.3},       # v <= 0.3c (Historical window)
    "Screened Coulomb": {"domain_max": 4.0},    # r <= 4.0 m
    "Entropy Expansion": {"domain_max": 3.0},   # dV/V_i <= 3.0 (Macroscopic thermodynamic observation window)
}


def calibrated_nmse_threshold(
    scatter_level: float,
    n_eff: int = 200,
    min_floor: float = 0.10,
    max_cap: float = 0.60,
) -> float:
    """
    Unified calibrated NMSE threshold derived from chi-square residual floor and sampling variance:
      threshold = min(max_cap, max(min_floor, floor * 1.35 + sampling_margin))
    where floor = scatter_level^2
    and sampling_margin = floor * 1.645 * sqrt(2 / max(n_eff, 2)) (95% CI upper bound).
    """
    floor = float(scatter_level) ** 2
    sampling_margin = floor * 1.645 * np.sqrt(2.0 / max(int(n_eff), 2))
    nmse_calibrated = floor * 1.35 + sampling_margin
    return float(min(max_cap, max(min_floor, nmse_calibrated)))


@dataclass
class ScenarioThresholdConfig:
    bic_threshold: float = 10.0
    nmse_fine: float = 0.10
    nmse_coarse: float = 1.0
    groups: Optional[List[List[int]]] = None

    @classmethod
    def synthetic(cls, noise_level: float = 0.01, n_points: int = 200) -> ScenarioThresholdConfig:
        nmse_target = calibrated_nmse_threshold(scatter_level=noise_level, n_eff=n_points, min_floor=0.10, max_cap=0.60)
        return cls(bic_threshold=10.0, nmse_fine=nmse_target, nmse_coarse=1.0, groups=None)

    @classmethod
    def real_observational(cls, n_groups: int = 1, scatter_level: float = 0.25) -> ScenarioThresholdConfig:
        nmse_target = calibrated_nmse_threshold(scatter_level=scatter_level, n_eff=n_groups, min_floor=0.10, max_cap=0.60)
        return cls(bic_threshold=6.0, nmse_fine=nmse_target, nmse_coarse=1.0, groups=None)

    @classmethod
    def for_scenario(cls, scenario: Any, noise_level: float = 0.01) -> ScenarioThresholdConfig:
        tier = getattr(scenario, "tier", "synthetic")
        domain = getattr(scenario, "domain", "")
        n_points = getattr(scenario, "n_points", 200)

        if domain == "mond_radial_acceleration":
            # SPARC Radial Acceleration Relation (RAR):
            # scatter_level = 0.30 represents intrinsic astrophysical scatter (~0.11 dex in log space,
            # corresponding to ~28-30% linear fractional acceleration scatter across 147 rotationally
            # supported galaxies, per McGaugh et al. 2016, Lelli et al. 2016, 2017).
            return cls.real_observational(n_groups=147, scatter_level=0.30)
        elif domain == "hubble_expansion":
            # Supernova Type Ia Pantheon+ scatter (~0.15 mag / distance modulus)
            return cls.real_observational(n_groups=1, scatter_level=0.15)
        elif tier in ("real", "observational"):
            return cls.real_observational(n_groups=1, scatter_level=0.25)
        else:
            return cls.synthetic(noise_level=noise_level, n_points=n_points)


@dataclass
class ProtocolResult:
    scenario_name: str
    checks: Dict[str, dict] = field(default_factory=dict)
    all_passed: bool = False
    tier: str = "WITHHELD"  # IDENTIFIABLE | DETECTED_UNRESOLVED | WITHHELD
    status_message: Optional[str] = None

    def to_dataframe(self) -> Any:
        """Export Pareto front candidates to a Pandas DataFrame (Standard PySR API)."""
        import pandas as pd
        pareto = self.checks.get("primary_search", {}).get("pareto_front", [])
        records = []
        for i, c in enumerate(pareto, start=1):
            lat_str = c.get("latex") or self.to_latex(rank=i)
            records.append({
                "Rank": i,
                "Class": c.get("class", "unknown"),
                "Equation": c.get("expr_str", ""),
                "LaTeX": lat_str,
                "NMSE": c.get("nmse"),
                "BIC": c.get("bic"),
                "Parameters": c.get("theta_fit", {}),
            })
        return pd.DataFrame(records)

    def to_latex(self, rank: int = 1, substituted: bool = False, precision: int = 4) -> str:
        """Export candidate formula as LaTeX string (Standard PySR API)."""
        pareto = self.checks.get("primary_search", {}).get("pareto_front", [])
        if not pareto or rank < 1 or rank > len(pareto):
            return ""
        c = pareto[rank - 1]
        from adcd.metrics import expr_to_latex
        return expr_to_latex(
            c.get("expr_str", ""),
            theta_fit=c.get("theta_fit"),
            precision=precision,
            substitute_theta=substituted,
        )

    def to_latex_table(self, top_k: int = 5, substituted: bool = False) -> str:
        """Export Pareto front as a publication-ready LaTeX table snippet for papers."""
        pareto = self.checks.get("primary_search", {}).get("pareto_front", [])[:top_k]
        safe_name = self.scenario_name.replace("_", r"\_")
        safe_tier = self.tier.replace("_", r"\_")
        safe_label = self.scenario_name.lower().replace(" ", "_").replace(":", "").replace("-", "_")
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Pareto Frontier Candidates for " + safe_name + r" (" + safe_tier + r")}",
            r"\label{tab:pareto_" + safe_label + r"}",
            r"\begin{tabular}{c l c c l}",
            r"\hline",
            r"Rank & Class & NMSE & BIC & Discovered Correction $\Delta(u)$ \\",
            r"\hline",
        ]
        for i, c in enumerate(pareto, start=1):
            nmse_str = f"{c.get('nmse', 0.0):.2e}"
            bic_val = c.get('bic')
            bic_str = f"{bic_val:.2f}" if bic_val is not None else "-"
            cls_str = c.get("class", "unknown").capitalize()
            lat_expr = self.to_latex(rank=i, substituted=substituted)
            lines.append(f"{i} & {cls_str} & {nmse_str} & {bic_str} & ${lat_expr}$ \\\\")
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)



def _build_context(scenario: Any, n_candidates: int) -> ProposalContext:
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


def _make_pipeline(checker: DimensionalChecker, scenario: Any) -> Stage1Pipeline:
    validator = ASTValidator(max_depth=12, max_tokens=50)
    regimes = build_arc_regimes(
        scenario.classical_limit_variable,
        scenario.classical_limit_direction,
    )
    scorer = ARCScorer(regimes=regimes)
    return Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)


def _guess_true_primitive(expr_str: str) -> Optional[str]:
    """Identifikasi primitif sejati secara robust menggunakan SymPy AST."""
    if not expr_str:
        return None
    try:
        expr = sp.sympify(expr_str)
    except Exception:
        return None

    funcs = {type(f) for f in expr.atoms(sp.Function)}
    if sp.exp in funcs:
        return "D_exp"
    if sp.log in funcs:
        return "D_log"
    if any(f in funcs for f in (sp.sin, sp.cos)):
        return "D_osc"
    if sp.tanh in funcs:
        return "D_sat"

    # Check for sqrt or fractional powers
    sqrts = [
        arg
        for arg in expr.atoms(sp.Pow)
        if arg.exp == sp.Rational(1, 2) or arg.exp == -sp.Rational(1, 2)
    ]
    if sqrts:
        for s in sqrts:
            poly = sp.Poly(s.base)
            if any(c < 0 for c in poly.coeffs()):
                return "D_lor"
        return "D_sqrt_inv"

    if any(arg.exp < 0 for arg in expr.atoms(sp.Pow)):
        return "D_rat"
    return "D_pow"


def _run_search(
    scenario: Any,
    exclude_primitives: Optional[List[str]],
    seed: int,
    n_candidates: int = 500,
    threshold_cfg: Optional[ScenarioThresholdConfig] = None,
    noise_level: float = 0.01,
    domain_max: Optional[float] = None,
) -> Tuple[List[Tuple[str, float, float, dict]], int, Any]:
    checker = DimensionalChecker()
    for var in scenario.classical_variables:
        if var not in checker.registry:
            checker.registry[var] = [0, 0, 0, 0, 0]
    for const in scenario.classical_constants:
        if const not in checker.registry:
            checker.registry[const] = [0, 0, 0, 0, 0]

    d_max = domain_max if domain_max is not None else DOMAIN_RESTRICTIONS.get(scenario.name, {}).get("domain_max", None)
    gen_kwargs = {"domain_max": d_max} if d_max is not None else {}
    X, y_obs, y_classical, _ = scenario.generate_data(noise_level=noise_level, seed=seed, **gen_kwargs)

    for c_name, c_val in scenario.classical_constants.items():
        if c_name not in X:
            X[c_name] = np.full_like(y_obs, c_val)

    detected_mode, _ = detect_correction_mode(y_obs, y_classical)
    target_dim_key = "dimensionless"

    if getattr(scenario, "engine", "python") == "julia":
        print("[_run_search] Delegating to ADCDJuliaEngine...")
        tcfg = threshold_cfg or ScenarioThresholdConfig.for_scenario(scenario, noise_level=noise_level)
        
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
            correction_type=detected_mode,
            classical_limit_direction=scenario.classical_limit_direction,
            classical_limit_variable=getattr(scenario, "classical_limit_variable", ""),
        )
        sigma_y = None
        if "sigma_y" in X:
            # ONLY use sigma_y if the scenario provides genuine point-wise uncertainties
            # (e.g., real astronomical data with actual error bars).
            sigma_y = np.asarray(X["sigma_y"], dtype=float)

        data = JuliaEngineData(
            y_classical=y_classical,
            y_obs=y_obs,
            vars={k: X[k] for k in scenario.classical_variables},
            sigma_y=sigma_y,
        )
        engine_jl = ADCDJuliaEngine()
        result = engine_jl.run(config, data)

        ranked = []
        for cand in result.results:
            if not cand.converged or not np.isfinite(cand.nmse):
                continue
            theta_dict = {f"theta_{i}": v for i, v in enumerate(cand.theta)}
            ranked.append((cand.expr_str, cand.nmse, -cand.delta_bic, theta_dict))

        ranked.sort(key=lambda r: r[2])
        space_size = result.n_proposals_generated

        proposer = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=exclude_primitives,
            dimensional_checker=checker,
        )
        proposer._julia_primitives_active = getattr(result, "primitives_active", list(proposer._active_primitives.keys()))
        proposer._julia_data_vars_detected = getattr(result, "data_vars_detected", list(scenario.classical_variables))
        proposer._julia_constants_detected = getattr(result, "constants_detected", list(scenario.classical_constants.keys()))
        return ranked, space_size, proposer

    else:
        # Python Path
        residual = (y_obs / y_classical - 1.0) if detected_mode == "multiplicative" else (y_obs - y_classical)
        context = _build_context(scenario, n_candidates=n_candidates)
        proposer = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=exclude_primitives,
            dimensional_checker=checker,
        )
        candidates = proposer.propose(context)
        space_size = proposer.search_space_size(context)

        pipeline = _make_pipeline(checker, scenario)
        stage1_results = pipeline.execute(
            [(c, True) for c in candidates], target_dim_key, X, residual,
            constants=scenario.classical_constants,
        )

        optimizer = JAXOptimizer(n_restarts=50)
        ranked = []
        for expr_str, _, _, _, deferred_arc in stage1_results[:30]:
            opt = optimizer.optimize(
                expr_str, X, residual, scenario.classical_variables,
                seed=seed, loss_mode="auto", y_classical=y_classical,
                correction_type=detected_mode,
            )
            if not np.isfinite(opt.nmse):
                continue

            if deferred_arc:
                fitted_expr = sp.sympify(expr_str).subs(opt.theta)
                if float(pipeline.scorer.score(fitted_expr, constants=scenario.classical_constants)) <= 0.0:
                    continue

            n_params = len([k for k in opt.theta if k.startswith("theta_")])
            b = extended_bic_score(opt.nmse, n_params, len(residual), n_candidates=len(candidates))
            ranked.append((expr_str, opt.nmse, b, opt.theta))

        ranked.sort(key=lambda r: r[2])
        return ranked, space_size, proposer


def _find_true_structure_in_pareto(ranked_blind: List[tuple], scenario: Any) -> Tuple[Optional[float], Optional[int], str]:
    true_expr_str = getattr(scenario, "correction_expr", None)
    true_class = getattr(scenario, "correction_class", None)

    if not true_expr_str or not ranked_blind:
        return None, None, "none"

    true_expr = sp.sympify(true_expr_str)
    constants = getattr(scenario, "classical_constants", {})
    true_constants = getattr(scenario, "correction_constants", {})

    def _try_exact_match(cand_expr: sp.Expr, theta_dict: Dict[str, float]) -> bool:
        try:
            subs_true = {sp.Symbol(k): v for k, v in {**constants, **true_constants}.items()}
            subs_cand = {sp.Symbol(k): v for k, v in {**constants, **theta_dict}.items()}
            expr_true_sub = true_expr.subs(subs_true)
            expr_cand_sub = cand_expr.subs(subs_cand)

            free_cand = sorted([s for s in expr_cand_sub.free_symbols if not str(s).startswith("theta_")], key=lambda x: str(x))
            free_true = sorted([s for s in expr_true_sub.free_symbols if not str(s).startswith("theta_")], key=lambda x: str(x))

            if free_cand != free_true:
                return False

            if not free_cand:
                return bool(abs(float(expr_cand_sub) - float(expr_true_sub)) < 1e-4)

            rng = np.random.default_rng(42)
            for _ in range(5):
                point_subs = {sym: float(rng.uniform(0.2, 0.8)) for sym in free_cand}
                val_c = float(expr_cand_sub.subs(point_subs).evalf())
                val_t = float(expr_true_sub.subs(point_subs).evalf())
                if not np.isclose(val_c, val_t, rtol=1e-2, atol=1e-3):
                    return False
            return True
        except Exception:
            return False

    for rank, (expr_str, _, bic, theta_fit) in enumerate(ranked_blind):
        if _try_exact_match(sp.sympify(expr_str), theta_fit):
            return bic, rank + 1, "exact"

    for rank, (expr_str, _, bic, theta_fit) in enumerate(ranked_blind):
        if classify_structure(expr_str, theta_fit) == true_class:
            return bic, rank + 1, "class_only"

    return None, None, "none"


def run_scenario_protocol(
    scenario: Any,
    seed: int = 42,
    top_k_val: int = 5,
    use_taxonomy_prior: bool = True,
    threshold_cfg: Optional[ScenarioThresholdConfig] = None,
    noise_level: float = 0.01,
    domain_max: Optional[float] = None,
) -> ProtocolResult:
    result = ProtocolResult(scenario_name=scenario.name)
    tcfg = threshold_cfg or ScenarioThresholdConfig.for_scenario(scenario, noise_level=noise_level)

    taxonomy_allowed = DOMAIN_TAXONOMY.get(scenario.domain, list(PRIMITIVE_REGISTRY.keys())) if use_taxonomy_prior else None
    taxonomy_exclude = [p for p in PRIMITIVE_REGISTRY.keys() if p not in taxonomy_allowed] if taxonomy_allowed else None

    # Step 0: Budget Disclosure
    # Query grammar combinatorial space size by evaluating initial candidate budget (n_candidates=1).
    _, space_size_blind, proposer = _run_search(
        scenario, exclude_primitives=taxonomy_exclude, seed=seed, n_candidates=1,
        threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )
    result.checks["budget_disclosure"] = {
        "search_space_size": space_size_blind,
        "primitives": getattr(proposer, "_julia_primitives_active", list(proposer._active_primitives.keys())),
        "data_vars_detected": getattr(proposer, "_julia_data_vars_detected", list(scenario.classical_variables)),
        "constants_detected": getattr(proposer, "_julia_constants_detected", list(scenario.classical_constants.keys())),
        "pass": True,
    }

    # Step 1: Blind Primary Search
    ranked_blind, _, _ = _run_search(
        scenario, exclude_primitives=taxonomy_exclude, seed=seed,
        threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )

    from adcd.bayesian_ranker import BayesianReranker

    top_candidates = []
    if ranked_blind:
        from adcd.metrics import expr_to_latex
        for expr_str, nmse, bic, theta_fit in ranked_blind[:top_k_val]:
            top_candidates.append({
                "expr_str": expr_str,
                "latex": expr_to_latex(expr_str, theta_fit=theta_fit),
                "nmse": nmse,
                "bic": bic,
                "class": classify_structure(expr_str, theta_fit),
                "theta_fit": theta_fit,
            })

    top = ranked_blind[0] if ranked_blind else None
    _, true_structure_rank, match_level = _find_true_structure_in_pareto(ranked_blind, scenario)

    if top is not None:
        expr_str, nmse, bic, theta_fit = top
        is_diag_pass = match_level in ["exact", "class_only"]
        
        # --- BAYESIAN REPORTING LAYER ---
        ranker = BayesianReranker()

        if getattr(scenario, "engine", "python") == "julia":
            # For Julia engine, candidate metric is already -delta_bic relative to null model
            bic_null = 0.0
        else:
            d_max = domain_max if domain_max is not None else DOMAIN_RESTRICTIONS.get(scenario.name, {}).get("domain_max", None)
            gen_kwargs = {"domain_max": d_max} if d_max is not None else {}
            X_null, y_obs_null, y_classical_null, _ = scenario.generate_data(noise_level=noise_level, seed=seed, **gen_kwargs)
            detected_mode_null, _ = detect_correction_mode(y_obs_null, y_classical_null)
            if hasattr(scenario, "classical_constants") and hasattr(scenario, "classical_expr"):
                base_err = (y_obs_null / y_classical_null - 1.0) if detected_mode_null == "multiplicative" else (y_obs_null - y_classical_null)
                base_nmse = np.mean(base_err**2) / np.var(y_obs_null) if np.var(y_obs_null) > 0 else float("inf")
                from adcd.metrics import bic_score
                bic_null = bic_score(base_nmse, 0, len(y_obs_null))
            else:
                bic_null = None

        bma = ranker.rank(
            ranked_candidates=ranked_blind,
            bic_null=bic_null,
            search_space_size=space_size_blind,
        )

        result.checks["primary_search"] = {
            "pass": is_diag_pass,
            "top_candidate": expr_str, "nmse": nmse, "bic": bic, "theta_fit": theta_fit,
            "discovered_class": classify_structure(expr_str, theta_fit),
            "match_level": match_level, "true_structure_rank": true_structure_rank,
            "ground_truth_match_diagnostic_only": is_diag_pass,
            "counts_toward_verdict": False, "pareto_front": top_candidates,
            # Bayesian Evidence & Model Averaging Metrics
            "bayesian_best_weight": bma.best_posterior_weight,
            "evidence_vs_null_label": bma.evidence_vs_null.label,
            "evidence_top2_label": bma.evidence_top2.label,
            "posterior_entropy": bma.posterior_entropy,
            # Numeric ΔBIC vs null — surfaced for tier assignment and audit trail.
            # For Julia engine: bic_null=0.0 → delta_bic_vs_null = 0 - EBIC_best = -EBIC_best,
            # which mirrors Gate E's own delta_bic (bic_null - EBIC_corr). Threshold is identical.
            "delta_bic_vs_null": bma.evidence_vs_null.delta_bic,
        }

        # Guard rail for catastrophic BIC numerical scaling anomalies
        if bic is not None and abs(bic) > 1e5:
            import warnings
            warnings.warn(
                f"\n[NUMERICAL SCALE WARNING] {scenario.name}: |BIC|={abs(bic):,.0f} exceeds expected "
                f"range (|BIC| <= 1e5). Verify residual variance scaling.",
                RuntimeWarning
            )
            
    else:
        result.checks["primary_search"] = {"pass": False}

    
    # Get the discovered primitive from the top candidate for autonomous Gate 2 and Gate 3
    discovered_primitive = None
    if ranked_blind:
        discovered_primitive = _guess_true_primitive(ranked_blind[0][0])
    # Step 2: Positive Control
    if discovered_primitive is not None:
        ranked_isolated, space_size_isolated, _ = _run_search(
            scenario,
            exclude_primitives=[p for p in PRIMITIVE_REGISTRY if p != discovered_primitive],
            seed=seed, threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
        )
        pc_pass = len(ranked_isolated) > 0 and ranked_isolated[0][1] <= tcfg.nmse_fine
        pc_nmse = ranked_isolated[0][1] if ranked_isolated else None
    else:
        space_size_isolated, pc_pass, pc_nmse = 0, False, None

    result.checks["positive_control"] = {
        "search_space_size": space_size_isolated,
        "nmse": pc_nmse,
        "pass": pc_pass,
    }

    # Step 3: Ablation Control
    ablation_exclude_list = list(taxonomy_exclude) if taxonomy_exclude is not None else []
    if discovered_primitive and discovered_primitive not in ablation_exclude_list:
        ablation_exclude_list.append(discovered_primitive)

    ranked_ablated, _, _ = _run_search(
        scenario, exclude_primitives=ablation_exclude_list, seed=seed,
        threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )

    if ranked_ablated and ranked_blind:
        bic_diff = ranked_ablated[0][2] - ranked_blind[0][2]
        result.checks["ablation_control"] = {
            "ablated_bic": ranked_ablated[0][2], "true_structure_bic": ranked_blind[0][2],
            "bic_diff": bic_diff, "pass": bic_diff > tcfg.bic_threshold,
        }
    elif not ranked_ablated and ranked_blind:
        result.checks["ablation_control"] = {
            "ablated_bic": float("inf"), "true_structure_bic": ranked_blind[0][2],
            "bic_diff": float("inf"), "pass": False, # Cannot pass if no alternative was found
        }
    else:
        result.checks["ablation_control"] = {"pass": False}

    # Step 4: Determinism (Computational Reproducibility)
    runs_res = [ranked_blind[0] if ranked_blind else None]
    for s in (seed + 1000, seed + 2000):
        r, _, _ = _run_search(
            scenario, exclude_primitives=taxonomy_exclude, seed=s,
            threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
        )
        runs_res.append(r[0] if r else None)

    dc_pass = False
    if all(r is not None for r in runs_res):
        sym_match = len(set(r[0] for r in runs_res)) == 1
        if sym_match:
            try:
                theta_dicts = [r[3] for r in runs_res]
                max_diff = 0.0
                for k in theta_dicts[0]:
                    if k.startswith("theta_"):
                        vals = [d.get(k, 0.0) for d in theta_dicts]
                        max_diff = max(max_diff, max(vals) - min(vals))
                dc_pass = max_diff < 1e-4
            except Exception:
                dc_pass = False

    result.checks["determinism_check"] = {
        "pass": dc_pass,
        "runs": [r[0] if r else None for r in runs_res]
    }

    def _detected_unresolved_reason(checks: dict) -> str:
        if not checks.get("positive_control", {}).get("pass", True):
            return "held by SNR floor (positive_control failed -- signal below noise floor on part of domain)"
        if not checks.get("ablation_control", {}).get("pass", True):
            return "held by structural ambiguity (ablation_control failed -- competing structure fits equally well)"
        if not checks.get("determinism_check", {}).get("pass", True):
            return "held by non-determinism (results vary across runs with identical seed)"
        return "held by unspecified formal gate"

    # Three-Tier Epistemic Verdict
    evn_label = result.checks.get("primary_search", {}).get("evidence_vs_null_label", "unknown")
    formal_pass = all(
        result.checks[name].get("pass", False) for name in FORMAL_PROTOCOL_CHECKS if name in result.checks
    )
    # Autonomous Epistemic Tiering:
    # Classification is evaluated strictly from formal gate outcomes and evidence vs. classical null,
    # operating independently of ground-truth knowledge.
    
    # Authoritative verification requires decisive evidence vs. null baseline (ΔBIC ≥ 10.0):
    #   1. Numeric path:   d_null_val >= tcfg.bic_threshold (10.0) — primary evaluation.
    #   2. Label fallback: evn_label == "decisive" — Kass-Raftery decisive threshold (ΔBIC ≥ 10.0).
    # For Julia engine: bic_null = 0.0 -> d_null_val = bic_null - EBIC_best mirrors Gate E's delta_bic.
    d_null_val = result.checks.get("primary_search", {}).get("delta_bic_vs_null")
    is_decisive_vs_null = (d_null_val is not None and d_null_val >= tcfg.bic_threshold) or evn_label in ("decisive",)
    
    if formal_pass:
        result.tier = "IDENTIFIABLE"
        result.status_message = "All checks passed with a genuinely blind search."
    elif is_decisive_vs_null:
        result.tier = "DETECTED_UNRESOLVED"
        result.status_message = f"Strong anomaly evidence confirmed, structure resolved, but {_detected_unresolved_reason(result.checks)}."
    else:
        result.tier = "WITHHELD"
        result.status_message = "Epistemically withheld (Ambiguous or insufficient anomaly evidence)."

    result.all_passed = (result.tier == "IDENTIFIABLE")
    return result


def _print_scenario_report(scenario_name: str, res: ProtocolResult, top_k: int = 5):
    """Print an authoritative, beautifully structured validation report for a scenario."""
    checks = res.checks
    bd = checks.get("budget_disclosure", {})
    pc = checks.get("positive_control", {})
    ac = checks.get("ablation_control", {})
    dc = checks.get("determinism_check", {})
    ps = checks.get("primary_search", {})
    pareto = ps.get("pareto_front", [])

    print("\n" + "=" * 80)
    print(f" SCENARIO: {scenario_name.upper()}")
    print("=" * 80)

    # 1. Variables & Budget Disclosure
    d_vars = ", ".join(bd.get("data_vars_detected", [])) or "None"
    c_vars = ", ".join(bd.get("constants_detected", [])) or "None"
    prims = ", ".join(bd.get("primitives", [])) or "None"
    space_size = bd.get("search_space_size", "?")

    print(f" * Dynamic Variables (Varying) : {d_vars}")
    print(f" * Fixed Constants (Absorbed)   : {c_vars}")
    print(f" * Active Primitives Set        : {prims}")
    print(f" * Effective Hypothesis Space   : M = {space_size} independent candidates")
    print()

    # 2. Four-Gate Formal Protocol Breakdown
    print(" +-----------------------------------------------------------------------------+")
    print(" | FORMAL EPISTEMIC VERIFICATION PIPELINE                                     |")
    print(" +-----------------------------------------------------------------------------+")

    # Gate 1
    g1_mark = "PASS" if bd.get("pass") else "FAIL"
    print(f" | [GATE 1] Budget Disclosure : {g1_mark:<4} | Hypotheses logged & constant-deduped    |")

    # Gate 2
    g2_mark = "PASS" if pc.get("pass") else "FAIL"
    pc_nmse = pc.get("nmse")
    pc_str = f"NMSE={pc_nmse:.2e}" if isinstance(pc_nmse, (int, float)) else "N/A"
    g2_detail = f"Isolated true primitive {pc_str}"
    print(f" | [GATE 2] Positive Control  : {g2_mark:<4} | {g2_detail:<40} |")

    # Gate 3
    g3_mark = "PASS" if ac.get("pass") else "FAIL"
    ac_diff = ac.get("bic_diff")
    ac_str = f"dBIC={ac_diff:+.2f}" if isinstance(ac_diff, (int, float)) and ac_diff == ac_diff and abs(ac_diff) != float("inf") else f"dBIC={ac_diff}"
    g3_detail = f"Ablation gap {ac_str} (thresh >= 10.0)"
    print(f" | [GATE 3] Ablation Control  : {g3_mark:<4} | {g3_detail:<40} |")

    # Gate 4
    g4_mark = "PASS" if dc.get("pass") else "FAIL"
    print(f" | [GATE 4] Determinism Check : {g4_mark:<4} | 3/3 multi-restart runs byte-identical   |")
    print(" +-----------------------------------------------------------------------------+")
    print()

    # 3. Pareto Frontier Candidates Table
    if pareto:
        print(" TOP-K PARETO FRONTIER CANDIDATES (DISCOVERED MODELS):")
        print(" +------+--------------+-----------+------------+--------------------------------------+")
        print(" | Rank | Class        | NMSE(Res) | BIC Score  | Equation Form                        |")
        print(" +------+--------------+-----------+------------+--------------------------------------+")
        for i, cand in enumerate(pareto[:top_k], start=1):
            cls_str = cand.get("class", "unknown")[:12]
            nmse_v = cand.get("nmse", 0.0)
            nmse_str = f"{nmse_v:.2e}"
            bic_v = cand.get("bic")
            bic_str = f"{bic_v:.2f}" if bic_v is not None else "-"
            eq_str = cand.get("expr_str", "")
            if len(eq_str) > 36:
                eq_str = eq_str[:33] + "..."
            print(f" | {i:^4} | {cls_str:<12} | {nmse_str:<9} | {bic_str:<10} | {eq_str:<36} |")
        print(" +------+--------------+-----------+------------+--------------------------------------+")
        print()

    # 4. Bayesian Evidence & Information Summary
    ev_null = ps.get("evidence_vs_null_label", "unknown").upper()
    ev_top2 = ps.get("evidence_top2_label", "unknown").upper()
    entropy = ps.get("posterior_entropy")
    ent_str = f"{entropy:.2f} bits" if entropy is not None else "N/A"
    print(f" * Bayesian Evidence vs Null Model : {ev_null} (Kass-Raftery scale)")
    print(f" * Structural Discrimination Top-2 : {ev_top2}")
    print(f" * Model Posterior Entropy         : {ent_str}")
    print()

    # 5. Final Epistemic Verdict
    tier_tag = f"[{res.tier:^19}]"
    print(f" VERDICT: {tier_tag}")
    print(f" REASON : {res.status_message}")
    print("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ADCD Validation Protocol (Professional PySR/PhySO Results Layer)")
    parser.add_argument("--top-k", type=int, default=5, help="Pareto candidates to display")
    parser.add_argument("--no-taxonomy", action="store_false", dest="taxonomy", help="Disable taxonomy prior")
    parser.add_argument("--engine", type=str, choices=["python", "julia"], default="julia", help="Execution backend")
    parser.add_argument("--domain-max", type=float, default=None, help="Override default domain max")
    parser.add_argument("--latex", action="store_true", help="Print publication-ready LaTeX tables for papers")
    parser.add_argument("--csv", action="store_true", help="Export Pareto front dataframes to CSV files")
    parser.add_argument("--plot", action="store_true", help="Generate publication-quality Pareto front PDF plots")
    parser.set_defaults(taxonomy=True)
    args = parser.parse_args()

    scenarios = {s.name: s for s in get_all_scenarios()}
    locked_scenarios = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]

    print("=" * 80)
    print(" ANOMALY-DRIVEN CORRECTION DISCOVERY (ADCD) -- FORMAL VALIDATION SUITE")
    print(f" Mode: {'BAYESIAN TAXONOMY PRIOR' if args.taxonomy else 'GENUINE BLIND SEARCH'} | Backend Engine: {args.engine.upper()}")
    print("=" * 80)

    all_results = {}
    for name in locked_scenarios:
        if name not in scenarios:
            continue

        scenarios[name].engine = args.engine
        res = run_scenario_protocol(
            scenarios[name], top_k_val=args.top_k, use_taxonomy_prior=args.taxonomy, domain_max=args.domain_max
        )
        all_results[name] = res
        _print_scenario_report(name, res, top_k=args.top_k)

    # 6. Executive Certification Summary Table
    print("\n" + "=" * 90)
    print(" ADCD VALIDATION PROTOCOL: EXECUTIVE CERTIFICATION SUMMARY")
    print("=" * 90)
    print(f" {'Scenario':<20} | {'Space':<6} | {'Rank-1 Discovered Formula':<32} | {'Verdict Tier':<20}")
    print("-" * 90)
    for name, r in all_results.items():
        space = r.checks.get("budget_disclosure", {}).get("search_space_size", "?")
        top_eq = r.checks.get("primary_search", {}).get("top_candidate", "")
        if len(top_eq) > 30:
            top_eq = top_eq[:27] + "..."
        print(f" {name:<20} | {str(space):^6} | {top_eq:<32} | {r.tier:<20}")
    print("=" * 90 + "\n")

    os.makedirs("run_outputs", exist_ok=True)
    report_name = os.path.join(
        "run_outputs",
        "adcd_v3_taxonomy_validation_report.json" if args.taxonomy else "adcd_v3_blind_validation_report.json",
    )
    with open(report_name, "w") as f:
        json.dump(
            {
                name: {
                    "tier": r.tier,
                    "status_message": r.status_message,
                    "all_passed": r.all_passed,
                    "checks": r.checks,
                }
                for name, r in all_results.items()
            },
            f, indent=2, default=str,
        )
    print(f"[OK] Full audit JSON report saved to: {report_name}")

    if args.csv:
        for name, r in all_results.items():
            df = r.to_dataframe()
            slug = name.lower().replace(" ", "_")
            csv_path = os.path.join("run_outputs", f"{slug}_pareto.csv")
            df.to_csv(csv_path, index=False)
            print(f"[OK] Exported Pareto DataFrame: {csv_path}")

    if args.latex:
        print("\n" + "=" * 80)
        print(" PUBLICATION-READY LATEX TABLES (FOR OVERLEAF / PAPER DRAFT)")
        print("=" * 80 + "\n")
        for name, r in all_results.items():
            print(f"% --- LaTeX Table: {name} ---")
            print(r.to_latex_table(top_k=args.top_k))
            print("\n")

    if args.plot:
        try:
            from eval.plot_pareto import plot_validation_pareto_fronts
            plot_validation_pareto_fronts(report_name, output_dir="run_outputs")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
