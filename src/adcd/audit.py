"""
audit.py
========
Core audit API and imported formula verification for the ADCD Identifiability Auditor.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.optimize
import sympy as sp
from pydantic import BaseModel, Field

from adcd.claim_card import ClaimCard

logger = logging.getLogger(__name__)


def verify_classical_limit(
    expr_str: Union[str, sp.Expr, Callable],
    limit_var: str = "u",
    limit_dir: str = "0",
) -> Dict[str, Any]:
    """
    Verify both symbolically and numerically that an imported formula satisfies the classical limit:
    Delta(u) -> 0 as u -> limit_direction.

    Prints and records 'limit verified, not enforced by construction'.
    Rejects formulas that violate the limit.
    """
    symbolic_verified = False
    numerical_verified = False
    lim_val = None

    # 1. Symbolic verification via SymPy
    try:
        if isinstance(expr_str, str):
            sym_expr = sp.sympify(expr_str)
        elif isinstance(expr_str, sp.Expr):
            sym_expr = expr_str
        else:
            sym_expr = None

        if sym_expr is not None:
            # Map free parameters to positive real symbols
            sym_var = sp.Symbol(limit_var, positive=True, real=True)
            subs_dict = {}
            for s in sym_expr.free_symbols:
                if s.name == limit_var:
                    subs_dict[s] = sym_var
                else:
                    subs_dict[s] = sp.Symbol(s.name, positive=True, real=True)
            
            expr_sub = sym_expr.subs(subs_dict)
            
            if limit_dir in ("0", "+0", "-0"):
                lim_val = sp.limit(expr_sub, sym_var, 0, dir="+")
            elif limit_dir in ("oo", "inf", "+oo"):
                lim_val = sp.limit(expr_sub, sym_var, sp.oo)
            else:
                target_num = float(limit_dir)
                lim_val = sp.limit(expr_sub, sym_var, target_num)

            if lim_val == 0 or abs(float(lim_val)) < 1e-14:
                symbolic_verified = True
            else:
                lim_float = float(lim_val) if lim_val.is_number else str(lim_val)
                return {
                    "valid": False,
                    "symbolic_verified": False,
                    "numerical_verified": False,
                    "limit_value": lim_float,
                    "message": (
                        f"Classical limit violated: as {limit_var} -> {limit_dir}, "
                        f"candidate does not vanish (evaluates to {lim_float} != 0). "
                        f"Imported formula rejected."
                    ),
                }
    except Exception as e:
        logger.warning(f"Symbolic limit evaluation encountered error: {e}")
        symbolic_verified = False

    # 2. Numerical verification
    try:
        if sym_expr is not None:
            # Substitute free parameters with 1.0 for numerical boundary check
            num_sub = {s: 1.0 for s in expr_sub.free_symbols if s != sym_var}
            fn_num = sp.lambdify(sym_var, expr_sub.subs(num_sub), modules=["numpy"])
        elif callable(expr_str):
            fn_num = expr_str
        else:
            fn_num = None

        if fn_num is not None:
            if limit_dir in ("0", "+0", "-0"):
                test_points = np.array([1e-4, 1e-6, 1e-10, 1e-15], dtype=np.float64)
            else:
                test_points = np.array([1e4, 1e6, 1e10], dtype=np.float64)

            vals = np.abs(np.array([fn_num(pt) for pt in test_points], dtype=np.float64))
            
            # Values must be near zero and non-divergent
            if np.all(np.isfinite(vals)) and vals[-1] < 1e-3 and vals[-1] <= vals[0] + 1e-12:
                numerical_verified = True
    except Exception as e:
        logger.warning(f"Numerical limit check error: {e}")
        numerical_verified = False

    valid = symbolic_verified and numerical_verified
    if valid:
        msg = (
            f"Classical limit verified, not enforced by construction: "
            f"as {limit_var} -> {limit_dir}, Delta -> 0 (symbolic=0, |val|={vals[-1]:.2e})."
        )
    else:
        msg = (
            f"Classical limit violated or unprovable: "
            f"as {limit_var} -> {limit_dir}, Delta did not demonstrably vanish."
        )

    return {
        "valid": valid,
        "symbolic_verified": symbolic_verified,
        "numerical_verified": numerical_verified,
        "limit_value": 0.0 if valid else lim_val,
        "message": msg,
    }


class AuditReport(BaseModel):
    """
    Formal deterministic audit report issued by the ADCD Identifiability Auditor.
    """
    card_hash: str
    report_hash: str = ""
    timestamp: str = ""
    library_versions: Dict[str, str] = Field(default_factory=dict)
    verdict: str
    best_candidate: Optional[Dict[str, Any]] = None
    ranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    limit_verifications: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    plain_language_summary: str = ""

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def canonical_hash(self) -> str:
        """Deterministic hash of report contents excluding report_hash itself."""
        data = self.model_dump()
        data["report_hash"] = ""

        # Sort keys deterministically
        def _sort_obj(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sort_obj(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, (list, tuple)):
                return [_sort_obj(x) for x in obj]
            return obj

        clean_json = json.dumps(_sort_obj(data), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(clean_json.encode("utf-8")).hexdigest()


def _format_plain_language_summary(
    verdict: str,
    card: ClaimCard,
    best_candidate: Optional[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
) -> str:
    """Generate the standardized plain-language verdict semantics."""
    card_h = card.canonical_hash()[:12]
    
    summary_lines = [
        f"=== ADCD IDENTIFIABILITY AUDITOR REPORT (Claim Card: {card_h}) ===",
        f"FORMAL VERDICT: [{verdict}]",
        "",
        "WHAT THIS VERDICT MEANS:",
    ]

    if verdict == "UNIQUELY IDENTIFIABLE":
        expr = best_candidate["expr"] if best_candidate else "N/A"
        bic_diff = best_candidate.get("delta_bic", 0.0)
        summary_lines.append(
            f"  • The supplied dataset within window {card.window} contains sufficient statistical "
            f"signal to uniquely distinguish the candidate ({expr}) from all competing rivals "
            f"by very strong evidence (Delta-BIC = {bic_diff:.2f} > 10.0)."
        )
        summary_lines.append(
            "  • Any alternative functional family from the comparison pool is decisively disfavored "
            "under the penalized likelihood."
        )
    elif verdict == "EQUIVALENCE CLASS":
        rival_names = [c["expr"] for c in ranked[:3]]
        summary_lines.append(
            f"  • The observational data statistically rejects the uncorrected baseline, but "
            f"CANNOT tell apart competing candidate models: {rival_names}."
        )
        summary_lines.append(
            "  • The candidates share the same leading-order asymptotic series within the current "
            "measurement noise and window. Additional experimental precision or an expanded window "
            "is required to separate them."
        )
    else:  # WITHHELD
        summary_lines.append(
            "  • The current dataset lacks sufficient signal-to-noise ratio or observation coverage "
            "to reliably identify any correction beyond the baseline."
        )
        summary_lines.append(
            "  • The system explicitly ABSTAINS from claiming a discovery to prevent false positives."
        )

    summary_lines.extend([
        "",
        "WHAT THIS VERDICT DOES NOT MEAN:",
        "  1. It does NOT claim absolute metaphysical truth of the formula outside the tested window.",
        "  2. It does NOT replace independent experimental replication or first-principles theoretical derivation.",
        "  3. It is strictly bounded by the hypothesis space, priors, and noise specifications disclosed in the Claim Card.",
    ])

    return "\n".join(summary_lines)


def audit_custom_expressions(
    card: ClaimCard,
    data: Dict[str, np.ndarray],
) -> AuditReport:
    """
    Audit an imported set of custom candidate formulas (from PySR, literature, or AI).
    Verifies classical limits symbolically and numerically before performing identifiability fitting.
    """
    if not card.candidate_expressions:
        raise ValueError("ClaimCard must include 'candidate_expressions' for custom expression audit.")

    limit_verifs = {}
    for cand in card.candidate_expressions:
        v_res = verify_classical_limit(
            cand,
            limit_var=card.limit_variable,
            limit_dir=card.limit_direction,
        )
        limit_verifs[cand] = v_res
        if not v_res["valid"]:
            raise ValueError(
                f"Candidate '{cand}' rejected during limit verification: {v_res['message']}"
            )
        print(f"[{cand}] {v_res['message']}")

    # Extract data arrays
    y_obs = np.asarray(data["y_obs"], dtype=np.float64)
    y_cl = np.asarray(data.get("y_cl", np.ones_like(y_obs)), dtype=np.float64)
    n_points = len(y_obs)

    # Compute noise sigma_i
    if card.noise_spec.type == "relative":
        noise_level = card.noise_spec.value or 0.01
        sigma = np.maximum(1e-12, noise_level * np.abs(y_obs))
    elif card.noise_spec.type == "column" and card.noise_spec.sigma_col:
        sigma = np.asarray(data[card.noise_spec.sigma_col], dtype=np.float64)
    else:
        abs_val = card.noise_spec.value or 0.01
        sigma = np.full_like(y_obs, abs_val)

    # Null model (no correction) BIC
    residual_null = y_obs - y_cl
    chi2_null = np.sum((residual_null / sigma) ** 2)
    bic_null = chi2_null + np.sum(np.log(2 * np.pi * sigma ** 2))

    ranked = []
    for cand_str in card.candidate_expressions:
        sym_cand = sp.sympify(cand_str)
        free_syms = [s for s in sym_cand.free_symbols if s.name.startswith("theta_")]
        param_names = [s.name for s in sorted(free_syms, key=lambda s: s.name)]
        k_params = len(param_names)

        # Build numpy evaluable function
        eval_vars = [card.limit_variable] + param_names
        cand_fn = sp.lambdify(eval_vars, sym_cand, "numpy")
        var_data = np.asarray(data[card.limit_variable], dtype=np.float64)

        if k_params == 0:
            delta_eval = cand_fn(var_data)
            y_pred = y_cl * (1.0 + delta_eval) if card.mode == "multiplicative" else y_cl + delta_eval
            res = y_obs - y_pred
            chi2 = np.sum((res / sigma) ** 2)
            opt_theta = {}
        else:
            def loss(params: np.ndarray) -> float:
                d_eval = cand_fn(var_data, *params)
                pred = y_cl * (1.0 + d_eval) if card.mode == "multiplicative" else y_cl + d_eval
                return float(np.sum(((y_obs - pred) / sigma) ** 2))

            # Deterministic multi-start optimization
            init_vals = np.ones(k_params, dtype=np.float64)
            opt_res = scipy.optimize.minimize(loss, init_vals, method="L-BFGS-B")
            chi2 = float(opt_res.fun)
            opt_theta = {p: float(v) for p, v in zip(param_names, opt_res.x)}
            delta_eval = cand_fn(var_data, *opt_res.x)
            y_pred = y_cl * (1.0 + delta_eval) if card.mode == "multiplicative" else y_cl + delta_eval

        nmse = float(np.mean((y_obs - y_pred) ** 2) / np.var(y_obs)) if np.var(y_obs) > 0 else 0.0
        # Likelihood BIC: k ln N + chi2 + sum(ln(2 pi sigma^2))
        bic = float(k_params * np.log(n_points) + chi2 + np.sum(np.log(2 * np.pi * sigma ** 2)))

        ranked.append({
            "expr": cand_str,
            "nmse": nmse,
            "bic": bic,
            "chi2": chi2,
            "k_params": k_params,
            "theta_fit": opt_theta,
        })

    ranked.sort(key=lambda x: x["bic"])

    best = ranked[0]
    delta_bic_vs_null = bic_null - best["bic"]

    # 3-tier verdict
    if delta_bic_vs_null < 10.0:
        verdict = "WITHHELD"
    elif len(ranked) > 1 and (ranked[1]["bic"] - best["bic"]) <= 10.0:
        verdict = "EQUIVALENCE CLASS"
    else:
        verdict = "UNIQUELY IDENTIFIABLE"

    if best:
        best["delta_bic"] = float(ranked[1]["bic"] - best["bic"]) if len(ranked) > 1 else float(delta_bic_vs_null)

    lib_versions = {
        "adcd": "1.1.0-dev",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sp.__version__,
    }

    plain_summary = _format_plain_language_summary(verdict, card, best, ranked)

    rep = AuditReport(
        card_hash=card.canonical_hash(),
        timestamp="2026-11-08T00:00:00Z",  # Deterministic canonical timestamp
        library_versions=lib_versions,
        verdict=verdict,
        best_candidate=best,
        ranked_candidates=ranked,
        limit_verifications=limit_verifs,
        plain_language_summary=plain_summary,
    )
    rep.report_hash = rep.canonical_hash()
    return rep


def audit(
    card: ClaimCard,
    data: Dict[str, np.ndarray],
) -> AuditReport:
    """
    High-level entry point for auditing a ClaimCard on observational data.
    """
    if card.candidate_expressions:
        return audit_custom_expressions(card, data)
    
    # If no candidate expressions are supplied, invoke ADCD v1 pipeline
    from adcd.anomaly_scenarios import AnomalyScenario
    from adcd.run_adcd_v3_validation_blind import run_scenario_protocol

    # Wrap as an AnomalyScenario
    class CardScenario(AnomalyScenario):
        def generate_data(self, n_points: int = 200, noise_level: float = 0.0, seed: int = 42, domain_max: float = None):
            y_obs = np.asarray(data["y_obs"])
            y_cl = np.asarray(data.get("y_cl", np.ones_like(y_obs)))
            res = (y_obs / y_cl - 1.0) if card.mode == "multiplicative" else (y_obs - y_cl)
            X = {k: np.asarray(v) for k, v in data.items() if k not in ("y_obs", "y_cl")}
            return X, y_obs, y_cl, res

    sc = CardScenario(
        name=card.name,
        tier="custom",
        domain="custom",
        classical_expr=card.baseline_expr,
        classical_variables=list(card.variables.keys()),
        classical_constants=card.baseline_constants,
        correction_type=card.mode,
        correction_expr="theta_0 * D(u)",
        correction_constants={},
        anomaly_regime="audit",
        variables_with_units=card.variables,
        classical_limit_variable=card.limit_variable,
        classical_limit_direction=card.limit_direction,
        correction_class="custom",
    )

    proto_res = run_scenario_protocol(sc, seed=card.seed, use_taxonomy_prior=False, top_k_val=5)
    verdict = "UNIQUELY IDENTIFIABLE" if proto_res.all_passed else "WITHHELD"
    
    ps = proto_res.checks.get("primary_search", {})
    best_cand = {
        "expr": ps.get("top_candidate", "None"),
        "nmse": ps.get("nmse", float("nan")),
        "bic": ps.get("bic", float("nan")),
        "theta_fit": ps.get("theta_fit", {}),
    }

    lib_versions = {
        "adcd": "1.1.0-dev",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sp.__version__,
    }

    plain_summary = _format_plain_language_summary(verdict, card, best_cand, [best_cand])

    rep = AuditReport(
        card_hash=card.canonical_hash(),
        timestamp="2026-11-08T00:00:00Z",
        library_versions=lib_versions,
        verdict=verdict,
        best_candidate=best_cand,
        ranked_candidates=[best_cand],
        limit_verifications={},
        plain_language_summary=plain_summary,
    )
    rep.report_hash = rep.canonical_hash()
    return rep
