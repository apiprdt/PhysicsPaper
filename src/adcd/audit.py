"""
audit.py
========
Core audit API and imported formula verification for the ADCD Identifiability Auditor.

Integrates ClaimCard (Phase 1), GaussianLikelihoodEngine & JointFit (Phase 2),
and the Three-Tier Blind Protocol (Phase 3) into a unified, deterministic pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.optimize
import sympy as sp
from pydantic import BaseModel, Field

from adcd.claim_card import ClaimCard
from adcd.joint_fit import fit_joint_baseline_correction
from adcd.likelihood import GaussianLikelihoodEngine
from adcd.validation_v2 import CandidateFit, infer_family_from_expr, run_three_tier_audit

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
    sym_expr = None
    expr_sub = None
    sym_var = sp.Symbol(limit_var, positive=True, real=True)

    # 1. Symbolic verification via SymPy
    try:
        if isinstance(expr_str, str):
            sym_expr = sp.sympify(expr_str)
        elif isinstance(expr_str, sp.Expr):
            sym_expr = expr_str

        if sym_expr is not None:
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

            if lim_val == 0 or (lim_val.is_number and abs(float(lim_val)) < 1e-14):
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
    # Avoids 1e-15 where float64 catastrophic cancellation occurs for (1 - cos(u))/u etc.
    vals = np.array([np.nan])
    try:
        if sym_expr is not None and expr_sub is not None:
            # Substitute free parameters with a non-resonant generic positive constant (e.g. 0.73)
            # instead of 1.0 to avoid accidental poles at u * theta = 1.
            num_sub = {s: 0.73 for s in expr_sub.free_symbols if s != sym_var}
            fn_num = sp.lambdify(sym_var, expr_sub.subs(num_sub), modules=["numpy"])
        elif callable(expr_str):
            fn_num = expr_str
        else:
            fn_num = None

        if fn_num is not None:
            if limit_dir in ("0", "+0", "-0"):
                test_points = np.array([1e-3, 1e-5, 1e-7, 1e-9], dtype=np.float64)
            elif limit_dir in ("oo", "inf", "+oo"):
                test_points = np.array([1e3, 1e5, 1e7, 1e9], dtype=np.float64)
            else:
                t_val = float(limit_dir)
                test_points = t_val + np.array([1e-3, 1e-5, 1e-7, 1e-9], dtype=np.float64)

            vals = np.abs(np.array([float(fn_num(pt)) for pt in test_points], dtype=np.float64))

            # Check convergence toward 0 across the non-cancellation points
            if np.all(np.isfinite(vals)) and vals[-1] < 1e-3 and vals[-1] <= vals[0] + 1e-12:
                numerical_verified = True
    except Exception as e:
        logger.warning(f"Numerical limit check error: {e}")
        numerical_verified = False

    valid = symbolic_verified and numerical_verified
    last_val = float(vals[-1]) if len(vals) > 0 and np.isfinite(vals[-1]) else float("nan")

    if valid:
        msg = (
            f"Classical limit verified, not enforced by construction: "
            f"as {limit_var} -> {limit_dir}, Delta -> 0 (symbolic=0, |val|={last_val:.2e})."
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
    equivalence_class: List[Dict[str, Any]] = Field(default_factory=list)
    shared_series_explanation: str = ""
    adequacy_warning: str = ""
    limit_verifications: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    plain_language_summary: str = ""

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def canonical_hash(self) -> str:
        """Deterministic hash of report contents excluding report_hash itself."""
        data = self.model_dump()
        data["report_hash"] = ""

        def _normalize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
            elif isinstance(obj, (list, tuple)):
                return [_normalize(x) for x in obj]
            elif isinstance(obj, (float, np.floating)):
                f = float(obj)
                if not math.isfinite(f):
                    return str(f)
                if f == 0.0:
                    return 0.0
                return float(f"{f:.15g}")
            elif isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
                return int(obj)
            return obj

        clean_json = json.dumps(_normalize(data), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(clean_json.encode("utf-8")).hexdigest()


def _format_plain_language_summary(
    verdict: str,
    card: ClaimCard,
    best_candidate: Optional[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    shared_series: str = "",
    adequacy_warn: str = "",
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
        bic_diff = best_candidate.get("delta_bic", 0.0) if best_candidate else 0.0
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
        rival_names = [c["expr"] for c in ranked if (c["bic"] - ranked[0]["bic"]) <= 10.0]
        summary_lines.append(
            f"  • The observational data statistically rejects the uncorrected baseline, but "
            f"CANNOT tell apart competing candidate models: {rival_names}."
        )
        if shared_series:
            summary_lines.append(f"  • {shared_series}")
        else:
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

    if adequacy_warn:
        summary_lines.extend(["", f"ADEQUACY DIAGNOSTIC: {adequacy_warn}"])

    summary_lines.extend([
        "",
        "WHAT THIS VERDICT DOES NOT MEAN:",
        "  1. It does NOT claim absolute metaphysical truth of the formula outside the tested window.",
        "  2. It does NOT replace independent experimental replication or first-principles theoretical derivation.",
        "  3. It is strictly bounded by the hypothesis space, priors, and noise specifications "
        "disclosed in the Claim Card.",
    ])

    return "\n".join(summary_lines)


def _resolve_sigma_array(
    card: ClaimCard,
    data: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    y_cl: np.ndarray,
) -> np.ndarray:
    """
    Construct the measurement error vector sigma_i consistently using y_cl as the
    scale reference when available (avoiding realization bias from noisy y_obs).
    """
    if card.noise_spec.type == "column" and card.noise_spec.sigma_col:
        if card.noise_spec.sigma_col not in data:
            raise ValueError(f"Noise column '{card.noise_spec.sigma_col}' not found in data dictionary.")
        sigma = np.asarray(data[card.noise_spec.sigma_col], dtype=np.float64)
    elif card.noise_spec.type == "relative":
        noise_level = card.noise_spec.value if card.noise_spec.value is not None else 0.01
        ref_scale = np.abs(y_cl) if "y_cl" in data else np.abs(y_obs)
        sigma = np.maximum(1e-12, noise_level * ref_scale)
    else:  # absolute
        abs_val = card.noise_spec.value if card.noise_spec.value is not None else 0.01
        sigma = np.full_like(y_obs, max(1e-12, abs_val), dtype=np.float64)

    return np.maximum(1e-15, sigma)


def audit_custom_expressions(
    card: ClaimCard,
    data: Dict[str, np.ndarray],
    verbose: bool = True,
) -> AuditReport:
    """
    Audit an imported set of custom candidate formulas (from PySR, literature, or AI).
    Integrates:
      - Classical limit verification (Phase 1)
      - GaussianLikelihoodEngine & JointFit when priors exist (Phase 2)
      - Blind Three-Tier Protocol with family equivalence classes (Phase 3)
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
        if verbose:
            print(f"[{cand}] {v_res['message']}")

    y_obs = np.asarray(data["y_obs"], dtype=np.float64)

    # Evaluate baseline y_cl if not explicitly in data
    if "y_cl" in data:
        y_cl_default = np.asarray(data["y_cl"], dtype=np.float64)
    else:
        sym_base = sp.sympify(card.baseline_expr)
        base_vars = list(card.variables.keys()) + list(card.baseline_constants.keys())
        base_fn = sp.lambdify(base_vars, sym_base, "numpy")
        call_args = [np.asarray(data[v], dtype=np.float64) for v in card.variables.keys()] + [
            float(card.baseline_constants[c]) for c in card.baseline_constants.keys()
        ]
        y_cl_default = np.broadcast_to(
            np.asarray(base_fn(*call_args), dtype=np.float64), y_obs.shape
        ).copy()

    sigma = _resolve_sigma_array(card, data, y_obs, y_cl_default)
    var_data = np.asarray(data[card.limit_variable], dtype=np.float64)

    lik_engine = GaussianLikelihoodEngine(
        fit_intrinsic_scatter=card.noise_spec.fit_intrinsic_scatter,
    )

    # Null model (no correction: Delta = 0)
    # When parameter_priors are active, fit the Null model over baseline parameters phi
    # under the exact same priors so Null and Candidate models are compared fairly.
    if card.parameter_priors:
        sym_base = sp.sympify(card.baseline_expr)
        phi_names = list(card.parameter_priors.keys())
        data_var_names = [v for v in card.variables.keys() if v in data]
        fixed_consts = {k: v for k, v in card.baseline_constants.items() if k not in phi_names}

        base_sym_sub = sym_base.subs(fixed_consts)
        base_fn_phi = sp.lambdify(data_var_names + phi_names, base_sym_sub, "numpy")

        def baseline_fn_null(phi_dict: Dict[str, float], x_dict: Dict[str, np.ndarray]) -> np.ndarray:
            args = [x_dict[v] for v in data_var_names] + [phi_dict[p] for p in phi_names]
            return np.broadcast_to(
                np.asarray(base_fn_phi(*args), dtype=np.float64), y_obs.shape
            ).copy()

        def zero_corr_fn(theta_dict: Dict[str, float], x_dict: Dict[str, np.ndarray]) -> np.ndarray:
            return np.zeros_like(y_obs)

        init_phi_null = {
            p: card.baseline_constants.get(p, card.parameter_priors[p][0])
            for p in phi_names
        }
        jf_null = fit_joint_baseline_correction(
            baseline_fn=baseline_fn_null,
            corr_fn=zero_corr_fn,
            X=data,
            y_obs=y_obs,
            sigma_y=sigma,
            phi_priors=card.parameter_priors,
            init_phi=init_phi_null,
            init_theta={},
            mode=card.mode,
        )
        y_cl_null = baseline_fn_null(jf_null.phi_opt, data)
        prior_penalty_null = jf_null.prior_penalty
    else:
        y_cl_null = y_cl_default
        prior_penalty_null = 0.0

    null_lik = lik_engine.evaluate(y_obs - y_cl_null, sigma)
    bic_null = lik_engine.compute_bic(
        null_lik,
        k_params=0,
        search_space_size=1,
        prior_penalty=prior_penalty_null,
    )

    pool_size = len(card.candidate_expressions)
    ranked_dicts: List[Dict[str, Any]] = []
    candidate_fits: List[CandidateFit] = []

    for cand_str in card.candidate_expressions:
        sym_cand = sp.sympify(cand_str)
        # Identify free optimization parameters (any symbol not in data variables or baseline constants)
        known_syms = set(card.variables.keys()) | set(card.baseline_constants.keys()) | {card.limit_variable}
        free_syms = [
            s for s in sym_cand.free_symbols
            if s.name.startswith("theta_") or s.name not in known_syms
        ]
        param_names = [s.name for s in sorted(free_syms, key=lambda s: s.name)]
        k_params = len(param_names)

        eval_vars = [card.limit_variable] + param_names
        cand_fn = sp.lambdify(eval_vars, sym_cand, "numpy")

        is_degenerate = False
        prior_penalty = 0.0
        phi_fit: Dict[str, float] = {}
        joint_diag = ""

        # Case A: Joint fit with baseline nuisance parameters phi if parameter_priors are given
        if card.parameter_priors:
            sym_base = sp.sympify(card.baseline_expr)
            phi_names = list(card.parameter_priors.keys())
            data_var_names = [v for v in card.variables.keys() if v in data]
            fixed_consts = {k: v for k, v in card.baseline_constants.items() if k not in phi_names}

            base_sym_sub = sym_base.subs(fixed_consts)
            base_fn_phi = sp.lambdify(data_var_names + phi_names, base_sym_sub, "numpy")

            def baseline_fn(phi_dict: Dict[str, float], x_dict: Dict[str, np.ndarray]) -> np.ndarray:
                args = [x_dict[v] for v in data_var_names] + [phi_dict[p] for p in phi_names]
                return np.broadcast_to(
                    np.asarray(base_fn_phi(*args), dtype=np.float64), y_obs.shape
                ).copy()

            def correction_fn(theta_dict: Dict[str, float], x_dict: Dict[str, np.ndarray]) -> np.ndarray:
                t_args = [theta_dict[p] for p in param_names]
                raw_d = cand_fn(x_dict[card.limit_variable], *t_args)
                return np.broadcast_to(
                    np.asarray(raw_d, dtype=np.float64), y_obs.shape
                ).copy()

            init_phi = {
                p: card.baseline_constants.get(p, card.parameter_priors[p][0])
                for p in phi_names
            }
            init_theta = {p: 0.5 for p in param_names}

            jf_res = fit_joint_baseline_correction(
                baseline_fn=baseline_fn,
                corr_fn=correction_fn,
                X=data,
                y_obs=y_obs,
                sigma_y=sigma,
                phi_priors=card.parameter_priors,
                init_phi=init_phi,
                init_theta=init_theta,
                mode=card.mode,
            )
            opt_theta = jf_res.theta_opt
            phi_fit = jf_res.phi_opt
            is_degenerate = jf_res.is_degenerate
            prior_penalty = jf_res.prior_penalty
            joint_diag = jf_res.diagnostic_message

            y_cl_eval = baseline_fn(phi_fit, data)
            d_eval = correction_fn(opt_theta, data)
            y_pred = y_cl_eval * (1.0 + d_eval) if card.mode == "multiplicative" else y_cl_eval + d_eval

        # Case B: Fixed baseline y_cl
        else:
            if k_params == 0:
                delta_eval = np.broadcast_to(
                    np.asarray(cand_fn(var_data), dtype=np.float64), y_obs.shape
                ).copy()
                y_pred = (
                    y_cl_default * (1.0 + delta_eval)
                    if card.mode == "multiplicative"
                    else y_cl_default + delta_eval
                )
                opt_theta = {}
            else:
                def obj_loss(params: np.ndarray) -> float:
                    try:
                        with np.errstate(all="ignore"):
                            d_eval = np.broadcast_to(
                                np.asarray(cand_fn(var_data, *params), dtype=np.float64),
                                y_obs.shape,
                            )
                        if not np.all(np.isfinite(d_eval)):
                            return 1e18
                        pred = (
                            y_cl_default * (1.0 + d_eval)
                            if card.mode == "multiplicative"
                            else y_cl_default + d_eval
                        )
                        res_l = lik_engine.evaluate(y_obs - pred, sigma)
                        return res_l.twice_nll
                    except Exception:
                        return 1e18

                # Deterministic multi-start optimization
                rng = np.random.default_rng(card.seed)
                starts = [
                    np.ones(k_params, dtype=np.float64),
                    np.full(k_params, 0.1, dtype=np.float64),
                    np.full(k_params, -0.1, dtype=np.float64),
                ]
                for _ in range(3):
                    starts.append(rng.uniform(-2.0, 2.0, size=k_params))

                best_opt = None
                best_fun = np.inf
                for s_init in starts:
                    try:
                        opt_res = scipy.optimize.minimize(obj_loss, s_init, method="L-BFGS-B")
                        if opt_res.fun < best_fun:
                            best_fun = float(opt_res.fun)
                            best_opt = opt_res
                    except Exception:
                        continue

                opt_x = best_opt.x if best_opt is not None else np.zeros(k_params, dtype=np.float64)
                opt_theta = {p: float(v) for p, v in zip(param_names, opt_x)}
                delta_eval = np.broadcast_to(
                    np.asarray(cand_fn(var_data, *opt_x), dtype=np.float64), y_obs.shape
                ).copy()
                y_pred = (
                    y_cl_default * (1.0 + delta_eval)
                    if card.mode == "multiplicative"
                    else y_cl_default + delta_eval
                )

        # Evaluate likelihood and Extended BIC through GaussianLikelihoodEngine
        lik_res = lik_engine.evaluate(y_obs - y_pred, sigma)
        bic = lik_engine.compute_bic(
            lik_res,
            k_params=k_params,
            search_space_size=pool_size,
            prior_penalty=prior_penalty,
        )
        var_y = float(np.var(y_obs))
        nmse = float(np.mean((y_obs - y_pred) ** 2) / var_y) if var_y > 0 else 0.0

        fam = (
            card.candidate_families.get(cand_str)
            if card.candidate_families and cand_str in card.candidate_families
            else infer_family_from_expr(cand_str)
        )

        ranked_dicts.append({
            "expr": cand_str,
            "family": fam,
            "nmse": nmse,
            "bic": bic,
            "chi2": lik_res.chi2,
            "twice_nll": lik_res.twice_nll,
            "sigma_int": lik_res.sigma_int,
            "k_params": k_params,
            "theta_fit": opt_theta,
            "phi_fit": phi_fit,
            "is_degenerate": is_degenerate,
            "joint_diagnostic": joint_diag,
        })

        candidate_fits.append(
            CandidateFit(
                expr=cand_str,
                bic=bic,
                nmse=nmse,
                k_params=k_params,
                family=fam,
                theta_fit=opt_theta,
                y_pred=y_pred,
                is_degenerate=is_degenerate,
            )
        )

    ranked_dicts.sort(key=lambda x: x["bic"])

    # Run Phase 3 Blind Three-Tier Audit Protocol
    proto_res = run_three_tier_audit(
        candidates=candidate_fits,
        null_bic=bic_null,
        search_space_size=pool_size,
        u_var=card.limit_variable,
        limit_dir=card.limit_direction,
        check_adequacy=card.check_adequacy,
        u_data=var_data,
        y_obs=y_obs,
        sigma=sigma,
    )

    verdict = proto_res.verdict.value
    best = None
    if proto_res.best_candidate is not None:
        for d in ranked_dicts:
            if d["expr"] == proto_res.best_candidate.expr and not d["is_degenerate"]:
                best = d
                break
    if best is None and ranked_dicts:
        best = ranked_dicts[0]

    if best is not None:
        if np.isfinite(proto_res.delta_bic_vs_rival):
            best["delta_bic"] = float(proto_res.delta_bic_vs_rival)
        else:
            best["delta_bic"] = float(proto_res.delta_bic_vs_null)
        best["delta_bic_vs_null"] = float(proto_res.delta_bic_vs_null)

    eq_class_dicts = [
        {
            "expr": c.expr,
            "family": c.family,
            "bic": c.bic,
            "nmse": c.nmse,
            "k_params": c.k_params,
            "theta_fit": c.theta_fit,
        }
        for c in proto_res.equivalence_class
    ]

    lib_versions = {
        "adcd": "1.1.0-dev",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sp.__version__,
    }

    plain_summary = _format_plain_language_summary(
        verdict=verdict,
        card=card,
        best_candidate=best,
        ranked=ranked_dicts,
        shared_series=proto_res.shared_series_explanation,
        adequacy_warn=proto_res.adequacy_warning,
    )

    rep = AuditReport(
        card_hash=card.canonical_hash(),
        timestamp="2026-11-08T00:00:00Z",
        library_versions=lib_versions,
        verdict=verdict,
        best_candidate=best,
        ranked_candidates=ranked_dicts,
        equivalence_class=eq_class_dicts,
        shared_series_explanation=proto_res.shared_series_explanation,
        adequacy_warning=proto_res.adequacy_warning,
        limit_verifications=limit_verifs,
        plain_language_summary=plain_summary,
    )
    rep.report_hash = rep.canonical_hash()
    return rep


def audit(
    card: ClaimCard,
    data: Dict[str, np.ndarray],
    verbose: bool = True,
) -> AuditReport:
    """
    High-level entry point for auditing a ClaimCard on observational data.
    """
    if card.candidate_expressions:
        return audit_custom_expressions(card, data, verbose=verbose)

    # If no candidate expressions are supplied, invoke ADCD dictionary pipeline
    from adcd.anomaly_scenarios import AnomalyScenario
    from adcd.run_adcd_v3_validation_blind import run_scenario_protocol

    class CardScenario(AnomalyScenario):
        def generate_data(
            self,
            n_points: int = 200,
            noise_level: float = 0.0,
            seed: int = 42,
            domain_max: float = None,
        ):
            y_obs = np.asarray(data["y_obs"], dtype=np.float64)
            y_cl = np.asarray(data.get("y_cl", np.ones_like(y_obs)), dtype=np.float64)
            res = (y_obs / y_cl - 1.0) if card.mode == "multiplicative" else (y_obs - y_cl)
            X = {k: np.asarray(v, dtype=np.float64) for k, v in data.items() if k not in ("y_obs", "y_cl")}
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
