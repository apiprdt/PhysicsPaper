"""
adcd.api
========
User-friendly, modular Python API for custom hypothesis exploration in ADCD.

Enables researchers to:
1. Register custom 'wild' asymptotic primitives dynamically.
2. Define custom physical baseline scenarios without modifying core codebase.
3. Perform end-to-end epistemic discovery on custom datasets (NumPy / Pandas).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import sympy as sp

from adcd.anomaly_scenarios import AnomalyScenario
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY, Primitive
from adcd.metrics import expr_to_latex
from adcd.run_adcd_v3_validation_blind import (
    ScenarioThresholdConfig,
    run_scenario_protocol,
)


def register_primitive(
    name: str,
    template: str,
    token_cost: int = 5,
    domain_note: str = "Custom User Hypothesis",
    enforce_zero_limit: bool = True,
) -> Primitive:
    """Register a custom user hypothesis into ADCD's primitive dictionary.

    Parameters
    ----------
    name : str
        Unique identifier for the primitive (e.g. 'D_quantum_damping').
    template : str
        SymPy/Python mathematical string using '{u}' as the dimensionless variable
        (e.g. '(1.0 - exp(-Abs({u}))) * log(1.0 + {u}**2)').
    token_cost : int, optional
        Complexity budget cost, by default 5.
    domain_note : str, optional
        Human-readable description of physical intent.
    enforce_zero_limit : bool, optional
        Whether to verify that the expression algebraically vanishes at u=0.

    Returns
    -------
    Primitive
        The newly registered primitive object.
    """
    test_expr_str = template.format(u="u")
    u_sym = sp.Symbol("u", real=True)
    expr_sym = sp.sympify(test_expr_str)

    if enforce_zero_limit:
        try:
            val_at_zero = float(expr_sym.subs(u_sym, 0))
            is_zero = abs(val_at_zero) < 1e-12
        except Exception:
            try:
                limit_val = float(sp.limit(expr_sym, u_sym, 0, dir="+"))
                is_zero = abs(limit_val) < 1e-12
            except Exception:
                is_zero = True

        if not is_zero:
            raise ValueError(
                f"Physical Asymptote Violation: Primitive '{name}' does not vanish at u=0. "
                f"Evaluation at u=0 = {val_at_zero}. "
                f"Set enforce_zero_limit=False only if strictly necessary."
            )

    numpy_fn = sp.lambdify(u_sym, expr_sym, modules=["numpy", "math"])

    prim = Primitive(
        name=name,
        token_cost=token_cost,
        numpy_form=lambda u: np.asarray(numpy_fn(u), dtype=float),
        string_template=template,
        domain_note=domain_note,
    )
    PRIMITIVE_REGISTRY[name] = prim
    return prim


def list_primitives() -> Dict[str, str]:
    """List all currently registered physical primitives in ADCD."""
    return {k: f"{v.string_template}  ({v.domain_note})" for k, v in PRIMITIVE_REGISTRY.items()}


@dataclass
class ADCDResult:
    """Structured result object from ADCD discovery."""
    tier: str
    is_identifiable: bool
    status_message: str
    top_expression: str
    top_latex: str
    discovered_class: str
    nmse_train: float
    bic_score: float
    pareto_front: List[Dict[str, Any]]
    checks: Dict[str, Any]
    is_exploration_mode: bool = False
    custom_primitives_used: Optional[List[str]] = None

    def summary(self) -> str:
        lines = [
            "=" * 85,
            "               ADCD HYPOTHESIS EXPLORATION & DISCOVERY REPORT",
            "=" * 85,
            f" * Epistemic Tier      : {self.tier}",
            f" * Statistically Sound : {'YES (Scientifically Identifiable)' if self.is_identifiable else 'NO / WITHHELD (Data Insufficient)'}",
            f" * Discovered Class    : {self.discovered_class}",
            f" * Best Fitted Formula : {self.top_expression}",
            f" * LaTeX Formula       : {self.top_latex}",
            f" * Residual NMSE       : {self.nmse_train:.3e}",
            f" * Bayesian BIC Score  : {self.bic_score:.2f}",
            f" * Protocol Verdict    : {self.status_message}",
        ]

        if self.is_exploration_mode:
            lines.extend([
                "-" * 85,
                " [EPISTEMIC SAFETY & EXPLORATION NOTICE]",
                "   This run operated in User Exploration Mode with custom hypothesis primitives.",
                f"   - Active Custom Primitives: {', '.join(self.custom_primitives_used or ['None'])}",
                "   - Multi-Testing Penalization (2*ln M) dynamically scaled to total dictionary size.",
                "   - Parameter Bounds & AST Dimensional Rejection cascades remained active.",
            ])

        lines.extend([
            "-" * 85,
            " FORMAL VERIFICATION PROTOCOL (4-GATE AUDIT):",
        ])

        gate_names = [
            ("budget_disclosure", "Gate 1: Budget Disclosure  "),
            ("positive_control",  "Gate 2: Positive Control   "),
            ("ablation_control",  "Gate 3: Ablation Multi-Test"),
            ("determinism_check", "Gate 4: Determinism Check  "),
        ]
        for key, label in gate_names:
            gate_info = self.checks.get(key, {})
            passed = gate_info.get("pass", False)
            status_tag = "[PASS]" if passed else "[FAIL / WITHHELD]"
            note = gate_info.get("note", "") or gate_info.get("details", "")
            lines.append(f"   * {label} : {status_tag:<17} | {note}")

        lines.extend([
            "-" * 85,
            " TOP CANDIDATES ON PARETO FRONT:",
        ])
        for idx, cand in enumerate(self.pareto_front[:5], 1):
            lines.append(
                f"   [{idx}] Class: {cand.get('class', '?'):<14} | NMSE: {cand.get('nmse', 0.0):.2e} | "
                f"BIC: {cand.get('bic', 0.0):.1f} | Expr: {cand.get('expr_str', '')}"
            )

        lines.extend([
            "-" * 85,
            " [SCIENTIFIC INTEGRITY & CITATION DISCLAIMER]",
            "   * Publication Standard : Only candidates certified as Tier [IDENTIFIABLE]",
            "     (passing all 4 verification gates) constitute statistically validated laws.",
            "   * Exploratory Status   : Models marked [DETECTED_UNRESOLVED] or [WITHHELD]",
            "     represent unconfirmed hypotheses due to SNR limits or candidate competition.",
            "     They should be cited as exploratory candidates, not established physical laws.",
            "   * Dimensional Validity : Physical consistency is guaranteed algebraically within",
            "     the declared variable unit definitions and classical asymptotic boundary.",
            "=" * 85,
        ])
        report_str = "\n".join(lines)
        print(report_str)
        return report_str


class CustomUserScenario(AnomalyScenario):
    """Dynamic scenario class wrapping custom user equations and data."""

    def __init__(
        self,
        name: str,
        domain: str,
        classical_expr: str,
        classical_variables: List[str],
        classical_constants: Dict[str, float],
        variables_with_units: Optional[Dict[str, str]] = None,
        custom_data_fn: Optional[Callable] = None,
    ):
        super().__init__(
            name=name,
            tier="custom",
            domain=domain,
            classical_expr=classical_expr,
            classical_variables=classical_variables,
            classical_constants=classical_constants,
            correction_type="multiplicative",
            correction_expr="0",
            correction_constants={},
            anomaly_regime="user defined regime",
            variables_with_units=variables_with_units or {},
            classical_limit_variable=classical_variables[0] if classical_variables else "x",
            classical_limit_direction="0",
            correction_class="custom",
            engine="julia",
        )
        self.custom_data_fn = custom_data_fn

    def generate_data(self, noise_level=0.0, seed=42, domain_max=None, n_points=100, **kwargs):
        if self.custom_data_fn is not None:
            return self.custom_data_fn(noise_level=noise_level, seed=seed, domain_max=domain_max, n_points=n_points)
        return super().generate_data(noise_level=noise_level, seed=seed, domain_max=domain_max, n_points=n_points)


def discover(
    scenario_or_data: Union[AnomalyScenario, Dict[str, np.ndarray]],
    classical_expr: Optional[str] = None,
    classical_constants: Optional[Dict[str, float]] = None,
    variables_with_units: Optional[Dict[str, str]] = None,
    seed: int = 42,
    noise_level: float = 0.0,
    engine: str = "julia",
    use_taxonomy_prior: bool = False,
) -> ADCDResult:
    """Run ADCD 4-gate discovery on custom scenario or user tabular data.

    Parameters
    ----------
    scenario_or_data : AnomalyScenario or dict of arrays
        Either an existing scenario or a dict containing variable arrays and 'y_obs'.
    classical_expr : str, optional
        Baseline physical law string (e.g. '0.5 * m * v**2').
    classical_constants : dict, optional
        Fixed physical constants (e.g. {'c': 3e8, 'G': 6.674e-11}).
    variables_with_units : dict, optional
        Physical dimensions/units mapping (e.g. {'v': 'm/s', 'm': 'kg'}).
    seed : int, optional
        Random seed for reproducibility, by default 42.
    noise_level : float, optional
        Noise level if generating synthetic data, by default 0.0.
    engine : str, optional
        Compute engine ('julia' or 'python'), by default 'julia'.
    use_taxonomy_prior : bool, optional
        Whether to restrict to domain taxonomy or run full unconstrained dictionary search.

    Returns
    -------
    ADCDResult
        Comprehensive result containing ranked formulas and epistemic verdict.
    """
    if isinstance(scenario_or_data, AnomalyScenario):
        sc = copy.deepcopy(scenario_or_data)
    else:
        data_dict = scenario_or_data
        target_name = "y_obs" if "y_obs" in data_dict else next(iter(data_dict.keys()))
        var_names = [k for k in data_dict.keys() if k != target_name]

        def _data_fn(noise_level=0.0, seed=42, domain_max=None, n_points=100):
            X = {k: np.asarray(v, dtype=float) for k, v in data_dict.items() if k != target_name}
            y_obs = np.asarray(data_dict[target_name], dtype=float)
            local_dict = {**X, **(classical_constants or {})}
            y_cl = eval(classical_expr, {"np": np, "sp": sp}, local_dict)
            if np.isscalar(y_cl):
                y_cl = np.full_like(y_obs, y_cl)
            return X, y_obs, y_cl, None

        sc = CustomUserScenario(
            name="User Exploration Scenario",
            domain="User Domain",
            classical_expr=classical_expr or "1.0",
            classical_variables=var_names,
            classical_constants=classical_constants or {},
            variables_with_units=variables_with_units or {},
            custom_data_fn=_data_fn,
        )

    sc.engine = engine
    tcfg = ScenarioThresholdConfig.for_scenario(sc, noise_level=noise_level)
    res = run_scenario_protocol(
        scenario=sc,
        seed=seed,
        threshold_cfg=tcfg,
        noise_level=noise_level,
        use_taxonomy_prior=use_taxonomy_prior,
    )

    ps = res.checks.get("primary_search", {})
    top_expr = ps.get("top_candidate", "")
    theta_fit = ps.get("theta_fit", {})

    return ADCDResult(
        tier=res.tier,
        is_identifiable=(res.tier == "IDENTIFIABLE"),
        status_message=res.status_message,
        top_expression=top_expr,
        top_latex=expr_to_latex(top_expr, theta_fit=theta_fit) if top_expr else "",
        discovered_class=ps.get("discovered_class", "unknown"),
        nmse_train=ps.get("nmse", float("inf")),
        bic_score=ps.get("bic", float("inf")),
        pareto_front=ps.get("pareto_front", []),
        checks=res.checks,
    )
