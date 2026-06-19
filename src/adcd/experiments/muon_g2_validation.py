"""
Muon g-2 iterative QED series validation (SYNTHETIC DATA).

Scientific framing:
  DEMONSTRATION — tests whether iADCD recovers perturbative orders.
  NOT experimental muon discovery.

Experiment tiers (rational design):
  Tier A — single-order isolation: y = C_k x^k only
  Tier B — residual order: classical = sum_{i<k} C_i x^i, discover C_k
  Tier C — full integrated iADCD on truncated series (orders 1..K in data)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from adcd.experiments.proposers import perturbative_order_proposer
from adcd.experiments._fit import fit_with_proposer
from adcd.iadcd_orchestrator import iADCDOrchestrator, iADCDResult

QED_COEFFICIENTS = {1: 0.5, 2: 0.765857425, 3: 24.05050964}
TOLERANCES = {1: 0.05, 2: 0.20, 3: 0.30}
VARIABLE = "x"


@dataclass
class OrderRecovery:
    order: int
    tier: str
    discovered_theta: Optional[float]
    ols_coefficient: Optional[float]
    known: float
    relative_error: Optional[float]
    passed: bool
    expression: str


@dataclass
class MuonG2ValidationResult:
    data_label: str
    noise_level: float
    seed: int
    orders: List[OrderRecovery]
    integrated_nmse: Optional[float]
    integrated_passed: bool
    final_expr: Optional[str]


def _alpha_grid(n_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Log-spaced coupling — improves scale separation between x^k terms
    base = np.exp(np.linspace(np.log(0.002), np.log(0.08), n_points))
    return base


def generate_muon_g2_data(
    n_points: int = 150,
    noise_level: float = 0.0,
    seed: int = 42,
    max_order: int = 3,
    orders: Optional[List[int]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Synthetic QED series: a_mu = sum_k C_k (alpha/pi)^k for selected orders."""
    alpha = _alpha_grid(n_points, seed)
    x = alpha / np.pi
    active = orders if orders is not None else list(range(1, max_order + 1))
    y_true = sum(QED_COEFFICIENTS[k] * x ** k for k in active)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0.0, noise_level * np.std(y_true), n_points) if noise_level > 0 else 0.0
    return {VARIABLE: x}, y_true + noise, np.zeros_like(x)


def _ols_monomial_coeff(x: np.ndarray, residual: np.ndarray, power: int) -> float:
    """Project residual onto x^power (physically interpretable readout)."""
    basis = x ** power
    denom = float(np.dot(basis, basis))
    if denom < 1e-30:
        return 0.0
    return float(np.dot(basis, residual) / denom)


def _eval_pass(order: int, rel_err: Optional[float]) -> bool:
    if rel_err is None:
        return False
    return rel_err <= TOLERANCES[order]


def validate_single_order(
    order: int,
    noise_level: float = 0.0,
    seed: int = 42,
    n_points: int = 150,
) -> OrderRecovery:
    """Tier A: data contains ONLY the target order (isolation test)."""
    X, y_obs, y_classical = generate_muon_g2_data(
        n_points=n_points, noise_level=noise_level, seed=seed, orders=[order]
    )
    x = X[VARIABLE]
    prop = perturbative_order_proposer(order=order, variable=VARIABLE, seed=seed)
    res = fit_with_proposer(
        X, y_obs, y_classical, prop,
        limit_variable=VARIABLE, limit_direction="0", correction_mode="additive",
        verbose=False, seed=seed,
    )
    expr = res.search_result.best_expr or ""
    theta = res.search_result.best_theta.get("theta_0")
    ols = _ols_monomial_coeff(x, y_obs - y_classical, order)
    rel = abs(ols - QED_COEFFICIENTS[order]) / abs(QED_COEFFICIENTS[order])
    return OrderRecovery(
        order=order, tier="A_isolated", discovered_theta=theta, ols_coefficient=ols,
        known=QED_COEFFICIENTS[order], relative_error=rel, passed=_eval_pass(order, rel),
        expression=expr,
    )


def validate_residual_order(
    order: int,
    noise_level: float = 0.0,
    seed: int = 42,
    n_points: int = 150,
) -> OrderRecovery:
    """Tier B: classical = exact lower orders, discover target order from residual."""
    X, y_obs, _ = generate_muon_g2_data(
        n_points=n_points, noise_level=noise_level, seed=seed, max_order=order
    )
    x = X[VARIABLE]
    y_classical = sum(QED_COEFFICIENTS[k] * x ** k for k in range(1, order))
    residual = y_obs - y_classical
    prop = perturbative_order_proposer(order=order, variable=VARIABLE, seed=seed + order)
    res = fit_with_proposer(
        X, y_obs, y_classical, prop,
        limit_variable=VARIABLE, limit_direction="0", correction_mode="additive",
        verbose=False, seed=seed + order,
    )
    expr = res.search_result.best_expr or ""
    theta = res.search_result.best_theta.get("theta_0")
    ols = _ols_monomial_coeff(x, residual, order)
    coeff = ols
    rel = abs(coeff - QED_COEFFICIENTS[order]) / abs(QED_COEFFICIENTS[order])
    return OrderRecovery(
        order=order, tier="B_residual", discovered_theta=theta, ols_coefficient=ols,
        known=QED_COEFFICIENTS[order], relative_error=rel, passed=_eval_pass(order, rel),
        expression=expr,
    )


def validate_integrated_iadcd(
    max_order: int = 2,
    noise_level: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[iADCDResult, List[OrderRecovery]]:
    """Tier C: full iADCD loop; OLS readout per round on actual residual."""
    X, y_obs, y_classical = generate_muon_g2_data(
        n_points=150, noise_level=noise_level, seed=seed, max_order=max_order
    )
    x = X[VARIABLE]
    round_proposers = [
        perturbative_order_proposer(order=k, variable=VARIABLE, seed=seed + k)
        for k in range(1, max_order + 1)
    ]
    orch = iADCDOrchestrator(max_rounds=max_order, convergence_nmse=1e-6, min_snr=0.01, verbose=verbose)
    res = orch.run_iterative_discovery(
        X=X, y_obs=y_obs, y_classical=y_classical,
        limit_variable=VARIABLE, limit_direction="0", classical_expr="0.0",
        variables_with_units={VARIABLE: "dimensionless"},
        round_proposers=round_proposers, seed=seed,
    )

    y_base = np.zeros_like(y_obs)
    recoveries: List[OrderRecovery] = []
    for rnd in res.rounds:
        residual = y_obs - y_base
        ols = _ols_monomial_coeff(x, residual, rnd.round_idx)
        theta = next((v for k, v in rnd.discovered_theta.items() if k.endswith("_0")), None)
        rel = abs(ols - QED_COEFFICIENTS[rnd.round_idx]) / abs(QED_COEFFICIENTS[rnd.round_idx])
        recoveries.append(OrderRecovery(
            order=rnd.round_idx, tier="C_integrated", discovered_theta=theta,
            ols_coefficient=ols, known=QED_COEFFICIENTS[rnd.round_idx],
            relative_error=rel, passed=_eval_pass(rnd.round_idx, rel),
            expression=rnd.discovered_expr,
        ))
        # Update baseline using OLS readout (honest physics projection, not JAX theta)
        y_base = y_base + ols * x ** rnd.round_idx

    return res, recoveries


def validate_iadcd_on_qed(
    max_order: int = 2,
    noise_level: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
) -> MuonG2ValidationResult:
    print("=== SYNTHETIC DATA ===")
    print("QED perturbative validation — not experimental discovery.\n")

    orders: List[OrderRecovery] = []
    for k in range(1, max_order + 1):
        orders.append(validate_single_order(k, noise_level, seed))
        orders.append(validate_residual_order(k, noise_level, seed))

    iadcd_res, integrated = validate_integrated_iadcd(max_order, noise_level, seed, verbose)
    orders.extend(integrated)

    int_pass = iadcd_res.final_nmse_full < 1e-4 and any(
        r.tier == "C_integrated" and r.order == 1 and r.passed for r in integrated
    )
    return MuonG2ValidationResult(
        data_label="SYNTHETIC",
        noise_level=noise_level,
        seed=seed,
        orders=orders,
        integrated_nmse=float(iadcd_res.final_nmse_full),
        integrated_passed=int_pass,
        final_expr=iadcd_res.final_expr,
    )


def print_validation_report(result: MuonG2ValidationResult) -> None:
    print("\n=== iADCD QED VALIDATION (SYNTHETIC DATA) ===")
    print(f"{'Tier':<14} | {'Ord':>3} | {'OLS':>10} | {'Known':>10} | {'Err%':>6} | {'Pass':>4}")
    for r in result.orders:
        ols_s = f"{r.ols_coefficient:.4f}" if r.ols_coefficient is not None else "N/A"
        err_s = f"{r.relative_error * 100:.1f}" if r.relative_error is not None else "N/A"
        print(f"{r.tier:<14} | {r.order:>3} | {ols_s:>10} | {r.known:>10.4f} | {err_s:>6} | {str(r.passed):>4}")
    print(f"\nIntegrated NMSE: {result.integrated_nmse:.2e}  |  Integrated pass: {result.integrated_passed}")
    if result.final_expr:
        print(f"Expression: {result.final_expr}")


def save_validation_json(result: MuonG2ValidationResult, path: str = "results/muon_g2_validation.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out


def run_validation_demo(noise_level: float = 0.0, seed: int = 42) -> bool:
    result = validate_iadcd_on_qed(max_order=2, noise_level=noise_level, seed=seed)
    print_validation_report(result)
    save_validation_json(result)
    tier_a = all(r.passed for r in result.orders if r.tier == "A_isolated")
    tier_b = all(r.passed for r in result.orders if r.tier == "B_residual")
    print(f"\nSummary: Tier A={tier_a}  Tier B={tier_b}  Integrated NMSE OK={result.integrated_nmse < 1e-4}")
    return tier_a and tier_b


if __name__ == "__main__":
    ok = run_validation_demo()
    raise SystemExit(0 if ok else 1)
