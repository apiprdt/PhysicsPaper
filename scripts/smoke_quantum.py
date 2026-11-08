"""
Smoke test: Planck and Einstein through ADCD v1 (Phase 0b)
Tests 5 cases:
- Planck x <= 0.3
- Planck x <= 1.0
- Planck x <= 3.0
- Einstein x <= 0.3
- Einstein x <= 1.0
Runs through ADCD v1 pipeline without any rewrite.
"""

from __future__ import annotations

import os
import sys
import io

# Force UTF-8 unbuffered output regardless of Windows console codepage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", write_through=True)

import numpy as np
import sympy as sp

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adcd.anomaly_scenarios import AnomalyScenario
from adcd.run_adcd_v3_validation_blind import run_scenario_protocol, _run_search
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY


class QuantumScenario(AnomalyScenario):
    """Synthetic quantum known-answer scenario for Phase 0b smoke testing."""
    def __init__(
        self,
        name: str,
        hi: float,
        is_planck: bool = True,
        domain: str = "boltzmann_thermodynamics",
    ):
        self.hi = hi
        self.is_planck = is_planck
        if is_planck:
            corr_expr = "x / (exp(x) - 1.0) - 1.0"
            corr_class = "exponential"
        else:
            corr_expr = "(x**2 * exp(x)) / ((exp(x) - 1.0)**2) - 1.0"
            corr_class = "rational"

        super().__init__(
            name=name,
            tier="synthetic",
            domain=domain,
            classical_expr="1.0",
            classical_variables=["x"],
            classical_constants={},
            correction_type="multiplicative",
            correction_expr=corr_expr,
            correction_constants={},
            anomaly_regime="quantum",
            variables_with_units={"x": "dimensionless"},
            classical_limit_variable="x",
            classical_limit_direction="0",
            correction_class=corr_class,
        )

    def generate_data(self, n_points: int = 200, noise_level: float = 0.01, seed: int = 42, domain_max: float = None):
        rng = np.random.RandomState(seed)
        hi_val = domain_max if domain_max is not None else self.hi
        x = np.linspace(0.01, hi_val, n_points)
        
        if self.is_planck:
            delta_true = x / np.expm1(x) - 1.0
        else:
            ex = np.exp(x)
            delta_true = (x**2 * ex) / (np.expm1(x)**2) - 1.0
            
        y_classical = np.ones(n_points, dtype=np.float64)
        y_true = y_classical * (1.0 + delta_true)
        
        if noise_level > 0.0:
            noise = rng.normal(0.0, noise_level, size=n_points)
            y_obs = y_true * (1.0 + noise)
        else:
            y_obs = y_true.copy()
            
        residual = y_obs / y_classical - 1.0
        X = {"x": x}
        return X, y_obs, y_classical, residual


def run_all_smoke_tests():
    cases = [
        ("Planck x<=0.3", 0.3, True),
        ("Planck x<=1.0", 1.0, True),
        ("Planck x<=3.0", 3.0, True),
        ("Einstein x<=0.3", 0.3, False),
        ("Einstein x<=1.0", 1.0, False),
    ]

    print("=" * 80)
    print("PHASE 0b SMOKE TEST: QUANTUM KNOWN-ANSWER CASES THROUGH ADCD v1")
    print("=" * 80)

    results = {}
    for name, hi, is_planck in cases:
        sc = QuantumScenario(name=name, hi=hi, is_planck=is_planck)
        
        # 1. Stand-in theoretical calculations
        X, y_obs, y_cl, res = sc.generate_data(n_points=200, noise_level=0.01, seed=42)
        x = X["x"]
        if is_planck:
            delta_true = x / np.expm1(x) - 1.0
        else:
            ex = np.exp(x)
            delta_true = (x**2 * ex) / (np.expm1(x)**2) - 1.0
            
        max_corr = float(np.max(np.abs(delta_true)) * 100)
        sigma = 0.01 * (1.0 + delta_true)
        delta_chi2_null = float(np.sum((delta_true / sigma)**2))
        noise_var = float(np.mean(sigma**2))
        sig_var = float(np.var(delta_true))
        nmse_floor = float(noise_var / sig_var) if sig_var > 0 else float("nan")

        print(f"\n--- Testing Scenario: {name} (hi={hi}) ---")
        print(f"Theoretical: Max corr={max_corr:.1f}%, Δχ²_null={delta_chi2_null:.1f}, NMSE_floor={nmse_floor:.4f}")

        # 2. Run blind search through v1
        # Run blind search with all primitives
        ranked_blind, space_size, _ = _run_search(sc, exclude_primitives=[], seed=42)
        print(f"Blind Search Space Size: {space_size}")
        print(f"Top 5 candidates from v1 Stage 2 optimizer:")
        for idx, (expr, nmse, bic, theta) in enumerate(ranked_blind[:5]):
            print(f"  [{idx+1}] NMSE={nmse:.3e} | BIC={bic:.2f} | {expr}")

        # 3. Run formal protocol
        # Note: positive control tries to isolate _guess_true_primitive(correction_expr).
        # In our case, correction_expr has expm1 which is not in PRIMITIVE_REGISTRY,
        # so let's observe what v1 protocol does.
        proto_res = run_scenario_protocol(sc, seed=42, use_taxonomy_prior=False, top_k_val=5)
        print(f"Protocol Verdict: {'IDENTIFIABLE' if proto_res.all_passed else 'WITHHELD'}")
        for chk, details in proto_res.checks.items():
            if chk == "primary_search":
                print(f"  check {chk}: top={details.get('top_candidate')} nmse={details.get('nmse'):.3e} bic={details.get('bic'):.2f}")
            else:
                p = details.get("pass")
                print(f"  check {chk}: pass={p} {details}")

        results[name] = {
            "max_corr": max_corr,
            "delta_chi2_null": delta_chi2_null,
            "nmse_floor": nmse_floor,
            "top_candidates": ranked_blind[:5],
            "proto_res": proto_res,
        }

    return results

if __name__ == "__main__":
    run_all_smoke_tests()
