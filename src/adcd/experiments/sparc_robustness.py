"""
Robustness and sensitivity analysis for SPARC MOND discovery.
Tests if the discovered ADCD formula parameters remain stable under different quality cuts.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from adcd.experiments.sparc_data import parse_sparc_mrt, DEFAULT_CACHE
from adcd.experiments.mond_comparison import nu_simple_mond, nu_rar
from adcd.jax_optimizer import JAXOptimizer


def _nmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_obs - y_pred) ** 2) / max(np.var(y_obs), 1e-15))


def run_robustness_test(cache_path: str = DEFAULT_CACHE, seed: int = 42):
    print("=" * 70)
    print("SPARC MOND DISCOVERY ROBUSTNESS TEST")
    print("=" * 70)
    
    # Load raw dataframe
    df = parse_sparc_mrt(cache_path)
    
    # Define quality cuts scenarios
    scenarios = [
        {
            "name": "Baseline (Standard)",
            "min_v_bar": 5.0,
            "max_nu": 10.0,
            "min_points_per_galaxy": 5,
        },
        {
            "name": "Strict (High Quality)",
            "min_v_bar": 10.0,
            "max_nu": 8.0,
            "min_points_per_galaxy": 10,
        },
        {
            "name": "Ultra-Strict (Pristine)",
            "min_v_bar": 20.0,
            "max_nu": 6.0,
            "min_points_per_galaxy": 15,
        }
    ]
    
    # Discovered formula
    discovered_expr = "theta_r1_0 * (sqrt(1.0 + theta_r1_1 / x) - 1.0) + 1.0"
    optimizer = JAXOptimizer(n_restarts=10)
    
    results = []
    
    for sc in scenarios:
        print(f"\n--- Running Scenario: {sc['name']} ---")
        
        x_all, nu_all = [], []
        galaxies_used = 0
        
        # Group by galaxy
        for _, grp in df.groupby("galaxy"):
            r_kpc = grp["radius_kpc"].values
            v_obs = grp["v_obs"].values
            v_gas = grp["v_gas"].values
            v_disk = grp["v_disk"].values
            v_bulge = grp["v_bulge"].values
            
            v_bar = np.sqrt(v_gas ** 2 + v_disk ** 2 + v_bulge ** 2)
            
            # Physics conversion
            KPC_TO_M = 3.085677581e19
            KM_TO_M = 1000.0
            G_DAGGER = 1.2e-10
            
            r_m = r_kpc * KPC_TO_M
            v_bar_ms = v_bar * KM_TO_M
            v_obs_ms = v_obs * KM_TO_M
            
            valid = (
                (r_kpc > 0)
                & (v_bar > sc["min_v_bar"])
                & np.isfinite(v_obs)
                & np.isfinite(v_bar)
            )
            # Filter by points per galaxy
            if valid.sum() < sc["min_points_per_galaxy"]:
                continue
                
            g_bar = v_bar_ms[valid] ** 2 / r_m[valid]
            x = g_bar / G_DAGGER
            nu = (v_obs_ms[valid] / v_bar_ms[valid]) ** 2
            
            good = (nu > 0) & (nu < sc["max_nu"]) & np.isfinite(nu) & np.isfinite(x) & (x > 0)
            if good.sum() < 3:
                continue
                
            x_all.extend(x[good])
            nu_all.extend(nu[good])
            galaxies_used += 1
            
        x_stack = np.asarray(x_all)
        nu_stack = np.asarray(nu_all)
        n_points = len(x_stack)
        
        print(f"Galaxies used: {galaxies_used} | Total data points: {n_points}")
        
        # Run fit
        opt_res = optimizer.optimize(
            expr_str=discovered_expr,
            X={"x": x_stack},
            y_obs=nu_stack,
            data_vars=["x"],
            loss_mode="residual",
            seed=seed
        )
        
        if opt_res.error:
            print(f"Fit failed: {opt_res.error}")
            continue
            
        theta_0_val = opt_res.theta["theta_r1_0"]
        theta_1_val = opt_res.theta["theta_r1_1"]
        
        # Benchmark comparisons
        nmse_adcd = opt_res.nmse
        nmse_sm = _nmse(nu_stack, nu_simple_mond(x_stack))
        nmse_rar = _nmse(nu_stack, nu_rar(x_stack))
        
        print("ADCD Discovered Formula Parameter Fit:")
        print(f"  theta_0 (scale)    : {theta_0_val:.4f}")
        print(f"  theta_1 (acc scale): {theta_1_val:.4f}")
        print("NMSE Comparison:")
        print(f"  ADCD Discovered    : {nmse_adcd:.5f} (lower is better)")
        print(f"  RAR (McGaugh)      : {nmse_rar:.5f}")
        print(f"  Simple MOND        : {nmse_sm:.5f}")
        
        results.append({
            "scenario": sc["name"],
            "galaxies": galaxies_used,
            "points": n_points,
            "theta_0": theta_0_val,
            "theta_1": theta_1_val,
            "nmse_adcd": nmse_adcd,
            "nmse_rar": nmse_rar,
            "nmse_sm": nmse_sm
        })
        
    # Save results to json
    out_path = Path("results/sparc_robustness_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRobustness test results saved to: {out_path}")
    
    # Print LaTeX Table format summary
    print("\n" + "=" * 50)
    print("LaTeX Table Summary:")
    print("=" * 50)
    print("Scenario & N_{gal} & N_{pts} & \\theta_0 & \\theta_1 & NMSE_{ADCD} & NMSE_{RAR} \\\\")
    for r in results:
        print(f"{r['scenario']} & {r['galaxies']} & {r['points']} & {r['theta_0']:.3f} & {r['theta_1']:.3f} & {r['nmse_adcd']:.4f} & {r['nmse_rar']:.4f} \\\\")
    print("=" * 50)


if __name__ == "__main__":
    run_robustness_test()
