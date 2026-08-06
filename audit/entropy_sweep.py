import sys
import json
import logging
import numpy as np
import sympy as sp
from adcd.run_adcd_v3_validation import run_full_protocol
from adcd.anomaly_scenarios import AnomalyScenario

logging.basicConfig(level=logging.INFO)

u_max_values = [5, 10, 20, 50, 100]
results_map = {}

original_generate_data = AnomalyScenario.generate_data

for u_max in u_max_values:
    print(f"\n==============================================")
    print(f"RUNNING ENTROPY SWEEP: u_max = {u_max}")
    print(f"==============================================\n")
    
    def mocked_generate_data(self, n_points=500, noise_level=0.01, seed=42, v_max_over_c=None):
        if self.name == "Entropy Expansion":
            rng = np.random.default_rng(seed)
            V_i = rng.uniform(1.0, 10.0, size=n_points)
            dV = rng.uniform(0.1, float(u_max), size=n_points) * V_i
            S_i = np.full_like(V_i, 15.0)
            N = np.full_like(V_i, 1.0)
            k_B = 1.380649e-23
            X = {"V_i": V_i, "dV": dV, "S_i": S_i, "N": N, "k_B": np.full_like(V_i, k_B)}
            
            input_vars = list(self.variables_with_units.keys())
            from sympy import lambdify
            syms = tuple(sp.Symbol(name) for name in input_vars)
            
            # Classical
            classical_expr_str = self.classical_expr
            for k, v in self.classical_constants.items():
                classical_expr_str = classical_expr_str.replace(k, str(v))
            f_class = lambdify(syms, sp.sympify(classical_expr_str), modules=["numpy"])
            y_classical = f_class(*(X[var] for var in input_vars))
            if isinstance(y_classical, (int, float)):
                y_classical = np.full_like(V_i, y_classical)
            
            # Full
            full_expr_str = f"({self.classical_expr}) * (1 + {self.correction_expr})"
            for k, v in self.correction_constants.items():
                full_expr_str = full_expr_str.replace(k, str(v))
            for k, v in self.classical_constants.items():
                full_expr_str = full_expr_str.replace(k, str(v))
            f_full = lambdify(syms, sp.sympify(full_expr_str), modules=["numpy"])
            y_true = f_full(*(X[var] for var in input_vars))
            
            noise = rng.normal(0, noise_level * np.abs(y_true))
            y_obs = y_true + noise
            residual = y_obs - y_classical
            
            return X, y_obs, y_classical, residual
        else:
            return original_generate_data(self, n_points, noise_level, seed, v_max_over_c)
            
    AnomalyScenario.generate_data = mocked_generate_data
    
    try:
        res = run_full_protocol(
            scenario_name="Entropy Expansion",
            ratio_symbol="dV / V_i",
            ground_truth_primitive="D_log",
            seed=42
        )
        
        ablation_ratio = None
        competing_primitive = None
        for r in res:
            if r.check_name == "ablation_control":
                import re
                m = re.search(r"Ablation Ratio = ([\d\.]+)", r.detail)
                if m:
                    ablation_ratio = float(m.group(1))
                m2 = re.search(r"Best alternative NMSE = .*? \((D_[a-z_]+)", r.detail)
                if m2:
                    competing_primitive = m2.group(1)
                elif "Fallback" in r.detail:
                    m3 = re.search(r"Fallback: (D_[a-z_]+)", r.detail)
                    if m3:
                        competing_primitive = m3.group(1)
        
        results_map[u_max] = {
            "ablation_ratio": ablation_ratio,
            "competing_primitive": competing_primitive
        }
        print(f"--> RESULT u_max={u_max}: Ratio={ablation_ratio}, Competitor={competing_primitive}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED for u_max={u_max}: {e}")

print("\n\nFINAL SWEEP RESULTS:")
print(json.dumps(results_map, indent=2))
