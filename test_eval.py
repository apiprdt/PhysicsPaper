import json
import sympy as sp
from adcd.anomaly_scenarios import get_all_scenarios

with open('run_outputs/adcd_v3_taxonomy_validation_report.json', 'r') as f:
    report = json.load(f)

scenarios = {s.name: s for s in get_all_scenarios()}
name = "Screened Coulomb"
scenario = scenarios[name]
X_clean, _, _, residual_clean = scenario.generate_data(noise_level=0, domain_max=4.0)

for k, v in scenario.classical_constants.items():
    if k not in X_clean:
        import numpy as np
        X_clean[k] = np.full_like(residual_clean, v)

cand = report[name]["checks"]["primary_search"]["pareto_front"][0]
print(cand.keys())
