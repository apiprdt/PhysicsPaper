import json
import sympy as sp
from adcd.anomaly_scenarios import get_all_scenarios
import numpy as np

with open('run_outputs/adcd_v3_taxonomy_validation_report.json', 'r') as f:
    report = json.load(f)

scenarios = {s.name: s for s in get_all_scenarios()}
name = "Screened Coulomb"
scenario = scenarios[name]

X_clean, _, _, delta_true = scenario.generate_data(noise_level=0, domain_max=4.0)
cand = report[name]["checks"]["primary_search"]["pareto_front"][0]

if 'theta_fit' in cand:
    expr = sp.sympify(cand['expr_str']).subs(cand['theta_fit'])
    free_syms = list(expr.free_symbols)
    subs_dict = {}
    for sym in free_syms:
        s_name = str(sym)
        if s_name in X_clean:
            subs_dict[s_name] = X_clean[s_name]
        elif s_name in scenario.classical_constants:
            subs_dict[s_name] = np.full_like(delta_true, scenario.classical_constants[s_name])
    
    if subs_dict:
        args = list(subs_dict.keys())
        func = sp.lambdify([sp.Symbol(arg) for arg in args], expr, modules=['numpy'])
        delta_pred = func(*[subs_dict[arg] for arg in args])
    else:
        delta_pred = np.zeros_like(delta_true) + float(expr)
    print('delta_pred computed via lambdify! Mean:', np.mean(delta_pred))
