import sys
import os
sys.path.insert(0, "e:\\ADCD\\PhysicsPaper\\src")
import numpy as np
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import _run_search, ScenarioThresholdConfig, _guess_true_primitive
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY

# Setup
scenarios = get_all_scenarios()
td = next(s for s in scenarios if "Time Dilation" in s.name)
td.engine = 'julia'

true_prim = _guess_true_primitive(td.correction_expr)
excluded = [p for p in PRIMITIVE_REGISTRY.keys() if p != true_prim]

print(f"Scenario: {td.name}")
print(f"True primitive: {true_prim}")
print(f"Engine: {td.engine}")

res = _run_search(td, exclude_primitives=excluded, seed=42, n_candidates=10, threshold_cfg=ScenarioThresholdConfig.for_scenario(td))

# FIX UNPACKING ORDER
ranked, space_size, proposer = res

print(f"\nPositive control results ({td.engine}):")
print(f"Generated: {space_size}")
if ranked:
    top = ranked[0]
    print(f"Top Expr: {top[0]}")
    print(f"Top NMSE: {top[1]:.6f}")
    print(f"Top DeltaBIC / BIC: {top[2]:.6f}")
    print(f"Top theta: {top[3]}")
else:
    print("NO RESULTS RETURNED")
