import sys
import os
sys.path.insert(0, "e:\\ADCD\\PhysicsPaper\\src")
import numpy as np
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import _run_search, ScenarioThresholdConfig, _guess_true_primitive
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY
from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig, JuliaEngineData

td = next(s for s in get_all_scenarios() if "Time Dilation" in s.name)
true_prim = _guess_true_primitive(td.correction_expr)
excluded = [p for p in PRIMITIVE_REGISTRY.keys() if p != true_prim]

# Generate data
X, y_obs, y_cls, _ = td.generate_data(n_points=100, noise_level=0.01, seed=42)

tcfg = ScenarioThresholdConfig.for_scenario(td)
config = JuliaEngineConfig(
    domain=td.domain,
    target_dim="dimensionless",
    input_vars=td.classical_variables,
    known_constants=td.classical_constants,
    bic_threshold=tcfg.bic_threshold,
    nmse_coarse=tcfg.nmse_coarse,
    nmse_fine=tcfg.nmse_fine,
    n_restarts=15,
    max_proposals=10,
    groups=tcfg.groups,
    excluded_primitives=list(excluded)
)
data = JuliaEngineData(y_classical=y_cls, y_obs=y_obs, vars=X)

engine = ADCDJuliaEngine()
print('Running Julia engine manually...')
res = engine.run(config, data)

print('\nGate Stats:', res.gate_stats)
for r in res.results:
    print('Cand:', r.expr, 'NMSE:', r.nmse, 'Theta:', r.theta)