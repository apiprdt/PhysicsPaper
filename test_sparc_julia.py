import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from adcd.real_scenarios import get_real_scenarios
from adcd.real_data_loader import load_sparc_rar
from adcd.run_adcd_v3_validation_blind import ScenarioThresholdConfig
from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineData, config_from_scenario

def main():
    print("================================================================")
    print(" ADCD V3 SPARC JULIA VALIDATION")
    print("================================================================")
    
    scenarios = get_real_scenarios()
    scenario = next(s for s in scenarios if "SPARC" in s.name)
    tcfg = ScenarioThresholdConfig.for_scenario(scenario)
    
    # Generate data
    X, y_obs, y_cl, _ = load_sparc_rar(42)
    
    print(f"Loaded {len(y_obs)} SPARC data points.")
    if "galaxy_id" in X:
        n_groups = len(np.unique(X["galaxy_id"]))
        print(f"Extracted {n_groups} galaxies for Hierarchical BIC.")
    
    # Configure the Julia config
    config = config_from_scenario(scenario, tcfg, X)
    config.max_proposals = 1000
    config.n_restarts = 50 # Evaluate full search space
    
    X_vars = {k: v for k, v in X.items() if k != "galaxy_id"}
    data = JuliaEngineData(
        y_classical=y_cl,
        y_obs=y_obs,
        vars=X_vars
    )
    
    engine = ADCDJuliaEngine()
    
    print("\nRunning ADCD Julia Engine on SPARC data...")
    import subprocess
    try:
        res = engine.run(config, data)
    except subprocess.CalledProcessError as e:
        print("ERROR RUNNING JULIA:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return
    
    print("\n================================================================")
    print(" RESULTS")
    print("================================================================")
    withheld = [r for r in res.results if not r.is_identifiable]
    if not res.identifiable and not withheld:
        print("No results returned.")
        return
        
    print("--- IDENTIFIABLE ---")
    if not res.identifiable:
        print("None.")
    for i, r in enumerate(res.identifiable[:5]):
        print(f"Rank {i+1}: {r.description} | NMSE={r.nmse:.6f} | delta_BIC={r.delta_bic:.2f} | Pattern={r.pattern}")
        print(f"  -> Params: {r.theta}")
        
    print("\n--- WITHHELD ---")
    if not withheld:
        print("None.")
    for i, r in enumerate(withheld[:5]):
        print(f"Rank {i+1}: {r.description} | NMSE={r.nmse:.6f} | delta_BIC={r.delta_bic:.2f} | Pattern={r.pattern} | Verdict={r.verdict}")
        print(f"  -> Params: {r.theta}")

if __name__ == "__main__":
    main()
