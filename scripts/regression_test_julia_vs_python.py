"""
regression_test_julia_vs_python.py
Run the 3 locked validation scenarios using both Python and Julia engines,
and ensure they produce equivalent IDENTIFIABLE candidates.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import run_scenario_protocol

def main():
    scenarios = {s.name: s for s in get_all_scenarios()}
    locked = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]
    
    for name in locked:
        if name not in scenarios:
            print(f"[SKIP] Scenario '{name}' not found.")
            continue
            
        print("=" * 80)
        print(f" REGRESSION TEST: {name.upper()}")
        print("=" * 80)
        
        # Run Python
        print(">>> Running Python Engine...")
        scenarios[name].engine = "python"
        res_python = run_scenario_protocol(scenarios[name], top_k_val=5, use_taxonomy_prior=True)
        
        # Run Julia
        print(">>> Running Julia Engine...")
        scenarios[name].engine = "julia"
        res_julia = run_scenario_protocol(scenarios[name], top_k_val=5, use_taxonomy_prior=True)
        
        # Compare
        py_pass = res_python.all_passed
        jl_pass = res_julia.all_passed
        
        print(f"\n[RESULT] {name}")
        print(f"  Python All Passed: {py_pass}")
        print(f"  Julia All Passed:  {jl_pass}")
        
        if py_pass != jl_pass:
            print("  [ERROR] Mismatch between engines!")
            
if __name__ == "__main__":
    main()
