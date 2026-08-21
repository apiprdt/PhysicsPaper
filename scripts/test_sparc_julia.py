"""
test_sparc_julia.py
Run the SPARC RAR scenario using the Julia engine.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from adcd.real_scenarios import get_real_scenarios
from adcd.run_adcd_v3_validation_blind import run_scenario_protocol

def main():
    scenarios = {s.name: s for s in get_real_scenarios()}
    sparc = scenarios.get("Real: SPARC RAR")
    
    if not sparc:
        print("[ERROR] SPARC RAR scenario not found!")
        return
        
    print("=" * 80)
    print(" SPARC RAR BENCHMARK (JULIA ENGINE)")
    print("=" * 80)
    
    sparc.engine = "julia"
    res = run_scenario_protocol(sparc, top_k_val=10, use_taxonomy_prior=True)
    
    print("\nRESULTS:")
    for step, info in res.checks.items():
        if step == "primary_search":
            print(f"Primary Search:")
            for cand in info.get("pareto_front", []):
                print(f"  - {cand['expr_str']} (BIC: {cand['bic']:.2f})")
        else:
            status = "PASS" if info.get("pass") else "FAIL"
            print(f"[{status:^6}] {step.upper():<20} | {info}")

if __name__ == "__main__":
    main()
