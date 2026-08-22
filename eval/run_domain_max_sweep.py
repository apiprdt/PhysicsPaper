from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import _run_search, ScenarioThresholdConfig
import sys

DOMAIN_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
SEEDS = [42, 43]

scenario_name = "Time Dilation"

def run_benchmark(engine):
    scenarios = {s.name: s for s in get_all_scenarios()}
    scenario = scenarios[scenario_name]
    t_cfg = ScenarioThresholdConfig.for_scenario(scenario)
    
    results = []
    
    print("================================================================")
    print(f" ADCD V3 DOMAIN MAX SWEEP : {scenario_name} (Engine: {engine})")
    print("================================================================")
    
    for dmax in DOMAIN_SWEEP:
        print(f"\n--- Testing Domain Max: {dmax} c ---")
        pass_count = 0
        match_count = 0
        withheld_count = 0
        
        for seed in SEEDS:
            import adcd.run_adcd_v3_validation_blind as vblind
            from adcd.julia_bridge import ADCDJuliaEngine
            if engine == "julia":
                vblind.engine = ADCDJuliaEngine()
                
            ranked, space_size, error_msg = _run_search(
                scenario, exclude_primitives=[], seed=seed, threshold_cfg=t_cfg, noise_level=0.01, domain_max=dmax
            )
            
            identifiable = [c for c in ranked if c[3].get("verdict") == "IDENTIFIABLE"]
            
            if len(identifiable) > 0:
                best = identifiable[0]
                is_match = ("D_lor" in best[3].get("primitives", []))
                pass_count += 1
                if is_match:
                    match_count += 1
                print(f"  Seed {seed}: IDENTIFIABLE -> {best[0]} (Match: {is_match})")
            else:
                withheld_count += 1
                print(f"  Seed {seed}: WITHHELD (No identifiable candidates)")
                
        results.append({
            "domain_max": dmax,
            "identifiable_rate": pass_count / len(SEEDS),
            "match_rate": match_count / len(SEEDS),
            "withheld_rate": withheld_count / len(SEEDS)
        })
        
    print("\n================================================================")
    print(" DOMAIN MAX SENSITIVITY SUMMARY")
    print("================================================================")
    print("v/c Max\t| Identifiable\t| True Match\t| Withheld")
    print("----------------------------------------------------------------")
    for r in results:
        print(f"{r['domain_max']:.2f} c\t| {r['identifiable_rate']*100:.0f}%\t\t| {r['match_rate']*100:.0f}%\t\t| {r['withheld_rate']*100:.0f}%")
        
if __name__ == "__main__":
    engine = sys.argv[1] if len(sys.argv) > 1 else "python"
    run_benchmark(engine)
