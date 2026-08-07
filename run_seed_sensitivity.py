import json
import sys
import numpy as np
sys.path.insert(0, 'src')

from adcd.run_adcd_v3_validation_blind import _run_search
from adcd.anomaly_scenarios import get_all_scenarios

def run_seed_sensitivity():
    seeds = [0, 1, 7, 13, 42]
    results = {}
    
    for s_name in ['Time Dilation', 'Screened Coulomb', 'Entropy Expansion']:
        scenario = next((s for s in get_all_scenarios() if s.name == s_name), None)
        if scenario is None:
            print(f"[SKIP] Scenario '{s_name}' not found")
            continue
        print(f"\nRunning {s_name}...")
        nmse_list = []
        for seed in seeds:
            # NOTE: exclude_primitives=None means fully blind search (no oracle)
            ranked_candidates, _, _ = _run_search(scenario, exclude_primitives=None, seed=seed)
            nmse = ranked_candidates[0][1] if ranked_candidates else float('nan')
            nmse_list.append(nmse)
            expr = ranked_candidates[0][0] if ranked_candidates else 'NONE'
            print(f"  Seed {seed}: NMSE={nmse:.6e}  top_expr={expr}")
        
        results[s_name] = {
            'seeds': seeds,
            'nmse_per_seed': [float(x) for x in nmse_list],
            'mean_nmse': float(np.nanmean(nmse_list)),
            'std_nmse': float(np.nanstd(nmse_list)),
            'cv_pct': float(np.nanstd(nmse_list) / np.nanmean(nmse_list) * 100)
        }
        print(f"  -> Mean={results[s_name]['mean_nmse']:.4e}  Std={results[s_name]['std_nmse']:.4e}  CV={results[s_name]['cv_pct']:.2f}%")

    with open('seed_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to seed_sensitivity.json")
    
    print("\n=== SEED SENSITIVITY SUMMARY ===")
    for name, r in results.items():
        print(f"{name}: mean={r['mean_nmse']:.4e} +/- {r['std_nmse']:.4e} (CV={r['cv_pct']:.2f}%)")

if __name__ == '__main__':
    run_seed_sensitivity()
