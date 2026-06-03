import json
from collections import defaultdict

def main():
    with open("scratch_correction_results.json", "r") as f:
        results = json.load(f)
        
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)
        
    for name, runs in by_scenario.items():
        print(f"### {name}")
        print("| Noise Level | Discovered Correction (\\Delta) | NMSE (Full) | Class Match | AST Edit Dist | Status |")
        print("|:---:|:---|:---:|:---:|:---:|:---|")
        for run in runs:
            noise_pct = f"{int(run['noise'] * 100)}%"
            expr = run["discovered_expr"]
            nmse = f"{run['nmse_full']:.2e}"
            match = "**YES**" if run["class_match"] else "NO"
            dist = run["ast_edit_distance"]
            status = "CONVERGED" if run["converged"] else "OK"
            print(f"| {noise_pct} | `{expr}` | {nmse} | {match} | {dist} | {status} |")
        print()

if __name__ == "__main__":
    main()
