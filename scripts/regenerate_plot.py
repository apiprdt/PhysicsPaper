"""
Standalone script: regenerate publication-quality SPARC RAR plot
from the saved sparc_discovery.json result (no ADCD re-run needed).
"""
import json
import sys
from pathlib import Path

# Ensure project src is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from adcd.experiments.sparc_data import load_sparc_stack
from adcd.experiments.sparc_stacking import plot_sparc_results

# ── Load results from last run ─────────────────────────────────────────────
result_path = Path("results/sparc_discovery.json")
if not result_path.exists():
    sys.exit("ERROR: results/sparc_discovery.json not found. Run sparc_stacking first.")

with open(result_path, encoding="utf-8") as f:
    data = json.load(f)

discovered_expr = data["discovered_expr"]
n_galaxies      = data["n_galaxies"]
boot_ci         = data.get("bootstrap_ci", {})

# Use bootstrap mean values as theta (most representative estimate)
# theta_r1_0 and theta_r1_1 sorted by suffix number
param_keys = sorted(boot_ci.keys(), key=lambda k: int(k.split("_")[-1]))
if param_keys:
    theta_vals = [boot_ci[k]["mean"] for k in param_keys]
    theta_syms = param_keys
else:
    # Fallback: use default values if bootstrap not available
    theta_vals = [2.241, 0.268]
    theta_syms = ["theta_r1_0", "theta_r1_1"]

print(f"Expression : {discovered_expr}")
print(f"Theta vals : {theta_vals}")
print(f"N galaxies : {n_galaxies}")

# ── Load SPARC stack ───────────────────────────────────────────────────────
print("\nLoading SPARC data stack...")
stack = load_sparc_stack(cache_path="data/sparc/MassModels_Lelli2016c.mrt")
print(f"Loaded {stack.n_galaxies} galaxies, {stack.n_points} data points.")

# ── Regenerate plot ────────────────────────────────────────────────────────
out_path = "results/sparc_discovery_plot.png"
print(f"\nGenerating plot -> {out_path}")
plot_sparc_results(
    x=stack.x,
    nu_obs=stack.nu_obs,
    discovered_expr=discovered_expr,
    theta_vals=theta_vals,
    theta_symbols=theta_syms,
    n_galaxies=stack.n_galaxies,
    output_path=out_path,
)
print("Done.")
