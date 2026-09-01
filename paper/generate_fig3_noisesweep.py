"""
Generate fig3_noisesweep.pdf from pysr_comparison_v2.json
3-panel: recovery rate (%) vs noise level, ADCD vs PySR, Wilson 95% CI shaded.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
JSON_PATH  = os.path.join(ROOT_DIR, "run_outputs", "pysr_comparison_v2.json")
OUT_PDF    = os.path.join(SCRIPT_DIR, "fig3_noisesweep.pdf")
OUT_PNG    = os.path.join(SCRIPT_DIR, "fig3_noisesweep.png")

SCENARIOS = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]
SCENARIO_SHORT = {
    "Time Dilation":    "Time Dilation (TD)",
    "Screened Coulomb": "Screened Coulomb (SC)",
    "Entropy Expansion":"Entropy Expansion (EE)",
}
ADCD_COLOR  = "#2166ac"
PYSR_COLOR  = "#d6604d"
NMSE_THRESH = 2.0

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

print("Loading:", JSON_PATH)
with open(JSON_PATH, "r") as f:
    data = json.load(f)

raw_runs = data["raw_runs"]
print("Total runs:", len(raw_runs))

noise_levels = sorted(set(r["noise"] for r in raw_runs))
print("Noise levels:", noise_levels)

def is_adcd_match(run):
    adcd = run.get("adcd", {})
    is_match    = adcd.get("is_match", False)
    nmse_extrap = adcd.get("nmse_extrap", 999.0)
    if nmse_extrap is None:
        nmse_extrap = 999.0
    return bool(is_match) and float(nmse_extrap) < NMSE_THRESH

def is_pysr_match(run):
    pysr = run.get("pysr", {})
    is_match    = pysr.get("is_match", False)
    nmse_extrap = pysr.get("nmse_extrap", 999.0)
    if nmse_extrap is None:
        nmse_extrap = 999.0
    return bool(is_match) and float(nmse_extrap) < NMSE_THRESH

results = {}
for sc in SCENARIOS:
    results[sc] = {
        "noise": [], "adcd_rate": [], "adcd_lo": [], "adcd_hi": [],
        "pysr_rate": [], "pysr_lo": [], "pysr_hi": []
    }

for noise in noise_levels:
    for sc in SCENARIOS:
        runs_sn = [r for r in raw_runs if r["noise"] == noise and r["scenario"] == sc]
        n = len(runs_sn)
        adcd_k = sum(1 for r in runs_sn if is_adcd_match(r))
        pysr_k = sum(1 for r in runs_sn if is_pysr_match(r))
        adcd_lo, adcd_hi = wilson_ci(adcd_k, n)
        pysr_lo, pysr_hi = wilson_ci(pysr_k, n)
        results[sc]["noise"].append(noise)
        results[sc]["adcd_rate"].append(adcd_k / n * 100 if n > 0 else 0)
        results[sc]["adcd_lo"].append(adcd_lo * 100)
        results[sc]["adcd_hi"].append(adcd_hi * 100)
        results[sc]["pysr_rate"].append(pysr_k / n * 100 if n > 0 else 0)
        results[sc]["pysr_lo"].append(pysr_lo * 100)
        results[sc]["pysr_hi"].append(pysr_hi * 100)
        print("  " + sc + " | noise=" + str(noise) + " | n=" + str(n) + " | ADCD " + str(adcd_k) + "/" + str(n) + " | PySR " + str(pysr_k) + "/" + str(n))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18, wspace=0.08)

for ax, sc in zip(axes, SCENARIOS):
    r = results[sc]
    x        = np.array(r["noise"])
    adcd_y   = np.array(r["adcd_rate"])
    pysr_y   = np.array(r["pysr_rate"])
    adcd_lo  = np.array(r["adcd_lo"])
    adcd_hi  = np.array(r["adcd_hi"])
    pysr_lo  = np.array(r["pysr_lo"])
    pysr_hi  = np.array(r["pysr_hi"])

    ax.fill_between(x, adcd_lo, adcd_hi, alpha=0.18, color=ADCD_COLOR, linewidth=0)
    ax.plot(x, adcd_y, "-o", color=ADCD_COLOR, linewidth=2.2, markersize=6, label="ADCD", zorder=3)
    ax.fill_between(x, pysr_lo, pysr_hi, alpha=0.18, color=PYSR_COLOR, linewidth=0)
    ax.plot(x, pysr_y, "--s", color=PYSR_COLOR, linewidth=2.0, markersize=6, label="PySR", zorder=3)

    ax.set_title(SCENARIO_SHORT[sc], fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Noise level (sigma)", fontsize=10)
    ax.set_xlim(0.005, 0.315)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v)) + "%"))
    ax.axhline(50, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=45, ha="right", fontsize=8)

axes[0].set_ylabel("Structural recovery rate (%)", fontsize=10)

adcd_patch = mpatches.Patch(color=ADCD_COLOR, label="ADCD (blind)")
pysr_patch  = mpatches.Patch(color=PYSR_COLOR, label="PySR (default)")
fig.legend(handles=[adcd_patch, pysr_patch], loc="upper center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.52, 0.995),
           frameon=True, edgecolor="#cccccc")

plt.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
plt.close()
print("Saved: " + OUT_PDF)
print("Saved: " + OUT_PNG)
