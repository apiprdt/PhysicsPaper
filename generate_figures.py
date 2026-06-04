"""
Generate publication-quality figures untuk ADCD paper.

Figures:
  1. paper/figures/fig1_noise_robustness.pdf   — ADCD vs PySR vs MLP curve
  2. paper/figures/fig2_nmse_heatmap.pdf       — NMSE heatmap semua skenario
  3. paper/figures/fig3_tier_bars.pdf          — bar chart per tier
  4. paper/figures/fig4_ablation.pdf           — ablation study bar chart

Cara jalankan:
    py -3.11 generate_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication styling
rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',        # Computer Modern math font for LaTeX feel
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs("paper/figures", exist_ok=True)

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
NOISE_LABELS = ["0%", "1%", "5%", "10%"]
SCENARIOS_ORDER = [
    "Relativistic KE", "Yukawa Gravity", "Anharmonic Spring",
    "Screened Coulomb", "Net Radiation", "Nonlinear Drag",
    "Mystery-A", "Mystery-B", "Mystery-C"
]


def load_all():
    with open("scratch_correction_results.json") as f:
        adcd = json.load(f)
    pysr = None
    if os.path.exists("pysr_baseline_results.json"):
        with open("pysr_baseline_results.json") as f:
            pysr = json.load(f)
    mlp = None
    if os.path.exists("mlp_baseline_results.json"):
        with open("mlp_baseline_results.json") as f:
            mlp = json.load(f)
    ablation = None
    if os.path.exists("ablation_results.json"):
        with open("ablation_results.json") as f:
            ablation = json.load(f)
    return adcd, pysr, mlp, ablation


def fig1_noise_robustness(adcd, pysr, mlp):
    """Figure 1: Noise robustness curve — Class Match Rate vs Noise Level."""
    fig, ax = plt.subplots(figsize=(6, 4))

    def get_rates(data):
        rates = []
        for noise in NOISE_LEVELS:
            subset = [r for r in data if abs(r["noise"] - noise) < 1e-9]
            rate = sum(1 for r in subset if r["class_match"]) / len(subset) * 100
            rates.append(rate)
        return rates

    # Colors: Navy blue for ADCD, Crimson for PySR
    adcd_color = '#1E3A8A'
    pysr_color = '#B91C1C'

    adcd_rates = get_rates(adcd)
    ax.plot(NOISE_LABELS, adcd_rates, 'o-', color=adcd_color, lw=2.5,
            ms=8, mfc='white', mew=2.5, label='ADCD (ours)', zorder=5)

    if pysr:
        pysr_rates = get_rates(pysr)
        ax.plot(NOISE_LABELS, pysr_rates, 's--', color=pysr_color, lw=1.8,
                ms=7, mfc='white', mew=1.8, label='PySR (unconstrained)', zorder=4)

    for i, rate in enumerate(adcd_rates):
        ax.annotate(f'{rate:.1f}%', (NOISE_LABELS[i], rate),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9.5, fontweight='bold', color=adcd_color)

    ax.set_xlabel("Observational Noise Level", fontsize=11, labelpad=8)
    ax.set_ylabel("Structural Class Match Rate (%)", fontsize=11, labelpad=8)
    ax.set_title("Noise Robustness: ADCD vs PySR", fontsize=12, pad=12, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9.5, loc='lower left', framealpha=0.9, frameon=True)
    ax.grid(True, alpha=0.15, ls='--', color='gray')
    ax.axhline(100, color='green', alpha=0.1, lw=1)

    plt.tight_layout()
    plt.savefig("paper/figures/fig1_noise_robustness.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/fig1_noise_robustness.png", bbox_inches='tight')
    print("[OK] fig1_noise_robustness.pdf")
    plt.close()


def fig2_nmse_heatmap(adcd):
    """Figure 2: NMSE heatmap — log10(NMSE) per scenario × noise."""
    nmse_matrix = np.zeros((len(SCENARIOS_ORDER), len(NOISE_LEVELS)))
    match_matrix = np.zeros((len(SCENARIOS_ORDER), len(NOISE_LEVELS)), dtype=bool)

    for i, name in enumerate(SCENARIOS_ORDER):
        for j, noise in enumerate(NOISE_LEVELS):
            r = [x for x in adcd if x["scenario"] == name and abs(x["noise"] - noise) < 1e-9]
            if r:
                nmse_matrix[i, j] = np.log10(max(r[0]["nmse_full"], 1e-35))
                match_matrix[i, j] = r[0]["class_match"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # Elegant, professional, color-blind friendly sequential colormap
    im = ax.imshow(nmse_matrix, aspect='auto', cmap='YlGnBu_r', vmin=-35, vmax=0)

    ax.set_xticks(range(len(NOISE_LABELS)))
    ax.set_xticklabels(NOISE_LABELS, fontsize=10)
    ax.set_yticks(range(len(SCENARIOS_ORDER)))
    ax.set_yticklabels(SCENARIOS_ORDER, fontsize=9.5)
    ax.set_xlabel("Noise Level", fontsize=11, labelpad=8)
    ax.set_title(r"$\log_{10}(\text{Full NMSE})$ across all Anomaly Scenarios", fontsize=12, pad=12, fontweight='bold')

    # Annotate cells with checkmarks and crosses
    for i in range(len(SCENARIOS_ORDER)):
        for j in range(len(NOISE_LEVELS)):
            sym = r"$\checkmark$" if match_matrix[i, j] else r"$\times$"
            # White text for dark backgrounds (lower NMSE values)
            text_color = "white" if nmse_matrix[i, j] < -16 else "black"
            ax.text(j, i, sym, ha="center", va="center", fontsize=14,
                    color=text_color, fontweight='bold' if not match_matrix[i, j] else 'normal')

    # Grid separation lines for clarity
    for i in range(len(SCENARIOS_ORDER) - 1):
        ax.axhline(i + 0.5, color='white', lw=1.0, alpha=0.3)
    for j in range(len(NOISE_LEVELS) - 1):
        ax.axvline(j + 0.5, color='white', lw=1.0, alpha=0.3)

    # Thick lines separating tiers
    ax.axhline(2.5, color='white', lw=2.0)
    ax.axhline(5.5, color='white', lw=2.0)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(r"$\log_{10}(\text{NMSE})$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("paper/figures/fig2_nmse_heatmap.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/fig2_nmse_heatmap.png", bbox_inches='tight')
    print("[OK] fig2_nmse_heatmap.pdf")
    plt.close()


def fig3_tier_bars(adcd):
    """Figure 3: Match rate grouped by tier and noise level."""
    tiers = ["textbook", "cross_domain", "synthetic"]
    tier_labels = ["Tier 1\n(Textbook)", "Tier 2\n(Cross-Domain)", "Tier 3\n(Synthetic)"]
    # Cohesive, beautiful palette (gradient from navy to violet/rose)
    colors = ['#1E3A8A', '#3B82F6', '#8B5CF6', '#EC4899']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tiers))
    bw = 0.18

    for j, (noise, color, label) in enumerate(zip(NOISE_LEVELS, colors, NOISE_LABELS)):
        rates = []
        for tier in tiers:
            subset = [r for r in adcd if r["tier"] == tier and abs(r["noise"] - noise) < 1e-9]
            rate = sum(1 for r in subset if r["class_match"]) / len(subset) * 100 if subset else 0
            rates.append(rate)
        bars = ax.bar(x + (j - 1.5) * bw, rates, bw, label=f'Noise {label}',
                      color=color, alpha=0.85, edgecolor='white', lw=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xlabel("Anomaly Tier", fontsize=11, labelpad=8)
    ax.set_ylabel("Structural Class Match Rate (%)", fontsize=11, labelpad=8)
    ax.set_title("ADCD Performance by Tier and Noise Level", fontsize=12, pad=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9.5, loc='lower left', ncol=2, frameon=True)
    ax.grid(True, alpha=0.15, axis='y', ls='--', color='gray')

    plt.tight_layout()
    plt.savefig("paper/figures/fig3_tier_bars.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/fig3_tier_bars.png", bbox_inches='tight')
    print("[OK] fig3_tier_bars.pdf")
    plt.close()


def fig4_ablation(ablation):
    """Figure 4: Ablation study bar chart."""
    if ablation is None:
        print("  ablation_results.json tidak ada — skip fig4")
        return

    conditions = list({r["condition"] for r in ablation})
    order = ["Full_ADCD", "No_ARC", "No_AST", "No_Dim", "No_DataGate", "No_Gates"]
    conditions = [c for c in order if c in conditions]

    # Human-readable labels mapping
    labels_map = {
        "Full_ADCD": "Full ADCD",
        "No_ARC": "w/o ARC Gate",
        "No_AST": "w/o AST Gate",
        "No_Dim": "w/o Dim Gate",
        "No_DataGate": "w/o Data Gate",
        "No_Gates": "No Gates"
    }
    x_labels = [labels_map.get(c, c) for c in conditions]

    rates = []
    for cond in conditions:
        subset = [r for r in ablation if r["condition"] == cond]
        rate = sum(1 for r in subset if r["class_match"]) / len(subset) * 100
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Elegant navy for Full ADCD, professional gray/slate for ablation cases
    colors = ['#1E3A8A' if c == 'Full_ADCD' else '#64748B' for c in conditions]
    bars = ax.bar(x_labels, rates, color=colors, edgecolor='white', lw=0.5, alpha=0.9)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_ylabel("Class Match Rate @ 5% Noise (%)", fontsize=11, labelpad=8)
    ax.set_title("Ablation Study: Contribution of Each Physics Gate", fontsize=12, pad=12, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.axhline(rates[0], color='#1E3A8A', ls='--', alpha=0.3, lw=1.5, label='Full ADCD Baseline')
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.15, axis='y', ls='--', color='gray')
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("paper/figures/fig4_ablation.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/fig4_ablation.png", bbox_inches='tight')
    print("[OK] fig4_ablation.pdf")
    plt.close()


def main():
    print("=" * 60)
    print("   GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    adcd, pysr, mlp, ablation = load_all()
    print(f"Loaded ADCD ({len(adcd)}), "
          f"PySR ({'OK' if pysr else 'N/A'}), "
          f"MLP ({'OK' if mlp else 'N/A'}), "
          f"Ablation ({'OK' if ablation else 'N/A'})")

    fig1_noise_robustness(adcd, pysr, mlp)
    fig2_nmse_heatmap(adcd)
    fig3_tier_bars(adcd)
    fig4_ablation(ablation)
    print("\n[OK] All figures in paper/figures/")


if __name__ == "__main__":
    main()
