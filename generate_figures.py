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
    for p in ("pysr_baseline_fair.json", "pysr_baseline_results.json"):
        if os.path.exists(p):
            with open(p) as f:
                pysr = json.load(f)
            break
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
    """Figure 1: Noise robustness curve — Class Match Rate vs Noise Level.
    Professional white-background version with all data-point labels annotated
    for both ADCD and PySR, plus a gap arrow at the 5% noise regime.
    """
    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    def get_rates(data):
        rates = []
        for noise in NOISE_LEVELS:
            subset = [r for r in data if abs(r["noise"] - noise) < 1e-9]
            if not subset:
                rates.append(None)
            else:
                rate = sum(1 for r in subset if r["class_match"]) / len(subset) * 100
                rates.append(rate)
        return rates

    adcd_color = '#1a3a6b'   # Deep navy – confidence, ADCD
    pysr_color = '#b91c1c'   # Deep crimson – PySR

    adcd_rates = get_rates(adcd)

    if pysr:
        pysr_rates = get_rates(pysr)
    else:
        pysr_rates = [None] * len(NOISE_LEVELS)

    xs = list(range(len(NOISE_LABELS)))

    # ── Main lines ──────────────────────────────────────────────────────
    ax.plot(xs, adcd_rates, 'o-', color=adcd_color, lw=2.8,
            ms=9, mfc='white', mew=2.5, label='ADCD (ours)', zorder=5,
            solid_capstyle='round')
    ax.plot(xs, pysr_rates, 's--', color=pysr_color, lw=2.0,
            ms=8, mfc='white', mew=2.0, label='PySR (fair, same residual)', zorder=4)

    # ── Label every ADCD point ──────────────────────────────────────────
    for i, rate in enumerate(adcd_rates):
        if rate is None:
            continue
        ax.annotate(
            f'{rate:.1f}%',
            xy=(xs[i], rate),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=adcd_color,
        )

    # ── Label every PySR point ──────────────────────────────────────────
    for i, rate in enumerate(pysr_rates):
        if rate is None:
            continue
        if i == 2:  # 11.1% point at 5% noise: place below the marker to avoid V-shape lines clashing
            ax.annotate(
                f'{rate:.1f}%',
                xy=(xs[i], rate),
                xytext=(0, -14),
                textcoords='offset points',
                ha='center', va='top',
                fontsize=10, fontweight='bold', color=pysr_color,
            )
        else:
            ax.annotate(
                f'{rate:.1f}%',
                xy=(xs[i], rate),
                xytext=(0, 7),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=pysr_color,
            )

    # ── Axes formatting ─────────────────────────────────────────────────
    ax.set_xticks(xs)
    ax.set_xticklabels(NOISE_LABELS, fontsize=11)
    ax.set_xlabel('Observational Noise Level', fontsize=12, labelpad=8)
    ax.set_ylabel('Structural Class Match Rate (%)', fontsize=12, labelpad=8)
    ax.set_title('Noise Robustness: ADCD vs PySR', fontsize=13, pad=12,
                 fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-0.35, len(xs) - 0.35)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.grid(True, alpha=0.18, ls='--', color='gray', lw=0.7)
    ax.legend(fontsize=10.5, loc='lower left', framealpha=0.95,
              frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig("paper/figures/fig1_noise_robustness.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/fig1_noise_robustness.png", bbox_inches='tight',
                dpi=300)
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
        "No_DataGate": "w/o Coarse Eval",
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


def fig5_gate_funnel():
    """Figure 5: Gate survival funnel from gate_telemetry.json."""
    if not os.path.exists("gate_telemetry.json"):
        print("  gate_telemetry.json tidak ada — skip fig5")
        return
    with open("gate_telemetry.json") as f:
        g = json.load(f)

    labels = ["Input", "After\nParse", "After\nAST", "After\nDim", "After\nTrans", "After\nARC", "Output"]
    parse_ok = g["input_count"] - g.get("parse_fail", 0)
    ast_ok = parse_ok - g.get("ast_reject", 0)
    dim_ok = ast_ok - g.get("dim_reject", 0)
    trans_ok = dim_ok - g.get("transcendental_reject", 0)
    arc_ok = trans_ok - g.get("arc_reject", 0)
    out_ok = g.get("output_count", 0)
    counts = [g["input_count"], parse_ok, ast_ok, dim_ok, trans_ok, arc_ok, out_ok]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(counts)))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{c:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Candidate count (aggregate)")
    ax.set_title("Stage 1 Gate Survival Funnel", fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")
    plt.tight_layout()
    plt.savefig("paper/figures/fig5_gate_funnel.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig5_gate_funnel.png", bbox_inches="tight")
    print("[OK] fig5_gate_funnel.pdf")
    plt.close()


def fig6_correction_scaling():
    """Figure 6: Class match vs correction magnitude scale."""
    if not os.path.exists("correction_scaling_results.json"):
        print("  correction_scaling_results.json tidak ada — skip fig6")
        return
    with open("correction_scaling_results.json") as f:
        data = json.load(f)
    adcd = [r for r in data if r["method"] == "ADCD"]
    if not adcd:
        return
    scales = [r["scale"] for r in adcd]
    matches = [100 if r["class_match"] else 0 for r in adcd]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(scales, matches, "o-", color="#1E3A8A", lw=2, ms=8)
    ax.set_xlabel("Correction amplitude scale (Relativistic KE)")
    ax.set_ylabel("ADCD class match @ 5% noise")
    ax.set_ylim(-5, 105)
    ax.set_title("Correction-First Regime Diagram", fontweight="bold")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig("paper/figures/fig6_correction_scaling.pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig6_correction_scaling.png", bbox_inches="tight")
    print("[OK] fig6_correction_scaling.pdf")
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
    fig5_gate_funnel()
    fig6_correction_scaling()
    print("\n[OK] All figures in paper/figures/")


if __name__ == "__main__":
    main()
