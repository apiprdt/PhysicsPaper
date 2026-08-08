import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_adcd_flowchart():
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Palette
    bg_color = "#ffffff"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    box_blue = "#e0f2fe"
    border_blue = "#0284c7"
    text_blue = "#0369a1"
    
    box_purple = "#f3e8ff"
    border_purple = "#9333ea"
    text_purple = "#6b21a8"
    
    box_amber = "#fef3c7"
    border_amber = "#d97706"
    text_amber = "#92400e"
    
    box_teal = "#ccfbf1"
    border_teal = "#0d9488"
    text_teal = "#115e59"
    
    box_green = "#dcfce7"
    border_green = "#16a34a"
    text_green = "#166534"
    
    box_red = "#ffe4e6"
    border_red = "#e11d48"
    text_red = "#9f1239"

    def draw_box(x, y, w, h, bg, border, title, desc=None, text_color="#1e293b", rx=0.2):
        box = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0.1,rounding_size={rx}",
            linewidth=1.5, edgecolor=border, facecolor=bg
        )
        ax.add_patch(box)
        
        if desc:
            ax.text(x, y + h*0.22, title, fontsize=11, fontweight='bold', ha='center', va='center', color=text_color)
            ax.text(x, y - h*0.18, desc, fontsize=8.5, ha='center', va='center', color="#475569", multialignment='center')
        else:
            ax.text(x, y, title, fontsize=11, fontweight='bold', ha='center', va='center', color=text_color)

    def draw_arrow(x1, y1, x2, y2, label=None, color="#64748b"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=1.8, color=color)
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx + 0.15, my, label, fontsize=8.5, fontweight='bold', color=color, va='center')

    # Title Banner
    ax.text(5, 13.4, "ADCD Framework Architecture", fontsize=15, fontweight='bold', ha='center', color="#0f172a")
    ax.text(5, 13.05, "Deterministic & Identifiability-Aware Physics Discovery Pipeline", fontsize=9.5, ha='center', color="#64748b")

    # 1. Inputs
    draw_box(5, 12.2, 7.8, 0.9, box_blue, border_blue, 
             "1. Observational Data & Classical Baseline", 
             "Inputs: X, y_obs, y_cl  |  Metadata: Units [L, M, T] & Limit Regime", text_blue)
    
    draw_arrow(5, 11.75, 5, 11.15)

    # 2. Candidate Generation
    draw_box(5, 10.6, 7.8, 1.1, box_purple, border_purple, 
             "2. Deterministic Candidate Enumeration (GrammarProposerV3)", 
             "• Buckingham-π Auto-derived Dimensionless Ratios\n• 5 Asymptotic Primitives  |  Generation Budget ≤ 25 Tokens\n• Candidate Pool Size |C| ≈ 140 – 260", text_purple)

    draw_arrow(5, 10.05, 5, 9.45, label="|C| Candidates")

    # 3. Physical Gate Cascade (Group box)
    cascade_bg = patches.FancyBboxPatch(
        (1.0, 4.75), 8.0, 4.4,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        linewidth=1.2, edgecolor="#cbd5e1", facecolor="#f8fafc", linestyle="--"
    )
    ax.add_patch(cascade_bg)
    ax.text(5, 8.9, "3. FIVE-GATE PHYSICAL CASCADE (Pre-fitting Filter)", fontsize=10, fontweight='bold', ha='center', color="#334155")

    gates = [
        ("Gate 1: AST Complexity", "SymPy Depth ≤ 12, Nodes ≤ 50"),
        ("Gate 2: Dimensional Check", "SI Unit Vectors Must Balance"),
        ("Gate 3: Transcendental Safety", "log(u>0), sqrt(u≥0) Domain Check"),
        ("Gate 4: Asymptotic Check (ARC)", "Hard Constraint: lim(u→0) Δ = 0"),
        ("Gate 5: Coarse Pre-Filter", "Discard NaN / inf Candidate Fits")
    ]
    
    gate_y = 8.35
    for i, (g_title, g_desc) in enumerate(gates):
        draw_box(5, gate_y - i*0.75, 7.2, 0.6, "#ffffff", border_amber, g_title + " — " + g_desc, text_color=text_amber, rx=0.15)
        if i < 4:
            draw_arrow(5, gate_y - i*0.75 - 0.3, 5, gate_y - (i+1)*0.75 + 0.3, color="#d97706")

    draw_arrow(5, 4.75, 5, 4.15, label="~10–50 Survivors", color="#0d9488")

    # 4. JAX L-BFGS-B Optimization
    draw_box(5, 3.6, 7.8, 1.1, box_teal, border_teal, 
             "4. Float64 JAX L-BFGS-B Numerical Optimizer", 
             "• Parameters in Log-Space: θ_i = s_i · exp(u_i)\n• Multi-Start Optimization (n_restarts = 15 per candidate)\n• High-Precision Loss Evaluation on Residuals", text_teal)

    draw_arrow(5, 3.05, 5, 2.45)

    # 5. Model Selection & BIC Ranking
    draw_box(5, 1.9, 7.8, 1.0, box_blue, border_blue, 
             "5. Bayesian Model Selection & Pareto Front", 
             "• BIC = k ln N - 2 ln L̂  |  Parsimony-vs-Fitness Ranking\n• True Structure vs. Ablated Best Search Comparison", text_blue)

    draw_arrow(5, 1.4, 2.8, 0.7, color="#16a34a")
    draw_arrow(5, 1.4, 7.2, 0.7, color="#e11d48")

    # 6. Final Outputs (Decision Branches)
    draw_box(2.8, 0.45, 3.8, 0.6, box_green, border_green, "IDENTIFIABLE (ΔBIC > 10)", text_color=text_green)
    draw_box(7.2, 0.45, 3.8, 0.6, box_red, border_red, "WITHHELD (ΔBIC ≤ 10)", text_color=text_red)

    os.makedirs("docs/assets", exist_ok=True)
    out_path = "docs/assets/adcd_flowchart.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"Flowchart successfully generated at: {out_path}")

if __name__ == "__main__":
    create_adcd_flowchart()
