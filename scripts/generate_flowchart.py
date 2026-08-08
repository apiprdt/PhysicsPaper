import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_adcd_flowchart():
    fig, ax = plt.subplots(figsize=(9.5, 13.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Background
    bg_color = "#ffffff"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Neutral Palette (Clean Minimalist Technical Palette)
    box_neutral = "#f8fafc"
    border_neutral = "#cbd5e1"
    text_dark = "#0f172a"
    text_muted = "#475569"
    arrow_color = "#64748b"

    # Accent Colors (ONLY for Important Outlines / Decision Outcomes)
    border_gate = "#3b82f6"       # Subtle Blue for Core Physical Gates
    text_gate = "#1d4ed8"
    
    box_green = "#f0fdf4"          # Green for IDENTIFIABLE
    border_green = "#16a34a"
    text_green = "#15803d"
    
    box_red = "#fff1f2"            # Red for WITHHELD
    border_red = "#e11d48"
    text_red = "#be123c"

    def draw_box(x, y, w, h, bg, border, title, desc=None, text_color=text_dark, rx=0.2, title_size=10.5):
        box = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0.08,rounding_size={rx}",
            linewidth=1.2, edgecolor=border, facecolor=bg
        )
        ax.add_patch(box)
        
        if desc:
            ax.text(x, y + h*0.22, title, fontsize=title_size, fontweight='bold', ha='center', va='center', color=text_color)
            ax.text(x, y - h*0.18, desc, fontsize=8.5, ha='center', va='center', color=text_muted, multialignment='center')
        else:
            ax.text(x, y, title, fontsize=title_size, fontweight='bold', ha='center', va='center', color=text_color)

    def draw_arrow(x1, y1, x2, y2, label=None, color=arrow_color):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.55", lw=1.5, color=color)
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx + 0.15, my, label, fontsize=8.5, fontweight='bold', color=color, va='center')

    # Title Banner (Minimal & Clean)
    ax.text(5, 13.4, "ADCD Framework Architecture", fontsize=14, fontweight='bold', ha='center', color=text_dark)
    ax.text(5, 13.05, "Deterministic & Identifiability-Aware Physics Discovery Pipeline", fontsize=9, ha='center', color=text_muted)

    # 1. Inputs
    draw_box(5, 12.2, 7.8, 0.85, box_neutral, border_neutral, 
             "1. Observational Data & Classical Baseline", 
             "Inputs: X, y_obs, y_cl  |  Metadata: Units [L, M, T] & Limit Regime")
    
    draw_arrow(5, 11.77, 5, 11.18)

    # 2. Candidate Generation
    draw_box(5, 10.6, 7.8, 1.05, box_neutral, border_neutral, 
             "2. Candidate Enumeration (GrammarProposerV3)", 
             "• Buckingham-π Auto-derived Dimensionless Ratios\n• 5 Asymptotic Primitives  |  Generation Budget ≤ 25 Tokens\n• Candidate Pool Size |C| ≈ 140 – 260")

    draw_arrow(5, 10.07, 5, 9.48, label="|C| Candidates")

    # 3. Physical Gate Cascade
    cascade_bg = patches.FancyBboxPatch(
        (1.1, 4.75), 7.8, 4.4,
        boxstyle="round,pad=0.08,rounding_size=0.25",
        linewidth=1.2, edgecolor=border_gate, facecolor="#ffffff", linestyle="--"
    )
    ax.add_patch(cascade_bg)
    ax.text(5, 8.9, "3. FIVE-GATE PHYSICAL CASCADE (Pre-fitting Filter)", fontsize=9.5, fontweight='bold', ha='center', color=text_gate)

    gates = [
        ("Gate 1: AST Complexity", "SymPy Depth ≤ 12, Nodes ≤ 50"),
        ("Gate 2: Dimensional Check", "SI Unit Vectors Must Balance"),
        ("Gate 3: Transcendental Safety", "log(u>0), sqrt(u≥0) Domain Check"),
        ("Gate 4: Asymptotic Check (ARC)", "Hard Constraint: lim(u→0) Δ = 0"),
        ("Gate 5: Coarse Pre-Filter", "Discard NaN / inf Candidate Fits")
    ]
    
    gate_y = 8.35
    for i, (g_title, g_desc) in enumerate(gates):
        # ARC gate subtle highlight
        b_color = border_gate if i == 3 else border_neutral
        t_color = text_gate if i == 3 else text_dark
        
        draw_box(5, gate_y - i*0.75, 7.2, 0.58, box_neutral, b_color, g_title + " — " + g_desc, text_color=t_color, rx=0.12, title_size=9.5)
        if i < 4:
            draw_arrow(5, gate_y - i*0.75 - 0.29, 5, gate_y - (i+1)*0.75 + 0.29, color=arrow_color)

    draw_arrow(5, 4.75, 5, 4.15, label="~10–50 Survivors")

    # 4. JAX L-BFGS-B Optimization
    draw_box(5, 3.6, 7.8, 1.05, box_neutral, border_neutral, 
             "4. Float64 JAX L-BFGS-B Numerical Optimizer", 
             "• Parameters in Log-Space: θ_i = s_i · exp(u_i)\n• Multi-Start Optimization (n_restarts = 15 per candidate)\n• High-Precision Loss Evaluation on Residuals")

    draw_arrow(5, 3.07, 5, 2.48)

    # 5. Model Selection & BIC Ranking
    draw_box(5, 1.9, 7.8, 0.95, box_neutral, border_neutral, 
             "5. Bayesian Model Selection & Pareto Front", 
             "• BIC = k ln N - 2 ln L̂  |  Parsimony-vs-Fitness Ranking\n• True Structure vs. Ablated Best Search Comparison")

    draw_arrow(5, 1.42, 2.8, 0.73, color=border_green)
    draw_arrow(5, 1.42, 7.2, 0.73, color=border_red)

    # 6. Final Outputs (Decision Branches - Accented Colors for Highlight)
    draw_box(2.8, 0.45, 3.8, 0.58, box_green, border_green, "IDENTIFIABLE (ΔBIC > 10)", text_color=text_green, title_size=10)
    draw_box(7.2, 0.45, 3.8, 0.58, box_red, border_red, "WITHHELD (ΔBIC ≤ 10)", text_color=text_red, title_size=10)

    os.makedirs("docs/assets", exist_ok=True)
    out_path = "docs/assets/adcd_flowchart.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"Minimalist Flowchart successfully generated at: {out_path}")

if __name__ == "__main__":
    create_adcd_flowchart()
