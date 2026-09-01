"""
Generate fig4_pipeline.pdf - ADCD Architecture Pipeline Figure
A clean horizontal flowchart showing the 5-stage ADCD pipeline with 3-tier epistemic verdicts.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF    = os.path.join(SCRIPT_DIR, "fig4_pipeline.pdf")
OUT_PNG    = os.path.join(SCRIPT_DIR, "fig4_pipeline.png")

fig, ax = plt.subplots(figsize=(15.5, 4.8))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 4.8)
ax.axis("off")

# Color scheme
CLR_INPUT   = "#e8f5e9"
CLR_DETECT  = "#fff9c4"
CLR_PROP    = "#e3f2fd"
CLR_GATE    = "#fce4ec"
CLR_OPT     = "#e8eaf6"
CLR_VAL     = "#f3e5f5"
CLR_IDENT   = "#c8e6c9"
CLR_UNRES   = "#fff3e0"
CLR_WITH    = "#ffebee"
CLR_EDGE    = "#546e7a"

def draw_box(ax, x, y, w, h, label, sublabel, color, fontsize=9.2):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor=CLR_EDGE,
                          linewidth=1.3, zorder=3)
    ax.add_patch(box)
    cx, cy = x + w/2, y + h/2
    if sublabel:
        ax.text(cx, cy + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=7.8, color="#37474f", zorder=4, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)

def arrow(ax, x1, x2, y=2.05, color="#455a64"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14),
                zorder=5)

def branch_arrow(ax, x_from, y_from, x_to, y_to, label, color):
    ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.3, mutation_scale=11),
                zorder=5)
    mx, my = (x_from + x_to)/2, (y_from + y_to)/2
    ax.text(mx, my + 0.08, label, fontsize=6.8, color=color,
            ha="center", va="bottom", zorder=6, fontweight="semibold")

BOX_Y  = 1.15
BOX_H  = 1.8
GAP    = 0.16
W_INPUT  = 1.55
W_DETECT = 1.65
W_PROP   = 1.80
W_GATE   = 2.05
W_OPT    = 1.80
W_VAL    = 2.00

x0 = 0.15
x1 = x0 + W_INPUT + GAP
x2 = x1 + W_DETECT + GAP
x3 = x2 + W_PROP + GAP
x4 = x3 + W_GATE + GAP
x5 = x4 + W_OPT + GAP

# Stage boxes
draw_box(ax, x0, BOX_Y, W_INPUT,  BOX_H, "Input",
         "y_obs, y_cl, X, units\nResidual: Δ", CLR_INPUT, fontsize=8.8)
draw_box(ax, x1, BOX_Y, W_DETECT, BOX_H, "Stage 1\nMode Detection",
         "Spearman |ρ|\nAdd. vs Mul. Mode", CLR_DETECT, fontsize=8.6)
draw_box(ax, x2, BOX_Y, W_PROP,   BOX_H, "Stage 2\nTemplate Proposal",
         "Buckingham-π\n|C| ≤ 260 candidates", CLR_PROP, fontsize=8.6)
draw_box(ax, x3, BOX_Y, W_GATE,   BOX_H, "Stage 3\nFilter Cascade",
         "Gates A–E (Julia)\nDim / Asymp / NMSE", CLR_GATE, fontsize=8.6)
draw_box(ax, x4, BOX_Y, W_OPT,    BOX_H, "Stage 4\nContinuous Fit",
         "Nelder-Mead (Julia)\n50 Restarts / Float64", CLR_OPT, fontsize=8.6)
draw_box(ax, x5, BOX_Y, W_VAL,    BOX_H, "Stage 5\n4-Step Validation",
         "Pos / Ablation ΔBIC\nReproducibility", CLR_VAL, fontsize=8.6)

# Horizontal arrows
for xa, xb in [(x0+W_INPUT, x1), (x1+W_DETECT, x2),
               (x2+W_PROP, x3), (x3+W_GATE, x4),
               (x4+W_OPT, x5)]:
    arrow(ax, xa, xb, y=BOX_Y + BOX_H/2)

# Verdict outputs (3 Tiers)
verdict_y   = BOX_Y + BOX_H + 0.52
verdict_h   = 0.62
verdict_w   = 1.48
cx_val      = x5 + W_VAL/2

# 1. IDENTIFIABLE
xi = cx_val - verdict_w - 0.22
box_i = FancyBboxPatch((xi, verdict_y), verdict_w, verdict_h,
                        boxstyle="round,pad=0.05",
                        facecolor=CLR_IDENT, edgecolor="#2e7d32", linewidth=1.4, zorder=3)
ax.add_patch(box_i)
ax.text(xi + verdict_w/2, verdict_y + verdict_h/2 + 0.07, "IDENTIFIABLE",
        ha="center", va="center", fontsize=8.2, fontweight="bold",
        color="#1b5e20", zorder=4)
ax.text(xi + verdict_w/2, verdict_y + verdict_h/2 - 0.14, "Formal 4-Step Pass",
        ha="center", va="center", fontsize=6.8, color="#2e7d32", zorder=4)

# 2. CANDIDATE
xu = cx_val - verdict_w/2
box_u = FancyBboxPatch((xu, verdict_y), verdict_w, verdict_h,
                        boxstyle="round,pad=0.05",
                        facecolor=CLR_UNRES, edgecolor="#ef6c00", linewidth=1.4, zorder=3)
ax.add_patch(box_u)
ax.text(xu + verdict_w/2, verdict_y + verdict_h/2 + 0.07, "CANDIDATE",
        ha="center", va="center", fontsize=7.2, fontweight="bold",
        color="#e65100", zorder=4)
ax.text(xu + verdict_w/2, verdict_y + verdict_h/2 - 0.14, "Ambiguous / ΔBIC < 10",
        ha="center", va="center", fontsize=6.8, color="#bf360c", zorder=4)

# 3. WITHHELD
xw = cx_val + verdict_w/2 + 0.22
box_w = FancyBboxPatch((xw, verdict_y), verdict_w, verdict_h,
                        boxstyle="round,pad=0.05",
                        facecolor=CLR_WITH, edgecolor="#c62828", linewidth=1.4, zorder=3)
ax.add_patch(box_w)
ax.text(xw + verdict_w/2, verdict_y + verdict_h/2 + 0.07, "WITHHELD",
        ha="center", va="center", fontsize=8.2, fontweight="bold",
        color="#b71c1c", zorder=4)
ax.text(xw + verdict_w/2, verdict_y + verdict_h/2 - 0.14, "Null Model Consistent",
        ha="center", va="center", fontsize=6.8, color="#c62828", zorder=4)

# Arrows to 3 verdicts
branch_arrow(ax, cx_val - 0.35, BOX_Y + BOX_H,
             xi + verdict_w/2, verdict_y,
             "All Pass", "#2e7d32")
branch_arrow(ax, cx_val, BOX_Y + BOX_H,
             xu + verdict_w/2, verdict_y,
             "Evidence > Null", "#ef6c00")
branch_arrow(ax, cx_val + 0.35, BOX_Y + BOX_H,
             xw + verdict_w/2, verdict_y,
             "Null Baseline", "#c62828")

# Title
ax.text(7.75, 4.45,
        "ADCD Asymptotic Discovery & Epistemic Certification Pipeline",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1a237e")

plt.tight_layout(pad=0.1)
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
plt.close()
print("Successfully generated Fig 4 PDF & PNG with 3-tier verdicts: " + OUT_PDF)
