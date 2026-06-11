#!/usr/bin/env python3
"""Generate a clean, professional animated GIF for ADCD.

Produces docs/adcd_discovery.gif — a minimalist dark-mode animation:
  Phase 1: Title card
  Phase 2: Data plot — anomaly vs classical baseline  
  Phase 3: Gate filtering (white text, green pass, red fail)
  Phase 4: Optimization convergence
  Phase 5: Result card with discovered formula

Color rules:
  - Labels/headers: grey (#8b949e)
  - Important text & numbers: white (#e6edf3)
  - Success/pass/match: green (#3fb950)
  - Failure/reject: red (#f85149)
  - Accent/brand: teal (#2dd4bf) — only for ADCD brand name + final formula
  - In-progress/pending: yellow (#d29922)

Usage:
    python scripts/generate_adcd_animation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent.parent / "docs" / "adcd_discovery.gif"
OUT.parent.mkdir(exist_ok=True)

# ── Palette (intuitive: white=info, green=pass, red=fail) ──────────────
BG      = "#0d1117"
SURFACE = "#161b22"
TEAL    = "#2dd4bf"       # brand accent only
BLUE    = "#58a6ff"       # classical baseline curve
ORANGE  = "#f0883e"       # observed data points
RED     = "#f85149"       # failure / reject
GREEN   = "#3fb950"       # success / pass
YELLOW  = "#d29922"       # in-progress / pending
GREY    = "#8b949e"       # labels, headers, secondary
WHITE   = "#e6edf3"       # primary text & numbers
FAINT   = "#30363d"       # borders

# ── Physics data ───────────────────────────────────────────────────────
np.random.seed(42)
N = 60
p = np.linspace(0.1, 3.8, N)
m, c = 1.0, 1.0
E_cls = p**2 / (2*m)
E_true = np.sqrt(p**2*c**2 + m**2*c**4) - m*c**2
E_obs = E_true + np.random.normal(0, 0.12, N)

# ── Frame plan ─────────────────────────────────────────────────────────
TOTAL = 210
FPS = 12

def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2*t)

# ── Figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 5.2), facecolor=BG, dpi=110)

# Left: data plot
ax = fig.add_axes([0.07, 0.13, 0.52, 0.78])
ax.set_facecolor(SURFACE)
for sp in ax.spines.values():
    sp.set_color(FAINT)
    sp.set_linewidth(0.8)
ax.tick_params(colors=GREY, labelsize=7)
ax.set_xlim(-0.1, 4.2)
ax.set_ylim(-0.8, max(E_obs) + 1.2)
ax.set_xlabel("Momentum  p", color=GREY, fontsize=8, labelpad=6)
ax.set_ylabel("Energy  E", color=GREY, fontsize=8, labelpad=6)

scat = ax.scatter([], [], s=14, color=ORANGE, alpha=0, zorder=3, linewidths=0)
line_cls, = ax.plot([], [], color=BLUE, ls="--", lw=1.5, alpha=0, zorder=2)
line_fit, = ax.plot([], [], color=TEAL, lw=2.2, alpha=0, zorder=4)

# ── Right panel text (fixed positions, no axes, no collision) ──────────
RX = 0.80   # center x for right panel
title_main = fig.text(RX, 0.93, "", ha="center", va="top",
                       fontsize=16, fontweight="bold", color=TEAL,
                       fontfamily="monospace")
title_sub  = fig.text(RX, 0.87, "", ha="center", va="top",
                       fontsize=8.5, color=GREY, fontfamily="sans-serif")

# Divider line
fig.text(RX, 0.835, "________________________", ha="center", va="top",
         fontsize=6, color=FAINT, fontfamily="monospace")

# 10 status lines, left-aligned at fixed columns to avoid overlaps
SX_lbl = 0.64   # left edge for labels / candidates
SX_val = 0.82   # left edge for values / status
SY0 = 0.79
SDY = 0.058
status_lbl = []
status_val = []
for i in range(10):
    lbl = fig.text(SX_lbl, SY0 - i*SDY, "", ha="left", va="top",
                  fontsize=8.5, color=WHITE, fontfamily="monospace")
    val = fig.text(SX_val, SY0 - i*SDY, "", ha="left", va="top",
                  fontsize=8.5, color=WHITE, fontfamily="monospace")
    status_lbl.append(lbl)
    status_val.append(val)

# Formula (centered, larger)
formula = fig.text(RX, 0.15, "", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=TEAL,
                    fontfamily="monospace")

# Bottom bar
prog = fig.text(0.50, 0.03, "", ha="center", va="center",
                 fontsize=7, color=GREY, fontfamily="monospace")

# ── Gate data ──────────────────────────────────────────────────────────
GATES = [
    ("Dimensionality", "PASS"),
    ("Asymptotics",    "PASS"),
    ("Symmetries",     "PASS"),
    ("Complexity",     "PASS"),
]

def _clear():
    for s in status_lbl:
        s.set_text("")
        s.set_color(WHITE)
        s.set_alpha(1.0)
    for s in status_val:
        s.set_text("")
        s.set_color(WHITE)
        s.set_alpha(1.0)
    formula.set_text("")
    formula.set_alpha(1.0)

def update(frame):
    # ── Phase 1: Title (0-24) ───────────────────────────────────
    if frame < 25:
        a = ease(frame / 20)
        title_main.set_text("ADCD")
        title_main.set_alpha(a)
        title_sub.set_text("Anomaly-Driven Correction Discovery")
        title_sub.set_alpha(a * 0.8)
        _clear()
        status_lbl[1].set_text("Physics-constrained")
        status_lbl[1].set_color(GREY)
        status_lbl[1].set_alpha(a * 0.7)
        status_lbl[2].set_text("symbolic regression")
        status_lbl[2].set_color(GREY)
        status_lbl[2].set_alpha(a * 0.7)
        status_lbl[3].set_text("for theory refinement")
        status_lbl[3].set_color(GREY)
        status_lbl[3].set_alpha(a * 0.7)
        prog.set_text("")
        return []

    # ── Phase 2: Data appears (25-54) ──────────────────────────
    elif frame < 55:
        f = frame - 25
        a = ease(f / 25)
        n = max(1, int(a * N))

        scat.set_offsets(np.column_stack([p[:n], E_obs[:n]]))
        scat.set_alpha(0.75)
        line_cls.set_data(p[:n], E_cls[:n])
        line_cls.set_alpha(0.8)

        _clear()
        title_main.set_text("ADCD")
        title_sub.set_text("Loading Scenario")
        title_sub.set_color(GREY)

        status_lbl[0].set_text("Scenario:")
        status_lbl[0].set_color(GREY)
        status_lbl[1].set_text("  Relativistic KE")
        status_lbl[1].set_color(WHITE)

        status_lbl[3].set_text("Baseline:")
        status_lbl[3].set_color(GREY)
        status_lbl[4].set_text("  E = p^2 / 2m")
        status_lbl[4].set_color(WHITE)

        status_lbl[6].set_text("Points:")
        status_lbl[6].set_color(GREY)
        status_val[6].set_text(f"{n}/{N}")
        status_val[6].set_color(WHITE)

        prog.set_text(f"loading data  {int(a*100)}%")
        return []

    # ── Phase 3: Gate cascade (55-94) ──────────────────────────
    elif frame < 95:
        f = frame - 55
        _clear()

        title_main.set_text("ADCD")
        title_sub.set_text("Stage 1: Physics Gates")
        title_sub.set_color(WHITE)

        status_lbl[0].set_text("Candidate:")
        status_lbl[0].set_color(GREY)
        status_lbl[1].set_text("  theta_0 * (v/c)^2")
        status_lbl[1].set_color(WHITE)

        status_lbl[3].set_text("Physics Gate")
        status_lbl[3].set_color(GREY)
        status_val[3].set_text("Status")
        status_val[3].set_color(GREY)

        idx = min(f // 8, len(GATES) - 1)
        sub = f % 8

        for i in range(min(idx + 1, len(GATES))):
            gate_name, result = GATES[i]
            line_i = i + 4  # offset by 4 for Candidate and headers

            status_lbl[line_i].set_text(f"  {gate_name}")
            status_lbl[line_i].set_color(WHITE)

            if i < idx or sub >= 4:
                # Decided
                status_val[line_i].set_text("PASS")
                status_val[line_i].set_color(GREEN)
            else:
                # Checking
                status_val[line_i].set_text("...")
                status_val[line_i].set_color(YELLOW)

        # Tally
        done = [g for j, g in enumerate(GATES[:idx+1]) if j < idx or sub >= 4]
        n_pass = sum(1 for g in done if g[1] == "PASS")

        status_lbl[9].set_text("Verification:")
        status_lbl[9].set_color(GREY)
        if n_pass == len(GATES):
            status_val[9].set_text("SUCCESS")
            status_val[9].set_color(GREEN)
        else:
            status_val[9].set_text("PENDING")
            status_val[9].set_color(YELLOW)

        prog.set_text(f"checking gates  {idx+1}/{len(GATES)}")
        return []

    # ── Phase 4: Optimization (95-154) ─────────────────────────
    elif frame < 155:
        f = frame - 95
        t = f / 59.0
        decay = 1.0 - np.exp(-5.0 * t)

        E_fit = E_cls + decay * (E_true - E_cls)
        line_fit.set_data(p, E_fit)
        line_fit.set_alpha(min(1.0, t * 3))

        residual = E_obs - E_fit
        nmse = np.mean(residual**2) / max(np.var(E_obs), 1e-10)

        _clear()
        title_main.set_text("ADCD")
        title_sub.set_text("Stage 2: JAX Optimizer")
        title_sub.set_color(WHITE)

        status_lbl[0].set_text("Candidate:")
        status_lbl[0].set_color(GREY)
        status_lbl[1].set_text("  theta_0 * (v/c)^2")
        status_lbl[1].set_color(WHITE)

        iter_n = int(t * 50)
        status_lbl[3].set_text("Iteration:")
        status_lbl[3].set_color(GREY)
        status_val[3].set_text(f"{iter_n}/50")
        status_val[3].set_color(WHITE)

        theta = 0.01 + decay * 0.49
        status_lbl[5].set_text("theta_0:")
        status_lbl[5].set_color(GREY)
        status_val[5].set_text(f"{theta:.5f}")
        status_val[5].set_color(WHITE)

        status_lbl[7].set_text("NMSE:")
        status_lbl[7].set_color(GREY)
        status_val[7].set_text(f"{nmse:.2e}")
        if nmse < 0.005:
            status_val[7].set_color(GREEN)
        else:
            status_val[7].set_color(WHITE)

        if t > 0.75:
            status_lbl[9].set_text("Status:")
            status_lbl[9].set_color(GREY)
            status_val[9].set_text("Converged")
            status_val[9].set_color(GREEN)

        prog.set_text(f"optimizing  iter {iter_n}/50")
        return []

    # ── Phase 5+6: Result (155-209) ────────────────────────────
    else:
        f = frame - 155
        a = ease(min(f / 12, 1.0))

        line_fit.set_data(p, E_true)
        line_fit.set_alpha(1.0)

        _clear()
        title_main.set_text("ADCD")
        title_sub.set_text("Discovery Complete")
        title_sub.set_color(GREEN)

        status_lbl[0].set_text("Scenario:")
        status_lbl[0].set_color(GREY)
        status_lbl[1].set_text("  Relativistic KE")
        status_lbl[1].set_color(WHITE)

        status_lbl[3].set_text("Class:")
        status_lbl[3].set_color(GREY)
        status_val[3].set_text("polynomial")
        status_val[3].set_color(WHITE)

        status_lbl[4].set_text("Match:")
        status_lbl[4].set_color(GREY)
        status_val[4].set_text("SUCCESS")
        status_val[4].set_color(GREEN)

        status_lbl[6].set_text("NMSE:")
        status_lbl[6].set_color(GREY)
        status_val[6].set_text("1.11e-05")
        status_val[6].set_color(GREEN)

        formula.set_text("D = theta_0 * (v/c)^2")
        formula.set_alpha(a)

        prog.set_text("pip install adcd")
        prog.set_color(GREY)
        return []


# ── Build & save ───────────────────────────────────────────────────────
print(f"Generating {TOTAL} frames at {FPS} fps ...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL,
                               interval=1000//FPS, blit=False)
print(f"Saving to {OUT} ...")
ani.save(str(OUT), writer="pillow", fps=FPS, dpi=110,
         savefig_kwargs={"facecolor": BG, "edgecolor": "none"})
sz = OUT.stat().st_size / 1024
print(f"Done: {OUT} ({sz:.0f} KB)")
plt.close()
