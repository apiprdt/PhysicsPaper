#!/usr/bin/env python3
"""Generate a polished, editorial-style animation for ADCD.

Produces, by default:
    docs/adcd_discovery.gif      (palette-optimized, for README / docs)
    docs/adcd_discovery.mp4      (high-bitrate 1080p, for talks / presentations)

Aesthetic: "Editorial Light" — off-white paper background, navy / coral /
slate palette, despined thin axes, serif title, italic LaTeX equations,
soft crossfaded scenes. Reads like a figure in Nature / Science rather
than a CI console log.

Pipeline: matplotlib renders PNG frames to a temp dir, then ffmpeg encodes
the MP4 (libx264, crf 18) and a two-pass palette GIF.

Usage:
    python scripts/generate_adcd_animation.py            # full render
    python scripts/generate_adcd_animation.py --test     # 6 sample frames only
    python scripts/generate_adcd_animation.py --no-encode# frames only, no ffmpeg
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_GIF = ROOT / "docs" / "adcd_discovery.gif"
OUT_MP4 = ROOT / "docs" / "adcd_discovery.mp4"
OUT_GIF.parent.mkdir(exist_ok=True)

# ── Editorial Light palette ────────────────────────────────────────────
PAPER   = "#fafafa"   # off-white paper background
CARD    = "#ffffff"   # info-card surface
INK     = "#1a1a1a"   # primary text
NAVY    = "#1e3a5f"   # fit curve / primary accent
NAVY2   = "#2e5a8a"   # lighter navy
CORAL   = "#c0504d"   # observed data
SLATE   = "#6b7280"   # labels / secondary text
SLATE2  = "#9ca3af"   # tertiary / ticks
MINT    = "#4a9d8e"   # success / pass (muted)
AMBER   = "#b8860b"   # highlight / in-progress
GRID    = "#ededed"   # faint gridlines
BORDER  = "#e5e7eb"   # card border
RULE    = "#d1d5db"   # header / footer rules

# Fonts (all verified present on this machine)
SERIF = "Cambria"
SANS  = "Segoe UI"
MONO  = "Consolas"

# ── Physics data (relativistic kinetic energy) ─────────────────────────
np.random.seed(42)
N = 60
p = np.linspace(0.1, 3.8, N)
m, c = 1.0, 1.0
E_cls = p**2 / (2 * m)                                   # classical baseline
E_true = np.sqrt(p**2 * c**2 + m**2 * c**4) - m * c**2   # relativistic (ground truth)
E_obs = E_true + np.random.normal(0, 0.12, N)            # noisy observations

# ── Timing ─────────────────────────────────────────────────────────────
FPS = 30
TOTAL = 240                  # 8.0 s loop
# Scene content ownership (non-overlapping); card fades at each boundary.
SCENE_BOUNDS = [0, 37, 97, 155, 213, TOTAL]
# Boundaries where the card dips to swap content:
DIP_AT = [37, 97, 155, 213]
DIP_HALF = 7                # frames of crossfade each side of a boundary


# ── Easing ─────────────────────────────────────────────────────────────
def clamp(x):
    return max(0.0, min(1.0, x))


def ease(t):
    """smoothstep."""
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def card_alpha(f):
    """Card content opacity: 1 inside a scene, dipping to ~0 at boundaries."""
    a = 1.0
    for b in DIP_AT:
        d = abs(f - b)
        if d < DIP_HALF:
            a *= ease(d / DIP_HALF)
    return a


# ── Evolution of the reference (fit) curve across the whole timeline ────
def fit_progress(f):
    """Return u in [0,1]: how far the fit has morphed classical→true."""
    # 0 before S4 start; ramps during S4 (155..213); 1 after.
    if f < 153:
        return 0.0
    if f > 213:
        return 1.0
    return ease_out((f - 153) / 60.0)


def E_ref(f):
    u = fit_progress(f)
    return E_cls + u * (E_true - E_cls)


# ── Figure scaffolding ─────────────────────────────────────────────────
def build_figure():
    fig = plt.figure(figsize=(16, 9), dpi=110, facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)

    # ── Header ───────────────────────────────────────────────────────
    fig.text(0.061, 0.932, "ADCD", fontsize=30, fontweight="bold",
             color=INK, family=SERIF, ha="left", va="center")
    fig.text(0.061, 0.884, "Anomaly-Driven Correction Discovery",
             fontsize=13.5, color=SLATE, family=SERIF, style="italic",
             ha="left", va="center")

    scene_lbl = fig.text(0.939, 0.932, "", fontsize=11, color=NAVY,
                         family=SANS, fontweight="bold", ha="right",
                         va="center")
    scene_sub = fig.text(0.939, 0.888, "", fontsize=9.5, color=SLATE,
                         family=SANS, ha="right", va="center")

    # Header rule
    hdr_rule = plt.Line2D([0.061, 0.939], [0.858, 0.858], transform=fig.transFigure,
                          color=RULE, lw=0.8)
    fig.add_artist(hdr_rule)

    # ── Main plot axes ───────────────────────────────────────────────
    ax = fig.add_axes([0.061, 0.135, 0.530, 0.665])
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE2)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=SLATE2, labelsize=10, length=4, width=0.8)
    ax.set_xlim(-0.1, 4.2)
    ax.set_ylim(-0.6, max(E_obs) + 1.4)
    ax.set_xlabel("momentum  $p$", color=SLATE, fontsize=12,
                  family=SERIF, labelpad=8)
    ax.set_ylabel("energy  $E$", color=SLATE, fontsize=12,
                  family=SERIF, labelpad=8)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Anomaly shading (baseline vs ground truth), alpha modulated in S2
    anom_fill = ax.fill_between(p, E_cls, E_true, color=CORAL, alpha=0.0,
                                zorder=1, linewidth=0)

    # Baseline (classical) dashed curve
    line_base, = ax.plot([], [], color=SLATE2, ls=(0, (5, 4)), lw=1.6,
                         alpha=0.0, zorder=2)

    # Fit curve (solid navy) — the evolving discovery
    line_fit, = ax.plot([], [], color=NAVY, lw=2.6, alpha=0.0,
                        solid_capstyle="round", zorder=4)

    # Residual ticks: data → reference curve (shrink as fit converges)
    resid = LineCollection([], colors=CORAL, linewidths=0.9, alpha=0.0,
                           zorder=3)
    ax.add_collection(resid)

    # Data scatter
    scat = ax.scatter([], [], s=26, color=CORAL, alpha=0.0, zorder=5,
                      edgecolors="white", linewidths=0.5)

    # Inline curve labels
    lbl_base = ax.text(0, 0, "", color=SLATE, fontsize=10.5, family=SANS,
                       alpha=0.0, style="italic")
    lbl_fit = ax.text(0, 0, "", color=NAVY, fontsize=10.5, family=SANS,
                      fontweight="bold", alpha=0.0)

    # Plot-area caption shown only during the intro scene (fades as data
    # appears), so the empty plot never reads as "broken".
    plot_caption = ax.text(0.5, 0.62,
                           "watch the data diverge\nfrom the classical baseline",
                           transform=ax.transAxes, ha="center", va="center",
                           fontsize=11, color=SLATE, family=SERIF, style="italic",
                           alpha=0.0, zorder=2,
                           linespacing=1.6)

    # ── Info card (right) ────────────────────────────────────────────
    card_l, card_b, card_w, card_h = 0.635, 0.135, 0.310, 0.665
    # subtle shadow
    shadow = FancyBboxPatch((card_l + 0.004, card_b - 0.006), card_w, card_h,
                            boxstyle="round,pad=0.0,rounding_size=0.012",
                            transform=fig.transFigure, facecolor="#e7e7ea",
                            edgecolor="none", zorder=6, alpha=0.55)
    fig.add_artist(shadow)
    card = FancyBboxPatch((card_l, card_b), card_w, card_h,
                          boxstyle="round,pad=0.0,rounding_size=0.012",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor=BORDER, lw=1.0, zorder=7)
    fig.add_artist(card)

    cx0 = card_l + 0.022       # left text margin
    cxR = card_l + card_w - 0.022
    cxc = (card_l + card_l + card_w) / 2

    card_hdr = fig.text(cx0, card_b + card_h - 0.052, "", fontsize=10,
                        color=NAVY, family=SANS, fontweight="bold",
                        ha="left", va="center", zorder=9)
    card_rule = plt.Line2D([cx0, cxR],
                           [card_b + card_h - 0.078, card_b + card_h - 0.078],
                           transform=fig.transFigure, color=BORDER, lw=0.8,
                           zorder=9)
    fig.add_artist(card_rule)

    rows = []
    ry0 = card_b + card_h - 0.115
    rdy = 0.0455
    for i in range(7):
        t = fig.text(cx0, ry0 - i * rdy, "", fontsize=11.5, color=INK,
                     family=SANS, ha="left", va="center", zorder=9)
        rows.append(t)

    # Check-mark artists (two short segments in fig coords), one per gate row
    checks = []
    for i in range(4):
        ln, = ax.plot([], [], color=MINT, lw=2.2, solid_capstyle="round",
                      alpha=0.0, zorder=10, transform=fig.transFigure,
                      clip_on=False)
        checks.append(ln)

    formula = fig.text(cxc, card_b + 0.255, "", fontsize=20, color=NAVY,
                       family=SERIF, ha="center", va="center", zorder=9)
    formula_sub = fig.text(cxc, card_b + 0.190, "", fontsize=9.5, color=SLATE,
                           family=SANS, ha="center", va="center", zorder=9)

    # Status pill near bottom of card
    status_bg = FancyBboxPatch((cxc - 0.062, card_b + 0.085), 0.124, 0.040,
                               boxstyle="round,pad=0.0,rounding_size=0.020",
                               transform=fig.transFigure, facecolor="#eef5f3",
                               edgecolor=MINT, lw=1.0, alpha=0.0, zorder=9)
    fig.add_artist(status_bg)
    status_txt = fig.text(cxc, card_b + 0.105, "", fontsize=10.5,
                          color=MINT, family=SANS, fontweight="bold",
                          ha="center", va="center", alpha=0.0, zorder=10)

    # ── Footer ───────────────────────────────────────────────────────
    foot_rule = plt.Line2D([0.061, 0.939], [0.078, 0.078],
                           transform=fig.transFigure, color=RULE, lw=0.8)
    fig.add_artist(foot_rule)
    fig.text(0.061, 0.045,
             "physics-constrained symbolic regression  ·  theory refinement",
             fontsize=9, color=SLATE, family=SANS, ha="left", va="center")
    prog = fig.text(0.939, 0.045, "", fontsize=9, color=SLATE, family=MONO,
                    ha="right", va="center")

    return {
        "fig": fig, "ax": ax, "scene_lbl": scene_lbl, "scene_sub": scene_sub,
        "anom_fill": anom_fill, "line_base": line_base, "line_fit": line_fit,
        "resid": resid, "scat": scat, "lbl_base": lbl_base, "lbl_fit": lbl_fit,
        "plot_caption": plot_caption,
        "card_hdr": card_hdr, "rows": rows, "checks": checks,
        "formula": formula, "formula_sub": formula_sub,
        "status_bg": status_bg, "status_txt": status_txt, "prog": prog,
        "card_l": card_l, "cx0": cx0,
    }


# ── Scene content ──────────────────────────────────────────────────────
GATES = [
    "dimensional homogeneity",
    "asymptotic consistency",
    "symmetry preservation",
    "complexity  (BIC)",
]


def clear_card(A):
    for r in A["rows"]:
        r.set_text("")
    for ck in A["checks"]:
        ck.set_alpha(0.0)
    A["formula"].set_text("")
    A["formula_sub"].set_text("")
    A["status_bg"].set_alpha(0.0)
    A["status_txt"].set_alpha(0.0)


def draw_check(A, idx, row_y, alpha):
    """Draw a check-mark glyph to the LEFT of the row text (in fig coords)."""
    if alpha <= 0.01:
        A["checks"][idx].set_data([], [])
        return
    x0 = A["cx0"] - 0.022          # sit just before the row text
    s = 0.010                       # x-size of check
    sy = s * (16 / 9)               # keep visually square in fig coords
    y = row_y - 0.004
    xs = [x0, x0 + s * 0.6, x0 + s * 1.9]
    ys = [y, y - sy, y + sy * 1.1]
    A["checks"][idx].set_data(xs, ys)
    A["checks"][idx].set_alpha(alpha)
    A["checks"][idx].set_alpha(alpha)


def set_scene_header(A, num, title):
    A["scene_lbl"].set_text(num)
    A["scene_sub"].set_text(title)
    A["card_hdr"].set_text(num + "  ·  " + title.upper())


def update(f, A):
    clear_card(A)
    ca = card_alpha(f)

    # determine scene
    if f < SCENE_BOUNDS[1]:
        scene = 1
    elif f < SCENE_BOUNDS[2]:
        scene = 2
    elif f < SCENE_BOUNDS[3]:
        scene = 3
    elif f < SCENE_BOUNDS[4]:
        scene = 4
    else:
        scene = 5

    # ── Plot evolution (continuous across whole timeline) ───────────
    u = fit_progress(f)
    Eref = E_ref(f)

    # data reveal: appears in S2
    if f < 34:
        reveal = 0.0
    elif f < 80:
        reveal = ease((f - 34) / 46.0)
    else:
        reveal = 1.0
    n = max(1, int(round(reveal * N)))
    A["scat"].set_offsets(np.column_stack([p[:n], E_obs[:n]]))
    A["scat"].set_alpha(0.90 * ease(min(1, (f - 34) / 10.0)))

    # baseline reveal in S2
    if f < 60:
        bprog = 0.0
    elif f < 92:
        bprog = ease((f - 60) / 32.0)
    else:
        bprog = 1.0
    nb = max(1, int(round(bprog * N)))
    A["line_base"].set_data(p[:nb], E_cls[:nb])
    A["line_base"].set_alpha(0.95 * ease(min(1, (f - 60) / 8.0)))
    A["lbl_base"].set_position((3.05, E_cls[np.searchsorted(p, 3.05)] + 0.18))
    A["lbl_base"].set_text(r"classical  $E = p^{2}/2m$")
    A["lbl_base"].set_alpha(0.85 * bprog)

    # anomaly shading: visible mainly in S2
    A["anom_fill"].set_alpha(0.10 * (1.0 - u) * ease(min(1, (f - 64) / 12.0))
                             * (1.0 if scene == 2 else 0.45))

    # fit curve: fades in as it diverges from baseline (S4), stays after
    if f < 150:
        fitalpha = 0.0
    elif f < 158:
        fitalpha = ease((f - 150) / 8.0)
    else:
        fitalpha = 1.0
    A["line_fit"].set_data(p, Eref)
    A["line_fit"].set_alpha(fitalpha)
    if u > 0.02 and fitalpha > 0.05:
        yi = np.searchsorted(p, 2.2)
        A["lbl_fit"].set_position((2.2, Eref[yi] + 0.28))
        A["lbl_fit"].set_text("ADCD fit")
        A["lbl_fit"].set_alpha(fitalpha)
    else:
        A["lbl_fit"].set_alpha(0.0)

    # residual ticks: data → Eref (only after data exists)
    if reveal > 0.02:
        segs = []
        for i in range(n):
            segs.append([(p[i], E_obs[i]), (p[i], Eref[i])])
        A["resid"].set_segments(segs)
        A["resid"].set_alpha(0.45 * reveal)
    else:
        A["resid"].set_segments([])

    # plot-area caption: fades in over the intro, out before data lands
    if f < 10:
        cap_a = ease(f / 10.0)
    elif f < 30:
        cap_a = 1.0
    elif f < 38:
        cap_a = 1.0 - ease((f - 30) / 8.0)
    else:
        cap_a = 0.0
    A["plot_caption"].set_alpha(cap_a * 0.85)

    # ── Card content per scene ──────────────────────────────────────
    rows = A["rows"]

    if scene == 1:
        set_scene_header(A, "ADCD", "overview")
        rows[0].set_text("Symbolic regression usually")
        rows[1].set_text("learns whole equations")
        rows[2].set_text("from scratch.")
        rows[3].set_text("")
        rows[4].set_text("ADCD instead discovers")
        rows[5].set_text("targeted correction terms —")
        rows[6].set_text("the logic of Newton → Einstein.")
        for r in rows:
            r.set_color(INK)
        A["formula"].set_text("")
        A["prog"].set_text("introducing")

    elif scene == 2:
        set_scene_header(A, "01", "anomaly")
        rows[0].set_text("Relativistic kinetic")
        rows[1].set_text("energy")
        rows[3].set_text("Classical baseline:")
        A["formula"].set_text(r"$E_{\rm cl}=\dfrac{p^{2}}{2m}$")
        A["formula_sub"].set_text("data diverges from it at high  p")
        A["prog"].set_text(f"loading data  {int(reveal*100):3d}%")

    elif scene == 3:
        set_scene_header(A, "02", "physics gates")
        rows[0].set_text("Candidate correction:")
        A["formula"].set_text(r"$\Delta E=\theta_{0}\!\left(\dfrac{v}{c}\right)^{\!2}$")
        A["formula_sub"].set_text("passed through cascaded gates")
        # gates cascade (rows 3..6 = 4 gate rows)
        sf = f - SCENE_BOUNDS[2]      # 0..58
        gi = min(int(sf / 11), len(GATES) - 1)
        sub = sf - gi * 11
        for i in range(min(gi + 1, len(GATES))):
            ry = 0.135 + 0.665 - 0.115 - (3 + i) * 0.0455
            rows[3 + i].set_text(GATES[i])
            rows[3 + i].set_color(INK)
            decided = (i < gi) or (sub >= 5)
            if decided:
                draw_check(A, i, ry, ca)
                rows[3 + i].set_color(INK)
            else:
                rows[3 + i].set_color(AMBER)
        npass = sum(1 for i in range(min(gi + 1, len(GATES)))
                    if (i < gi) or (sub >= 5))
        A["prog"].set_text(f"verifying  {npass}/{len(GATES)}")

    elif scene == 4:
        set_scene_header(A, "03", "optimization")
        rows[0].set_text("JAX gradient fit")
        it = int(u * 60)
        theta = 0.0 + u * 0.50
        rmse = float(np.sqrt(np.mean((E_obs - Eref) ** 2)))
        rows[2].set_text(f"iteration      {it:>3d} / 60")
        rows[3].set_text(rf"$\theta_0$  =  {theta:5.3f}")
        rows[4].set_text(f"RMSE       =  {rmse:5.3f}")
        rows[2].set_color(INK)
        rows[3].set_color(INK)
        rows[4].set_color(MINT if rmse < 0.20 else INK)
        A["formula"].set_text(r"$\Delta E=\theta_{0}\!\left(\dfrac{v}{c}\right)^{\!2}$")
        A["formula_sub"].set_text("minimising residual under the gates")
        A["prog"].set_text(f"optimizing  iter {it:3d}/60")

    else:  # scene 5
        set_scene_header(A, "04", "discovery")
        rmse = float(np.sqrt(np.mean((E_obs - E_true) ** 2)))
        rows[0].set_text("Recovered correction:")
        A["formula"].set_text(r"$\Delta E=\theta_{0}\!\left(\dfrac{v}{c}\right)^{\!2}$")
        A["formula_sub"].set_text(rf"$\theta_0 = 0.500$    ·    RMSE = {rmse:.3f}")
        A["status_bg"].set_alpha(ca)
        A["status_txt"].set_text("DISCOVERY VERIFIED")
        A["status_txt"].set_alpha(ca)
        A["prog"].set_text("pip install adcd")

    # apply card crossfade alpha to all card text artists
    card_artists = ([A["card_hdr"], A["formula"], A["formula_sub"]]
                    + rows)
    for art in card_artists:
        base = 1.0 if art.get_text() else 0.0
        art.set_alpha(base * ca)
    # scene header dips with the card swap too
    A["scene_lbl"].set_alpha(ca if A["scene_lbl"].get_text() else 0)
    A["scene_sub"].set_alpha(ca if A["scene_sub"].get_text() else 0)

    return []


# ── Render & encode ────────────────────────────────────────────────────
def render_frames(A, outdir, frames):
    fig = A["fig"]
    paths = []
    for i, f in enumerate(frames):
        update(f, A)
        fp = outdir / f"frame_{i:04d}.png"
        fig.savefig(fp, facecolor=PAPER, edgecolor="none", dpi=110)
        paths.append(fp)
    return paths


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def encode_mp4(frame_glob_dir, dest):
    run(["ffmpeg", "-y", "-framerate", str(FPS),
         "-i", str(frame_glob_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart", str(dest)])


def encode_gif(frame_glob_dir, dest):
    palette = frame_glob_dir / "palette.png"
    run(["ffmpeg", "-y", "-framerate", str(FPS),
         "-i", str(frame_glob_dir / "frame_%04d.png"),
         "-vf", "scale=880:-1:flags=lanczos,palettegen=stats_mode=diff",
         "-update", "1", str(palette)])
    run(["ffmpeg", "-y", "-framerate", str(FPS),
         "-i", str(frame_glob_dir / "frame_%04d.png"),
         "-i", str(palette),
         "-lavfi", "scale=880:-1:flags=lanczos [x]; [x][1:v] "
                   "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
         str(dest)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="render 6 sample frames only (no encode)")
    ap.add_argument("--no-encode", action="store_true",
                    help="render all frames, skip ffmpeg")
    args = ap.parse_args()

    A = build_figure()

    if args.test:
        samples = [15, 55, 125, 185, 230, 239]
        outdir = ROOT / "docs" / "_anim_test"
        outdir.mkdir(exist_ok=True)
        print(f"Rendering {len(samples)} test frames to {outdir} ...")
        for f in samples:
            update(f, A)
            fp = outdir / f"sample_{f:03d}.png"
            A["fig"].savefig(fp, facecolor=PAPER, edgecolor="none", dpi=110)
            print(f"  frame {f:>3d} -> {fp.name}")
        print("Done. Inspect the PNGs before a full render.")
        return

    tmp = Path(tempfile.mkdtemp(prefix="adcd_anim_"))
    try:
        frames = list(range(TOTAL))
        print(f"Rendering {TOTAL} frames to {tmp} ...")
        render_frames(A, tmp, frames)

        if args.no_encode:
            print(f"--no-encode: frames left in {tmp}")
            return

        print(f"Encoding MP4 -> {OUT_MP4.name} ...")
        encode_mp4(tmp, OUT_MP4)
        print(f"Encoding GIF -> {OUT_GIF.name} ...")
        encode_gif(tmp, OUT_GIF)

        for f in (OUT_MP4, OUT_GIF):
            sz = f.stat().st_size / 1024
            print(f"  {f.name}: {sz:,.0f} KB")
        print("Done.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
