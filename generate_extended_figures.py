"""Generate extended figures for ADCD paper Sections 5.7-5.9 + Appendix.

Figures:
  - paper/figures/fig_cosmological_probes.pdf
        (a) fσ8 residual Δ(z) + inverse-variance bin means + constant-offset fit
        (b) H(z) residual Δ_H(z) + constant-offset fit
  - paper/figures/fig_wide_binary.pdf
        Predicted velocity boost γv = √ν(x) vs separation for ADCD / Simple /
        Standard / RAR, with Chae 2023 observed band 1.20±0.06
  - paper/figures/fig_dichotomy.pdf
        2-panel summary: galactic functional vs cosmological amplitude

Run:
    py -3.11 generate_extended_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# Publication styling — matches generate_figures.py
rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs("paper/figures", exist_ok=True)

# Physical constants for wide-binary computation
G = 6.674e-11
MSUN = 1.989e30
AU = 1.496e11
A0 = 1.2e-10


# ---------------------------------------------------------------------------
# Interpolating functions (MOND family)
# ---------------------------------------------------------------------------
# ADCD-discovered SPARC form parameters (no refit)
T0_ADCD, T1_ADCD = 1.8281385981770948, 0.2615186295495788


def nu_adcd(x):
    return T0_ADCD * (np.sqrt(1.0 + T1_ADCD / x) - 1.0) + 1.0


def nu_simple(x):
    return (1.0 + np.sqrt(1.0 + 4.0 / x)) / 2.0


def nu_standard(x):
    return 1.0 / np.sqrt(1.0 - np.exp(-np.sqrt(x)))


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def sep_to_x(s_kau, m_total_msun=1.5):
    s_m = s_kau * 1e3 * AU
    g_n = G * m_total_msun * MSUN / s_m ** 2
    return g_n / A0


# ---------------------------------------------------------------------------
# Figure: cosmological probes (fσ8 + H(z))
# ---------------------------------------------------------------------------
def fig_cosmological_probes():
    """Two-panel residual plot for fσ8 and H(z), both showing constant wins."""
    # Load raw data to plot residuals
    growth = json.load(open("results/growth_rate_discovery.json"))
    cc = json.load(open("results/cosmic_chronometers.json"))

    # fσ8 bin means
    fs8_bins = growth["bin_means"]

    # H(z): recompute bins from raw data
    cc_rows = []
    with open("data/cosmic_chronometers/CC_Hz_compilation.txt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                cc_rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    cc_arr = np.array(cc_rows)
    z_cc, h_cc, sig_cc = cc_arr[:, 0], cc_arr[:, 1], cc_arr[:, 2]
    # residual = H_obs - H_GR, normalized to its error
    # We don't have H_GR per-point in JSON; use best-fit constant offset on residual
    # Reconstruct Δ_H residual using constant offset param
    delta_h_const = cc["candidates"][0]["params"]["theta_0"]
    # Pull = (Δ_obs - const)/σ_H, but here we just plot Δ_obs/σ_H ≈ pull
    # Simpler: show weighted bin means of (H_obs - H_GR_fit)
    # We approximate H_GR via the fitted (H0, Om); recompute here
    H0_fit = cc["H0_fit"]
    Om_fit = cc["Om_m0_fit"]
    h_gr = H0_fit * np.sqrt(Om_fit * (1 + z_cc) ** 3 + (1 - Om_fit))
    delta_h = h_cc - h_gr

    # Bin H(z) residual for display
    cc_bins_def = [(0.0, 0.4), (0.4, 0.9), (0.9, 1.4), (1.4, 2.5)]

    def weighted_bins(z, y, sig, bins):
        out = []
        for lo, hi in bins:
            m = (z >= lo) & (z < hi)
            if m.sum() == 0:
                continue
            w = 1.0 / sig[m] ** 2
            mean = float(np.sum(y[m] * w) / np.sum(w))
            err = float(1.0 / np.sqrt(np.sum(w)))
            out.append((0.5 * (lo + hi), mean, err, int(m.sum())))
        return out

    cc_bins = weighted_bins(z_cc, delta_h, sig_cc, cc_bins_def)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.patch.set_facecolor("white")

    # --- Panel (a): fσ8 ---
    ax1.set_facecolor("white")
    # Plot bin means with error bars
    bin_centers = [0.5 * (b["z_lo"] + b["z_hi"]) for b in fs8_bins]
    bin_means = [b["weighted_mean_Delta"] for b in fs8_bins]
    bin_errs = [b["error"] for b in fs8_bins]
    ax1.errorbar(bin_centers, bin_means, yerr=bin_errs, fmt="o",
                 color="#c0392b", markersize=7, capsize=4, zorder=5,
                 label="Inverse-variance bin mean")
    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=1)
    ax1.axhline(growth["candidates"][0]["params"]["theta_0"],
                color="#2c3e50", linestyle="--", linewidth=1.5, zorder=2,
                label="ADCD-selected: constant offset")
    ax1.set_xlabel("redshift $z$")
    ax1.set_ylabel(r"$\Delta(z) = f\sigma_8^{\rm obs} - f\sigma_8^{\rm GR}$")
    ax1.set_title("(a) $f\\sigma_8$ growth rate ($N=63$, Alestas+22)")
    ax1.set_xlim(-0.05, 2.05)
    ymax = max(0.05, 1.4 * max(abs(np.array(bin_means) + np.array(bin_errs))))
    ax1.set_ylim(-ymax, ymax)
    ax1.legend(loc="upper right", frameon=False)
    ax1.text(0.03, 0.05,
             r"$\sigma_{8,0}^{\rm fit}=0.7815$ ($-3.7\%$ vs Planck)"
             "\n" + r"$\Delta$BIC vs null $=+0.00$" +
             "\n" + r"$\rightarrow$ \textsc{constant\_wins}",
             transform=ax1.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdf6e3", ec="none", alpha=0.9))

    # --- Panel (b): H(z) ---
    ax2.set_facecolor("white")
    cc_c = [b[0] for b in cc_bins]
    cc_m = [b[1] for b in cc_bins]
    cc_e = [b[2] for b in cc_bins]
    ax2.errorbar(cc_c, cc_m, yerr=cc_e, fmt="s",
                 color="#c0392b", markersize=7, capsize=4, zorder=5,
                 label="Inverse-variance bin mean")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=1)
    ax2.axhline(delta_h_const, color="#2c3e50", linestyle="--",
                linewidth=1.5, zorder=2,
                label="ADCD-selected: constant offset")
    ax2.set_xlabel("redshift $z$")
    ax2.set_ylabel(r"$\Delta_H(z) = H^{\rm obs} - H^{\Lambda\rm CDM}$ [km/s/Mpc]")
    ax2.set_title("(b) $H(z)$ cosmic chronometers ($N=34$)")
    ax2.set_xlim(-0.05, 2.1)
    ymax2 = max(5.0, 1.4 * max(abs(np.array(cc_m) + np.array(cc_e))))
    ax2.set_ylim(-ymax2, ymax2)
    ax2.legend(loc="upper right", frameon=False)
    ax2.text(0.03, 0.05,
             r"$H_0^{\rm fit}=68.48,\ \Omega_{m,0}^{\rm fit}=0.296$"
             "\n" + r"$\Delta$BIC vs null $=+0.00$" +
             "\n" + r"$\rightarrow$ \textsc{constant\_wins}",
             transform=ax2.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdf6e3", ec="none", alpha=0.9))

    fig.tight_layout()
    out = "paper/figures/fig_cosmological_probes.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure: wide binary velocity boost
# ---------------------------------------------------------------------------
def fig_wide_binary():
    s_arr = np.linspace(2.0, 50.0, 400)
    x_arr = sep_to_x(s_arr)
    g_adcd = np.sqrt(nu_adcd(x_arr))
    g_simple = np.sqrt(nu_simple(x_arr))
    g_standard = np.sqrt(nu_standard(x_arr))
    g_rar = np.sqrt(nu_rar(x_arr))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(s_arr, g_simple, color="#2980b9", linewidth=2.0,
            label="Simple MOND ($c=4.0$)")
    ax.plot(s_arr, g_rar, color="#8e44ad", linewidth=2.0, linestyle="-.",
            label="RAR (McGaugh)")
    ax.plot(s_arr, g_standard, color="#27ae60", linewidth=2.0, linestyle=":",
            label="Standard MOND")
    ax.plot(s_arr, g_adcd, color="#c0392b", linewidth=2.5,
            label=r"ADCD ($c\approx 0.27$, no refit)")

    # Chae 2023 observed band
    ax.axhline(1.20, color="black", linewidth=1.0, alpha=0.6)
    ax.fill_between(s_arr, 1.14, 1.26, color="gray", alpha=0.18,
                    label="Chae 2023: $1.20 \\pm 0.06$")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(48, 1.005, "Newtonian", ha="right", va="bottom", fontsize=8.5,
            color="gray")

    # Mark anomaly zone
    ax.axvspan(7.0, 30.0, color="#f39c12", alpha=0.07)
    ax.text(18.5, 0.45, "Hernandez / Chae\nanomaly zone",
            ha="center", va="bottom", fontsize=8.5, color="#b9770e",
            style="italic")

    ax.set_xlabel("projected separation $s$ (kAU)")
    ax.set_ylabel(r"velocity boost $\gamma_v = v_{\rm obs}/v_{\rm Newton} = \sqrt{\nu(x)}$")
    ax.set_title("Wide-binary predicted velocity boost "
                 "($M_{\\rm tot}=1.5\\,M_\\odot$)")
    ax.set_xlim(2, 50)
    ax.set_ylim(0.4, 2.6)
    ax.legend(loc="upper left", frameon=False)

    # Twin x-axis with acceleration
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    s_ticks = [5, 10, 20, 40]
    x_ticks = [sep_to_x(s) for s in s_ticks]
    ax2.set_xticks(s_ticks)
    ax2.set_xticklabels([f"{xt:.2f}" for xt in x_ticks])
    ax2.set_xlabel(r"$x = g_N/a_0$", fontsize=10)
    ax2.spines["top"].set_visible(True)

    fig.tight_layout()
    out = "paper/figures/fig_wide_binary.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure: structural dichotomy summary
# ---------------------------------------------------------------------------
def fig_dichotomy():
    """2-panel summary: galactic functional vs cosmological amplitude."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("white")

    # --- Panel (a): SPARC functional recovery ---
    ax1.set_facecolor("white")
    # Use stacked SPARC relation: plot ν vs x for ADCD vs canonical zero-param
    x = np.logspace(-2.5, 2.0, 400)
    adcd = nu_adcd(x)
    simple = nu_simple(x)
    standard = nu_standard(x)
    rar = nu_rar(x)
    ax1.loglog(x, adcd, color="#c0392b", linewidth=2.5,
               label=r"ADCD ($c\approx 0.27$)")
    ax1.loglog(x, simple, color="#2980b9", linewidth=1.5, linestyle="--",
               label="Simple MOND")
    ax1.loglog(x, rar, color="#8e44ad", linewidth=1.5, linestyle="-.",
               label="RAR (McGaugh)")
    ax1.loglog(x, standard, color="#27ae60", linewidth=1.5, linestyle=":",
               label="Standard MOND")
    ax1.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    ax1.axvline(1.0, color="gray", linewidth=0.8, alpha=0.3)
    ax1.fill_between([1, 1e2], [1, 1], [1, 1], color="green", alpha=0.04)
    ax1.text(8, 1.15, "Newtonian\nregime", fontsize=8, color="gray")
    ax1.text(0.004, 6, "deep-MOND\nregime", fontsize=8, color="gray")
    ax1.set_xlabel(r"$x = g_{\rm bar}/a_0$")
    ax1.set_ylabel(r"$\nu(x) = g_{\rm obs}/g_{\rm bar}$")
    ax1.set_title("(a) Galactic: SPARC — functional recovery")
    ax1.set_xlim(1e-2, 1e2)
    ax1.set_ylim(0.7, 20)
    ax1.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax1.text(0.03, 0.96,
             "Functional correction recovered\n"
             r"41\% NMSE reduction over canonical" + "\n" +
             "parity with 2-param expert forms",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="none",
                       alpha=0.9))

    # --- Panel (b): cosmological amplitude-only ---
    ax2.set_facecolor("white")
    # Show fσ8 observed vs GR baseline as a function of z, demonstrating
    # that the residual is just an amplitude offset
    growth = json.load(open("results/growth_rate_discovery.json"))
    bins = growth["bin_means"]
    bin_centers = [0.5 * (b["z_lo"] + b["z_hi"]) for b in bins]
    bin_means = [b["weighted_mean_Delta"] for b in bins]
    bin_errs = [b["error"] for b in bins]
    ax2.errorbar(bin_centers, bin_means, yerr=bin_errs, fmt="o",
                 color="#c0392b", markersize=8, capsize=4, zorder=5,
                 label=r"observed $\Delta(z)$ (bin mean)")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    const = growth["candidates"][0]["params"]["theta_0"]
    ax2.axhline(const, color="#2c3e50", linestyle="--", linewidth=2.0,
                label=r"ADCD: constant offset $\theta_0$")
    # Overlay best non-constant candidate (Rational) for comparison
    rat = next(c for c in growth["candidates"] if c["name"] == "Rational")
    zz = np.linspace(0.0, 2.0, 200)
    A, z0 = rat["params"]["theta_0"], rat["params"]["theta_1"]
    # avoid divide-by-zero if z0 ~ 0
    if abs(z0) > 1e-6:
        rat_pred = A / (1.0 + (zz / z0) ** 2)
    else:
        rat_pred = np.full_like(zz, A)
    ax2.plot(zz, rat_pred, color="#2980b9", linewidth=1.2, alpha=0.7,
             linestyle="-.", label="Best z-dep. candidate (Rational)")
    ax2.set_xlabel("redshift $z$")
    ax2.set_ylabel(r"$\Delta(z) = f\sigma_8^{\rm obs} - f\sigma_8^{\rm GR}$")
    ax2.set_title("(b) Cosmological: $f\\sigma_8$ — amplitude only")
    ax2.set_xlim(-0.05, 2.05)
    ymax = max(0.05, 1.5 * max(abs(np.array(bin_means) + np.array(bin_errs))))
    ax2.set_ylim(-ymax, ymax)
    ax2.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax2.text(0.03, 0.05,
             "No functional correction detected\n"
             r"$\Delta$BIC vs null $=+0.00$" + "\n" +
             r"$\rightarrow$ \textsc{constant\_wins}",
             transform=ax2.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="none",
                       alpha=0.9))

    fig.tight_layout()
    out = "paper/figures/fig_dichotomy.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("Generating extended figures for ADCD paper...")
    fig_cosmological_probes()
    fig_wide_binary()
    fig_dichotomy()
    print("Done.")
