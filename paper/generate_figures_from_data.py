"""
generate_figures_from_data.py
==============================
Generates fig1_recovery.pdf and fig2_bic.pdf for the ADCD paper.

Sources:
  Fig 1 : run_outputs/pysr_comparison_noiseless_20seeds.json
          seed=42, noise=0.0  (guided ADCD, has theta_fit)
          Domain limits from DEFAULT_CLEAN_DOMAINS (verified from code):
            Time Dilation     : v  <= 0.3c
            Screened Coulomb  : r  <= 4.0
            Entropy Expansion : dV/V_i <= 3.0   ← CRITICAL (not 1.0)

  Fig 2 : run_outputs/adcd_v3_blind_validation_report.json
          BIC values (seed=42, noise=1%):
            TD  : BIC=-160.88, ΔBIC=-0.74  → WITHHELD
            SC  : BIC=-1617.66, ΔBIC=25.74 → IDENTIFIABLE
            EE  : BIC=-811.34, ΔBIC=37.99  → IDENTIFIABLE

No numbers are hardcoded in the plotting code itself;
all numerical values are read from the JSON files above.
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Make sure project src is importable ──────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from adcd.anomaly_scenarios import get_all_scenarios

# ── NeurIPS-compatible font settings ─────────────────────────────────────────
rcParams.update({
    'font.family':       'serif',
    'mathtext.fontset':  'cm',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.linewidth':    0.5,
    'grid.color':        '#cbd5e1',
    'grid.linestyle':    '--',
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'figure.dpi':        150,
})

ADCD_COLOR  = '#083c7d'
TRUTH_COLOR = '#1e40af'
OBS_COLOR   = '#94a3b8'
ABL_COLOR   = '#94a3b8'
RED_THRESH  = '#dc2626'

# ── Verified domain limits (from eval/benchmark_pysr_comparison.py) ───────────
DOMAIN_LIMITS = {
    'Time Dilation':     0.3,   # v <= 0.3c
    'Screened Coulomb':  4.0,   # r <= 4.0
    'Entropy Expansion': 3.0,   # dV/V_i <= 3.0  (NOT 1.0)
}

SCENARIO_NAMES = ['Time Dilation', 'Screened Coulomb', 'Entropy Expansion']

X_LABELS = [
    r"Time Dilation" + "\n" + r"(Einstein, $v{\leq}0.3c$)",
    r"Screened Coulomb" + "\n" + r"(Debye, $r{\leq}4.0$)",
    r"Entropy Expansion" + "\n" + r"(Carnot, $dV/V_i{\leq}3$)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def verdict_style(passed: bool):
    if passed:
        return 'IDENTIFIABLE', '#16a34a'
    return 'WITHHELD', '#d97706'


def add_verdict_badge(ax, label, color, loc='upper right'):
    x  = 0.97 if 'right' in loc else 0.03
    ha = 'right' if 'right' in loc else 'left'
    ax.text(x, 0.97, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color='white',
            ha=ha, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                      edgecolor='none', alpha=0.92))


def get_x_axis(scenario_name, X):
    """Return (x_values, x_axis_label) for a scenario."""
    if scenario_name == 'Time Dilation':
        return X['v'], r'$v/c$'
    elif scenario_name == 'Screened Coulomb':
        return X['r'], r'$r$'
    else:  # Entropy Expansion
        return X['dV'] / X['V_i'], r'$dV/V_i$'


def evaluate_prediction(expr_str, theta_fit, X, scenario, y_classical, delta_true):
    """
    Numerically evaluate the ADCD candidate expression with fitted theta.
    Falls back to zeros if substitution fails.
    """
    import sympy as sp

    try:
        expr = sp.sympify(expr_str)
        if theta_fit:
            sub = {sp.Symbol(k): v for k, v in theta_fit.items()}
            expr = expr.subs(sub)

        free = [str(s) for s in expr.free_symbols]
        subs_dict = {}
        for sym_name in free:
            if sym_name in X:
                subs_dict[sym_name] = X[sym_name]
            elif sym_name in scenario.classical_constants:
                subs_dict[sym_name] = np.full(len(delta_true),
                                               scenario.classical_constants[sym_name])

        if not subs_dict:
            val = float(expr)
            delta_pred = np.full(len(delta_true), val)
        else:
            args  = list(subs_dict.keys())
            func  = sp.lambdify([sp.Symbol(a) for a in args], expr, modules=['numpy'])
            delta_pred = func(*[subs_dict[a] for a in args])

        # Mode-detection fix: if multiplicative scenario but y_classical has zero
        # variance (EE), the optimizer fitted an additive residual; convert back.
        if scenario.correction_type == 'multiplicative' and np.std(y_classical) < 1e-15:
            delta_pred = delta_pred / y_classical

        return np.asarray(delta_pred, dtype=float)

    except Exception as e:
        print(f'  [WARN] evaluate_prediction failed for {scenario.name}: {e}')
        return np.zeros_like(delta_true)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_noiseless_seed42():
    """
    Returns per-scenario dict with ADCD guided run (seed=42, noise=0.0)
    from pysr_comparison_noiseless_20seeds.json.
    """
    path = os.path.join('run_outputs', 'pysr_comparison_noiseless_20seeds.json')
    ns   = json.load(open(path))
    result = {}
    for run in ns['raw_runs']:
        if run['seed'] == 42 and run['noise'] == 0.0:
            result[run['scenario']] = run['adcd']
    return result


def load_bic_diagnostic():
    """
    Returns per-scenario BIC data from adcd_v3_blind_validation_report.json.
    This is the source of the BIC values reported in Table 6 and Fig 2.
    """
    path = os.path.join('run_outputs', 'adcd_v3_blind_validation_report.json')
    vr   = json.load(open(path))
    result = {}
    for sc in SCENARIO_NAMES:
        checks = vr[sc]['checks']
        bs     = checks['blind_search']
        abl    = checks['ablation_control']
        result[sc] = {
            'bic_correct':  abl['true_structure_bic'],
            'bic_ablated':  abl['ablated_bic'],
            'delta_bic':    abl['bic_diff'],
            'passed':       abl['pass'],
            'expr_str':     bs['top_candidate'],
            'nmse':         bs['nmse'],
            # pareto_front for annotation purposes
            'pareto':       bs.get('pareto_front', []),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Recovery curves (seed=42, zero noise)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1(noiseless_runs, bic_data, all_scenarios):
    """
    Three-panel line plot of ADCD correction recovery at zero noise (seed 42).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle('ADCD Correction Recovery (Zero Noise, Seed 42)',
                 fontsize=13, fontweight='bold', y=1.01)

    for i, name in enumerate(SCENARIO_NAMES):
        ax       = axes[i]
        scenario = next(s for s in all_scenarios if s.name == name)
        run      = noiseless_runs[name]
        bic_info = bic_data[name]

        # Generate data deterministically (seed=42, zero noise)
        X_clean, y_obs, y_cl, delta_clean = scenario.generate_data(
            seed=42, noise_level=0.0,
            domain_max=DOMAIN_LIMITS[name]
        )

        # Evaluate ADCD prediction from expr_str + theta_fit
        delta_pred = evaluate_prediction(
            run['expr_str'], run['theta_fit'],
            X_clean, scenario, y_cl, delta_clean
        )

        x_val, x_label = get_x_axis(name, X_clean)
        sort_idx = np.argsort(x_val)
        xs  = x_val[sort_idx]
        yt  = delta_clean[sort_idx]
        yp  = delta_pred[sort_idx]

        ax.plot(xs, yt, color=TRUTH_COLOR, linewidth=2.0, label='Ground truth', zorder=3)
        ax.plot(xs, yp, color='#dc2626',   linewidth=1.8,
                linestyle='--', label='ADCD Rank-1 candidate', zorder=4)

        ax.set_xlabel(x_label, fontsize=11)
        if i == 0:
            ax.set_ylabel(r'Correction $\Delta$', fontsize=11)
        ax.set_title(name, fontsize=11, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=8.5, loc='upper left')

        # Verdict badge (from BIC diagnostic data)
        v_label, v_color = verdict_style(bic_info['passed'])
        add_verdict_badge(ax, v_label, v_color, loc='upper right')

        # NMSE from noiseless run
        nmse = run['nmse_train']
        nmse_pos = {
            'Time Dilation':     (0.97, 0.05, 'right'),
            'Screened Coulomb':  (0.97, 0.55, 'right'),
            'Entropy Expansion': (0.97, 0.05, 'right'),
        }
        nx, ny, nha = nmse_pos[name]
        ax.text(nx, ny, f'NMSE={nmse:.2e}',
                transform=ax.transAxes, fontsize=8, ha=nha, va='bottom',
                color='#374151',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cbd5e1', pad=2))

    plt.tight_layout()
    fig.savefig('paper/fig1_recovery.pdf')
    fig.savefig('paper/fig1_recovery.png', dpi=150)
    plt.close(fig)
    print('Saved paper/fig1_recovery.pdf + .png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: BIC comparison and ΔBIC evidence
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2(bic_data):
    """
    Two-panel figure: (a) BIC bar chart, (b) ΔBIC bar chart with KR threshold.
    All values sourced from adcd_v3_blind_validation_report.json.
    """
    bics_correct = [bic_data[sc]['bic_correct'] for sc in SCENARIO_NAMES]
    bics_ablated = [bic_data[sc]['bic_ablated'] for sc in SCENARIO_NAMES]
    delta_bics   = [bic_data[sc]['delta_bic']   for sc in SCENARIO_NAMES]
    passeds      = [bic_data[sc]['passed']       for sc in SCENARIO_NAMES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x     = np.arange(len(SCENARIO_NAMES))
    width = 0.32

    # ── Panel (a): Absolute BIC ──────────────────────────────────────────────
    rects1 = ax1.bar(x - width / 2, bics_correct, width,
                     label='Rank-1 candidate (blind search)',
                     color='#1e3a8a', edgecolor='black', linewidth=0.5)
    rects2 = ax1.bar(x + width / 2, bics_ablated, width,
                     label='Best alternative (ablated)',
                     color=ABL_COLOR,  edgecolor='black', linewidth=0.5)

    # Value labels inside bars
    for rect in rects1:
        h = rect.get_height()
        yp = h * 0.85 if abs(h) > 500 else h * 0.65
        ax1.text(rect.get_x() + rect.get_width() / 2, yp, f'{int(round(h))}',
                 ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        yp = h * 0.85 if abs(h) > 500 else h * 0.65
        ax1.text(rect.get_x() + rect.get_width() / 2, yp, f'{int(round(h))}',
                 ha='center', va='center', fontsize=8, color='#0f172a', fontweight='bold')

    ax1.set_ylabel('BIC (lower is better)', fontsize=11, labelpad=8)
    ax1.set_title('(a) Absolute BIC Model Selection', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(X_LABELS, fontsize=9)
    ax1.axhline(y=0, color='black', linewidth=0.8, zorder=2)
    ax1.set_ylim(min(min(bics_correct), min(bics_ablated)) * 1.15, 350)
    ax1.legend(loc='lower left', frameon=True, facecolor='white',
               framealpha=0.95, edgecolor='#cbd5e1', fontsize=7.5,
               labelspacing=0.2, handlelength=1.0, handletextpad=0.4, borderpad=0.3)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Verdict badges above each group
    for i, (sc, passed) in enumerate(zip(SCENARIO_NAMES, passeds)):
        v_label, v_color = verdict_style(passed)
        ax1.text(i, 280, v_label, ha='center', va='center', fontsize=7.5,
                 fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor=v_color,
                           edgecolor='none', alpha=0.9))

    # ── Panel (b): ΔBIC ──────────────────────────────────────────────────────
    bar_colors = [verdict_style(p)[1] for p in passeds]
    bars2 = ax2.bar(x, delta_bics, width=0.45, color=bar_colors,
                    edgecolor='black', linewidth=0.5)

    ax2.set_yscale('symlog', linthresh=1)
    ax2.set_ylim(-5, 6000)
    ax2.axhline(y=10, color=RED_THRESH, linestyle='--', linewidth=1.5, zorder=3,
                label=r'Kass--Raftery threshold ($\Delta\mathrm{BIC}=10$, very strong)')
    ax2.axhline(y=0,  color='black',    linestyle='-',  linewidth=0.8, zorder=2)
    ax2.axhspan(-5, 10, color='#fef3c7', alpha=0.45, zorder=0,
                label=r'$\Delta\mathrm{BIC} < 10$ (below threshold)')

    # ΔBIC annotations
    for i, (v, sc) in enumerate(zip(delta_bics, SCENARIO_NAMES)):
        label_str = fr'$\Delta\mathrm{{BIC}}={v:.2f}$'
        offset = 8 if v >= 0 else -18
        va = 'bottom' if v >= 0 else 'top'
        ax2.annotate(label_str, xy=(i, v), xytext=(0, offset),
                     textcoords='offset points',
                     ha='center', va=va, fontsize=8.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                               edgecolor='#cbd5e1', alpha=0.95))

    ax2.set_ylabel(r'$\Delta\mathrm{BIC} = \mathrm{BIC}_\mathrm{ablated} - \mathrm{BIC}_\mathrm{correct}$',
                   fontsize=11, labelpad=8)
    ax2.set_title(r'(b) Identifiability Evidence ($\Delta\mathrm{BIC}$)',
                  fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(X_LABELS, fontsize=9)
    ax2.legend(loc='upper left', frameon=True, facecolor='white',
               framealpha=0.95, edgecolor='#cbd5e1', fontsize=8.5)
    ax2.grid(axis='y', which='both', linestyle='--', alpha=0.3)

    plt.subplots_adjust(wspace=0.30, bottom=0.18, top=0.88, left=0.07, right=0.97)
    fig.savefig('paper/fig2_bic.pdf')
    fig.savefig('paper/fig2_bic.png', dpi=150)
    plt.close(fig)
    print('Saved paper/fig2_bic.pdf + .png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...')
    noiseless_runs = load_noiseless_seed42()
    bic_data       = load_bic_diagnostic()
    all_scenarios  = get_all_scenarios()

    print('\nVerification:')
    for sc in SCENARIO_NAMES:
        bd = bic_data[sc]
        v = 'IDENTIFIABLE' if bd['passed'] else 'WITHHELD'
        print(f'  {sc}: BIC={bd["bic_correct"]:.2f}, '
              f'dBIC={bd["delta_bic"]:.2f} -> {v}')

    os.makedirs('paper', exist_ok=True)

    print('\nGenerating Figure 1 (recovery curves, zero noise, seed 42)...')
    make_fig1(noiseless_runs, bic_data, all_scenarios)

    print('\nGenerating Figure 2 (BIC comparison)...')
    make_fig2(bic_data)

    print('\nDone. Both figures saved to paper/.')


if __name__ == '__main__':
    main()
