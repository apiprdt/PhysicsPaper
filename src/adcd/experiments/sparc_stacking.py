from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adcd.experiments.mond_comparison import (
    print_mond_comparison,
    score_mond_models,
    scores_to_dict,
    nu_simple_mond,
    nu_standard_mond,
    nu_rar,
)
from adcd.experiments.proposers import mond_correction_proposer
from adcd.experiments.sparc_data import (
    G_DAGGER,
    SparcStackResult,
    load_sparc_stack,
    parse_sparc_mrt,
    stack_sparc_galaxies,
)
from adcd.iadcd_orchestrator import iADCDOrchestrator


def _nmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_obs - y_pred) ** 2) / max(np.var(y_obs), 1e-15))


@dataclass
class SparcDiscoveryResult:
    data_source: str
    n_galaxies: int
    n_points: int
    discovered_expr: str
    final_nmse: float
    g_dagger: float
    simple_mond_reference: str
    mond_comparison: List[Dict[str, Any]] = field(default_factory=list)
    identifiability: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    bootstrap_ci: Optional[Dict[str, Any]] = None


SIMPLE_MOND_NU = "(1 + sqrt(1 + 4/x)) / 2"


def run_sparc_discovery(
    cache_path: str = "data/sparc/MassModels_Lelli2016c.mrt",
    seed: int = 42,
    verbose: bool = True,
) -> SparcDiscoveryResult:
    stack: SparcStackResult = load_sparc_stack(cache_path=cache_path)

    X = {"x": stack.x}
    y_obs = stack.nu_obs
    y_classical = stack.nu_classical

    orchestrator = iADCDOrchestrator(
        max_rounds=1,
        convergence_nmse=1e-3,
        min_snr=0.5,
        verbose=verbose,
    )

    if verbose:
        print("\nRunning iADCD on stacked (x, nu) dataset...")
    res = orchestrator.run_iterative_discovery(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable="x",
        limit_direction="oo",
        classical_expr="1.0",
        variables_with_units={"x": "dimensionless"},
        round_proposers=[mond_correction_proposer(variable="x", seed=seed)],
        seed=seed,
    )

    adcd_nmse = float(res.final_nmse_full)
    n_params = len(res.rounds[0].discovered_theta) if res.rounds else 2
    mond_scores = score_mond_models(stack.x, stack.nu_obs, adcd_nmse=adcd_nmse, adcd_n_params=n_params)

    ident: Optional[Dict[str, Any]] = None
    if res.rounds and res.rounds[0].adcd_result.search_result.identifiability_report:
        rep = res.rounds[0].adcd_result.search_result.identifiability_report
        ident = {
            "is_identifiable": rep.is_identifiable,
            "failure_mode": rep.failure_mode,
            "snr": rep.snr,
            "weight_ratio": rep.weight_ratio,
            "summary": rep.summary,
        }

    # Cross-validation & Bootstrap & Plot (only if REAL data)
    cv_res = None
    boot_res = None
    if stack.data_source == "REAL":
        try:
            print("\nRunning galaxy-level Cross-Validation (10 repeats)...")
            df = parse_sparc_mrt(cache_path)
            cv_res = run_galaxy_cv(df, res.final_expr, n_repeats=10, seed=seed)
            print("Cross-Validation NMSE summary:")
            for m, s in cv_res.items():
                print(f"  {m:<18}: {s['mean_nmse']:.5f} ± {s['std_error']:.5f}")
                
            print("\nRunning galaxy-level Bootstrapping (50 repeats)...")
            boot_res = run_galaxy_bootstrap(df, res.final_expr, n_bootstraps=50, seed=seed)
            print("Bootstrap Parameter Estimates (95% CI):")
            for p, s in boot_res.items():
                print(f"  {p:<8}: {s['mean']:.4f} ± {s['std']:.4f} (95% CI: [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}])")
                
            print("\nGenerating publication-quality plots...")
            # Extract discovered theta
            theta_symbols = sorted(
                list(res.rounds[0].discovered_theta.keys()),
                key=lambda s: int(s.split("_")[-1])
            )
            theta_vals = [res.rounds[0].discovered_theta[k] for k in theta_symbols]
            plot_sparc_results(
                stack.x, stack.nu_obs, res.final_expr, theta_vals, theta_symbols,
                n_galaxies=stack.n_galaxies,
                output_path="results/sparc_discovery_plot.png"
            )
        except Exception as e:
            print(f"Error during CV/bootstrap/plotting: {e}")
            import traceback
            traceback.print_exc()

    result = SparcDiscoveryResult(
        data_source=stack.data_source,
        n_galaxies=stack.n_galaxies,
        n_points=stack.n_points,
        discovered_expr=res.final_expr,
        final_nmse=adcd_nmse,
        g_dagger=G_DAGGER,
        simple_mond_reference=SIMPLE_MOND_NU,
        mond_comparison=scores_to_dict(mond_scores),
        identifiability=ident,
        cross_validation=cv_res,
        bootstrap_ci=boot_res,
    )

    print("\n=== SPARC MOND Discovery Results ===")
    print(f"Data source:     {result.data_source}")
    print(f"Galaxies/points: {result.n_galaxies} / {result.n_points}")
    print(f"Discovered nu(x): {result.discovered_expr}")
    print(f"ADCD NMSE:       {result.final_nmse:.5f}")
    print_mond_comparison(mond_scores)
    if ident:
        print(f"\nIdentifiability: {ident['summary']}")
    if result.data_source == "SIMULATED":
        print("\nWARNING: Simulated benchmark — do NOT claim real SPARC discovery.")

    return result


def run_galaxy_cv(
    df: pd.DataFrame,
    discovered_expr: str,
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform repeated 50/50 train/test splits on galaxy level."""
    from adcd.jax_optimizer import JAXOptimizer, jnp
    
    galaxies = df["galaxy"].unique()
    n_gals = len(galaxies)
    rng = np.random.default_rng(seed)
    
    optimizer = JAXOptimizer(n_restarts=5)
    
    models = ["Simple MOND", "Standard MOND", "RAR (McGaugh)", "ADCD discovered"]
    nmse_history = {m: [] for m in models}
    
    # Pre-parse expr
    expr, theta_symbols = optimizer._parse_expression(discovered_expr, ["x"])
    jax_fn = optimizer._build_jax_fn(expr, theta_symbols, ["x"])
    
    for run in range(n_repeats):
        rng.shuffle(galaxies)
        split_idx = n_gals // 2
        train_gals = galaxies[:split_idx]
        test_gals = galaxies[split_idx:]
        
        df_train = df[df["galaxy"].isin(train_gals)]
        df_test = df[df["galaxy"].isin(test_gals)]
        
        x_tr, nu_tr, _, _, _ = stack_sparc_galaxies(df_train)
        x_te, nu_te, _, _, _ = stack_sparc_galaxies(df_test)
        
        # Benchmarks
        nmse_history["Simple MOND"].append(_nmse(nu_te, nu_simple_mond(x_te)))
        nmse_history["Standard MOND"].append(_nmse(nu_te, nu_standard_mond(x_te)))
        nmse_history["RAR (McGaugh)"].append(_nmse(nu_te, nu_rar(x_te)))
        
        # Fit on train
        opt_res = optimizer.optimize(
            expr_str=discovered_expr,
            X={"x": x_tr},
            y_obs=nu_tr,
            data_vars=["x"],
            loss_mode="residual"
        )
        if opt_res.error is None:
            theta_vals = jnp.array([opt_res.theta[str(s)] for s in theta_symbols])
            pred_adcd = np.array(jax_fn(theta_vals, {"x": jnp.array(x_te)}))
            nmse_history["ADCD discovered"].append(_nmse(nu_te, pred_adcd))
        else:
            nmse_history["ADCD discovered"].append(float("inf"))
            
    summary = {}
    for model in models:
        vals = [v for v in nmse_history[model] if np.isfinite(v)]
        summary[model] = {
            "mean_nmse": float(np.mean(vals)) if vals else float("inf"),
            "std_error": float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        }
    return summary


def run_galaxy_bootstrap(
    df: pd.DataFrame,
    discovered_expr: str,
    n_bootstraps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap galaxies to estimate parameter uncertainty (95% CI)."""
    from adcd.jax_optimizer import JAXOptimizer
    
    galaxies = df["galaxy"].unique()
    n_gals = len(galaxies)
    rng = np.random.default_rng(seed)
    
    optimizer = JAXOptimizer(n_restarts=5)
    expr, theta_symbols = optimizer._parse_expression(discovered_expr, ["x"])
    param_names = [str(s) for s in theta_symbols]
    
    bootstrapped_params = {p: [] for p in param_names}
    
    for i in range(n_bootstraps):
        boot_gals = rng.choice(galaxies, size=n_gals, replace=True)
        # Create bootstrapped dataframe by concatenating
        df_boot = pd.concat([df[df["galaxy"] == g] for g in boot_gals], ignore_index=True)
        
        x_b, nu_b, _, _, _ = stack_sparc_galaxies(df_boot)
        
        opt_res = optimizer.optimize(
            expr_str=discovered_expr,
            X={"x": x_b},
            y_obs=nu_b,
            data_vars=["x"],
            loss_mode="residual"
        )
        if opt_res.error is None:
            for p in param_names:
                bootstrapped_params[p].append(opt_res.theta[p])
                
    summary = {}
    for p in param_names:
        vals = bootstrapped_params[p]
        if len(vals) > 0:
            summary[p] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lower": float(np.percentile(vals, 2.5)),
                "ci_upper": float(np.percentile(vals, 97.5)),
            }
        else:
            summary[p] = {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    return summary


def plot_sparc_results(
    x: np.ndarray,
    nu_obs: np.ndarray,
    discovered_expr: str,
    theta_vals: List[float],
    theta_symbols: List[Any],
    n_galaxies: int = 0,
    output_path: str = "results/sparc_discovery_plot.png",
) -> None:
    """Generate publication-quality RAR plot (PRL/MNRAS style)."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
    })
    from adcd.jax_optimizer import JAXOptimizer, jnp

    # ── Evaluate all models ───────────────────────────────────────────────────
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    optimizer = JAXOptimizer()
    expr_sym, parsed_syms = optimizer._parse_expression(discovered_expr, ["x"])
    jax_fn = optimizer._build_jax_fn(expr_sym, parsed_syms, ["x"])
    theta_jnp = jnp.array(theta_vals)

    pred_adcd_s = np.array(jax_fn(theta_jnp, {"x": jnp.array(x_sorted)}))
    pred_adcd   = np.array(jax_fn(theta_jnp, {"x": jnp.array(x)}))
    pred_sm_s   = nu_simple_mond(x_sorted)
    pred_std_s  = nu_standard_mond(x_sorted)
    pred_rar_s  = nu_rar(x_sorted)

    # ── Residuals & metrics ───────────────────────────────────────────────────
    res_adcd = nu_obs - pred_adcd
    res_sm   = nu_obs - nu_simple_mond(x)
    res_rar  = nu_obs - nu_rar(x)

    nmse_adcd = _nmse(nu_obs, pred_adcd)
    nmse_sm   = _nmse(nu_obs, nu_simple_mond(x))
    nmse_rar  = _nmse(nu_obs, nu_rar(x))

    # ── Legend label (LaTeX + fitted values) ──────────────────────────────────
    t0, t1 = float(theta_vals[0]), float(theta_vals[1])
    adcd_label = (
        r"ADCD: $\nu=\theta_0(\sqrt{1+\theta_1/x}-1)+1$"
        "\n"
        rf"$\theta_0={t0:.3f},\ \theta_1={t1:.3f}$  (2 free params)"
    )

    # ── Color palette (colorblind-friendly) ───────────────────────────────────
    c_data = "#888888"
    c_sm   = "#2166AC"
    c_std  = "#4DAC26"
    c_rar  = "#762A83"
    c_adcd = "#D6604D"
    nu_ymax = 7.5

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
    )

    # ── Upper panel ───────────────────────────────────────────────────────────
    mask_in  = nu_obs <= nu_ymax
    mask_out = ~mask_in
    n_gal_label = n_galaxies if n_galaxies > 0 else "?"

    ax1.scatter(
        x[mask_in], nu_obs[mask_in],
        s=2.5, alpha=0.12, color=c_data, rasterized=True, zorder=1,
        label=rf"SPARC obs. ($N_{{\rm gal}}={n_gal_label}$, $N_{{\rm pts}}={len(x)}$)",
    )
    if mask_out.sum() > 0:
        ax1.scatter(
            x[mask_out], np.full(mask_out.sum(), nu_ymax * 0.97),
            marker="^", s=8, alpha=0.30, color=c_data, rasterized=True, zorder=1,
            label=rf"$\nu>{nu_ymax:.0f}$ (clipped, $N={mask_out.sum()}$)",
        )

    ax1.plot(x_sorted, pred_sm_s,  color=c_sm,  ls="--", lw=1.7, zorder=3,
             label="Simple MOND (0 free params)")
    ax1.plot(x_sorted, pred_std_s, color=c_std, ls=":",  lw=1.7, zorder=3,
             label="Standard MOND (0 free params)")
    ax1.plot(x_sorted, pred_rar_s, color=c_rar, ls="-.", lw=1.7, zorder=3,
             label="RAR McGaugh+16 (0 free params)")
    ax1.plot(
        x_sorted, np.clip(pred_adcd_s, None, nu_ymax + 0.3),
        color=c_adcd, ls="-", lw=2.4, zorder=5, label=adcd_label,
    )

    ax1.set_xscale("log")
    ax1.set_ylim(0.0, nu_ymax)
    ax1.set_ylabel(r"$\nu(x) \equiv (V_{\rm obs}/V_{\rm bar})^2$")
    ax1.legend(loc="upper right", framealpha=0.92, edgecolor="0.75",
               handlelength=2.2, labelspacing=0.35)
    ax1.set_title(
        "SPARC Stacked Radial Acceleration Relation — Interpolating Function Comparison",
        pad=6, fontsize=10.5,
    )
    ax1.grid(which="major", ls="--", lw=0.45, color="0.80")
    ax1.grid(which="minor", ls=":",  lw=0.28, color="0.90")
    ax1.text(
        0.015, 0.04,
        r"$x \equiv g_{\rm bar}/a_0,\quad a_0 = 1.2\times10^{-10}\ {\rm m\,s}^{-2}$",
        transform=ax1.transAxes, fontsize=8.5, color="0.40", va="bottom",
    )

    # ── Lower panel: residuals (multi-model) ──────────────────────────────────
    ax2.axhline(0, color="0.20", lw=1.0, zorder=4)
    ax2.scatter(x, res_sm,   s=2,   alpha=0.07, color=c_sm,   rasterized=True, zorder=2,
                label=f"Simple MOND  (NMSE={nmse_sm:.3f})")
    ax2.scatter(x, res_rar,  s=2,   alpha=0.07, color=c_rar,  rasterized=True, zorder=2,
                label=f"RAR McGaugh+16  (NMSE={nmse_rar:.3f})")
    ax2.scatter(x, res_adcd, s=2.5, alpha=0.14, color=c_adcd, rasterized=True, zorder=3,
                label=f"ADCD discovered  (NMSE={nmse_adcd:.3f})")

    ax2.set_xscale("log")
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_xlabel(r"$x \equiv g_{\rm bar}/a_0$")
    ax2.set_ylabel(r"$\nu_{\rm obs} - \nu_{\rm model}$")
    ax2.legend(loc="upper right", framealpha=0.92, edgecolor="0.75",
               handlelength=2.2, labelspacing=0.35)
    ax2.grid(which="major", ls="--", lw=0.45, color="0.80")
    ax2.grid(which="minor", ls=":",  lw=0.28, color="0.90")

    fig.align_ylabels([ax1, ax2])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def save_sparc_json(result: SparcDiscoveryResult, path: str = "results/sparc_discovery.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    if payload.get("identifiability") and payload["identifiability"].get("weight_ratio") == float("inf"):
        payload["identifiability"]["weight_ratio"] = None
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run_sparc_discovery()
    save_sparc_json(r)
