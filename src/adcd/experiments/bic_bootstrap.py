"""Cluster bootstrap on delta-BIC for the SPARC ADCD-vs-MOND comparison.

Companion to :mod:`adcd.experiments.bic_eff_analysis`. Where ``bic_eff``
applies the *closed-form* independent-unit BIC correction to the point-estimate
NMSEs, this module quantifies the **uncertainty** on the in-sample delta-BIC
via a galaxy-level (cluster) bootstrap.

Why a *cluster* bootstrap? The 3342 stacked rotation-curve points are clustered
within 171 galaxies, and within-galaxy measurements are correlated
observations of the same physical system. The independent unit is the galaxy,
so we resample *galaxies with replacement* (not points), re-stack, refit each
model, and recompute BIC at the effective scale ``N_eff = N_galaxies``. This
propagates both the fit uncertainty and the clustering correctly.

Delta-BIC convention (shared with :mod:`bic_eff_analysis`):

    delta_bic = BIC_ADCD - BIC_model  at the effective scale  (positive => ADCD loses)

Per Kass & Raftery (1995): |delta| < 2 is "not worth mentioning". The central
question this module answers is whether the bootstrap 95% CI on ``delta_bic``
brackets zero, i.e. whether the in-sample ordering is statistically
distinguishable from a tie.

References:
    Efron & Tibshirani (1993), An Introduction to the Bootstrap -- cluster
        bootstrap for correlated data.
    Kass & Raftery (1995), JASA 90(430) -- Bayes factors / BIC evidence scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adcd.experiments.bic_eff_analysis import _NMSE_FLOOR, interpret_delta_bic
from adcd.metrics import bic_score


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BootModelResult:
    """Per-model bootstrap outcome.

    Attributes:
        name: Model name (e.g. "ADCD discovered", "Simple MOND (2-param)").
        n_params: Free parameters k (used in the BIC penalty term).
        nmse_mean: Bootstrap mean of the per-resample fitted NMSE.
        nmse_ci95: 95% CI on the fitted NMSE [lo, hi].
        bic_eff_mean: Bootstrap mean of the per-resample BIC at N_eff.
        bic_eff_ci95: 95% CI on the BIC_eff [lo, hi].
        delta_eff_mean: Bootstrap mean of delta_bic_eff (BIC_ADCD - BIC_model),
            populated relative to the ADCD candidate.
        delta_eff_ci95: 95% CI on delta_bic_eff [lo, hi].
        delta_eff_zero_in_ci: True if 0 lies within delta_eff_ci95 (inconclusive).
    """

    name: str
    n_params: int
    nmse_mean: float
    nmse_ci95: Tuple[float, float]
    bic_eff_mean: float
    bic_eff_ci95: Tuple[float, float]
    delta_eff_mean: Optional[float] = None
    delta_eff_ci95: Optional[Tuple[float, float]] = None
    delta_eff_zero_in_ci: Optional[bool] = None


@dataclass
class BootSummary:
    """Aggregate bootstrap-delta-BIC summary.

    Attributes:
        n_galaxies: Independent units (N_eff for the BIC scale).
        n_points: Total stacked rotation-curve measurements in the original data.
        n_bootstrap: Number of galaxy-level resamples performed.
        n_failed_resamples: Resamples dropped (e.g. optimizer did not converge).
        seed: RNG seed used (for reproducibility).
        models: Ordered list of per-model bootstrap results (deltas populated),
            ordered by ``bic_eff_mean`` ascending (lower = better).
        adcd_key: Name of the ADCD candidate used as the delta reference.
        headline_delta_mean: Mean delta_bic_eff for the primary competitor.
        headline_delta_ci95: 95% CI on that headline delta.
        headline_zero_in_ci: Whether the CI brackets 0 (inconclusive-vs-decisive).
        reviewer_message: Ready-to-quote explanation for the rebuttal/paper.
    """

    n_galaxies: int
    n_points: int
    n_bootstrap: int
    n_failed_resamples: int
    seed: int
    models: List[BootModelResult] = field(default_factory=list)
    adcd_key: str = "ADCD discovered"
    headline_delta_mean: Optional[float] = None
    headline_delta_ci95: Optional[Tuple[float, float]] = None
    headline_zero_in_ci: Optional[bool] = None
    reviewer_message: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable representation (tuples -> lists, all floats resolved)."""
        def _ci(ci):
            return None if ci is None else [float(ci[0]), float(ci[1])]
        return {
            "n_galaxies": self.n_galaxies,
            "n_points": self.n_points,
            "n_bootstrap": self.n_bootstrap,
            "n_failed_resamples": self.n_failed_resamples,
            "seed": self.seed,
            "adcd_key": self.adcd_key,
            "models": [
                {
                    "name": m.name,
                    "n_params": m.n_params,
                    "nmse_mean": m.nmse_mean,
                    "nmse_ci95": _ci(m.nmse_ci95),
                    "bic_eff_mean": m.bic_eff_mean,
                    "bic_eff_ci95": _ci(m.bic_eff_ci95),
                    "delta_eff_mean": m.delta_eff_mean,
                    "delta_eff_ci95": _ci(m.delta_eff_ci95),
                    "delta_eff_zero_in_ci": m.delta_eff_zero_in_ci,
                    "interpretation_eff": (
                        interpret_delta_bic(m.delta_eff_mean)
                        if m.delta_eff_mean is not None else "baseline"
                    ),
                }
                for m in self.models
            ],
            "headline_delta_mean": self.headline_delta_mean,
            "headline_delta_ci95": _ci(self.headline_delta_ci95),
            "headline_zero_in_ci": self.headline_zero_in_ci,
            "reviewer_message": self.reviewer_message,
        }


# ---------------------------------------------------------------------------
# Model adapters
#
# A "model" in this module is anything callable that takes (x, nu_obs) and
# returns a fitted NMSE. We expose a tiny adapter API so the bootstrap can
# treat ADCD (JAX fit of a discovered expression) and MOND 2-param (scipy
# L-BFGS-B of a closed form) uniformly.
# ---------------------------------------------------------------------------

ModelFitFn = Callable[[np.ndarray, np.ndarray], float]
"""A fitted model: consumes stacked (x, nu_obs) and returns the fitted NMSE."""


def _fit_adcd_factory(discovered_expr: str, seed: int) -> Tuple[ModelFitFn, int]:
    """Build a ModelFitFn that fits the ADCD discovered expression via JAX.

    Returns ``(fit_fn, n_params)``.
    """
    from adcd.jax_optimizer import JAXOptimizer

    optimizer = JAXOptimizer(n_restarts=5)
    expr, theta_symbols = optimizer._parse_expression(discovered_expr, ["x"])
    n_params = len(theta_symbols)

    def fit_fn(x: np.ndarray, nu_obs: np.ndarray) -> float:
        opt_res = optimizer.optimize(
            expr_str=discovered_expr,
            X={"x": x},
            y_obs=nu_obs,
            data_vars=["x"],
            loss_mode="residual",
        )
        if opt_res.error is not None:
            return float("inf")
        # Recompute NMSE on the (possibly bootstrapped) sample consistently
        # with the rest of the pipeline.
        from adcd.experiments.sparc_stacking import _nmse
        from adcd.jax_optimizer import jnp
        jax_fn = optimizer._build_jax_fn(expr, theta_symbols, ["x"])
        theta_vals = jnp.array([opt_res.theta[str(s)] for s in theta_symbols])
        pred = np.array(jax_fn(theta_vals, {"x": jnp.array(x)}))
        return _nmse(nu_obs, pred)

    return fit_fn, n_params


def _fit_2param_factory(fn, n_restarts: int, seed: int) -> Tuple[ModelFitFn, int]:
    """Build a ModelFitFn that fits a 2-param MOND/RAR closed form via scipy.

    Reuses the same log-parameterised L-BFGS-B + multi-restart scheme as
    :func:`adcd.experiments.mond_comparison._fit_2param` /
    :func:`sparc_stacking._fit_2param_on_train`, but returns NMSE on the
    sample it was fit on (no train/test split here -- this is bootstrap,
    not cross-validation).
    """
    from scipy.optimize import minimize

    def fit_fn(x: np.ndarray, nu_obs: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        nu_obs = np.asarray(nu_obs, dtype=float)

        def objective(log_params: np.ndarray) -> float:
            theta0 = np.exp(log_params[0])
            theta1 = np.exp(log_params[1])
            pred = fn(x, theta0, theta1)
            cost = float(np.mean((nu_obs - pred) ** 2))
            return cost if np.isfinite(cost) else 1e15

        rng = np.random.default_rng(seed)
        best_cost = np.inf
        best_params = np.array([0.0, 0.0])
        for _ in range(n_restarts):
            x0 = np.array([
                rng.uniform(np.log(0.01), np.log(100.0)),
                rng.uniform(np.log(0.1), np.log(10.0)),
            ])
            try:
                res = minimize(objective, x0, method="L-BFGS-B",
                               options={"maxiter": 3000, "ftol": 1e-15})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_params = res.x.copy()
            except Exception:
                continue

        theta0 = float(np.exp(best_params[0]))
        theta1 = float(np.exp(best_params[1]))
        pred = fn(x, theta0, theta1)
        from adcd.experiments.sparc_stacking import _nmse
        return _nmse(nu_obs, pred)

    return fit_fn, 2


# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_delta_bic(
    galaxies_data: Sequence[np.ndarray],
    models: Dict[str, Tuple[ModelFitFn, int]],
    n_galaxies: int,
    n_points: int,
    adcd_key: str,
    primary_competitor_key: str,
    n_bootstrap: int = 1000,
    seed: int = 2026,
    verbose: bool = False,
) -> BootSummary:
    """Cluster bootstrap on delta-BIC at the effective (galaxy) scale.

    Args:
        galaxies_data: Sequence of per-galaxy ``(x, nu_obs)`` arrays (each galaxy
            contributes its own stacked points). Resampling is done at this
            level so within-galaxy correlation is preserved.
        models: Mapping ``{name: (fit_fn, n_params)}`` -- one entry MUST be the
            ADCD candidate (``adcd_key``) and one the primary competitor.
        n_galaxies: N_eff -- number of independent units (galaxies). The BIC is
            evaluated at this scale to match :mod:`bic_eff_analysis`.
        n_points: Total number of stacked measurements in the original data
            (provenance only; not used in the BIC_eff formula).
        adcd_key: Key in ``models`` identifying the ADCD candidate.
        primary_competitor_key: Key whose delta drives the headline CI / message.
        n_bootstrap: Number of galaxy-level resamples.
        seed: RNG seed.
        verbose: Print per-resample progress to stdout.

    Returns:
        A :class:`BootSummary` with per-model BIC_eff CIs and pairwise
        delta-BIC_eff CIs relative to the ADCD candidate.

    Raises:
        ValueError: If ``adcd_key`` / ``primary_competitor_key`` absent, or
            sample sizes invalid.
    """
    if adcd_key not in models:
        raise ValueError(f"ADCD key {adcd_key!r} not in models.")
    if primary_competitor_key not in models:
        raise ValueError(
            f"Primary competitor key {primary_competitor_key!r} not in models."
        )
    if n_galaxies <= 1:
        raise ValueError("n_galaxies must be > 1.")
    if len(galaxies_data) == 0:
        raise ValueError("galaxies_data is empty.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be >= 2 for a CI.")

    rng = np.random.default_rng(seed)
    n_gal = len(galaxies_data)

    # Per-model per-resample NMSE histories.
    nmse_hist: Dict[str, List[float]] = {name: [] for name in models}
    n_failed = 0

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_gal, size=n_gal)
        x_boot = np.concatenate([galaxies_data[i][0] for i in idx])
        nu_boot = np.concatenate([galaxies_data[i][1] for i in idx])

        resample_ok = True
        for name, (fit_fn, _k) in models.items():
            try:
                nmse = float(fit_fn(x_boot, nu_boot))
            except Exception:
                nmse = float("inf")
            if not np.isfinite(nmse):
                resample_ok = False
            nmse_hist[name].append(nmse)

        if not resample_ok:
            n_failed += 1
        if verbose and (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"  [bootstrap] {b + 1}/{n_bootstrap} resamples done "
                  f"({n_failed} failed)")

    # Drop failed resamples (any model inf => whole resample unusable).
    n_per = len(next(iter(nmse_hist.values())))
    keep = [
        i for i in range(n_per)
        if all(np.isfinite(nmse_hist[name][i]) for name in models)
    ]
    if len(keep) < 2:
        raise RuntimeError(
            f"Only {len(keep)} usable bootstrap resamples (of {n_bootstrap}); "
            "cannot compute a CI. Check optimizer convergence."
        )

    # Per-model BIC_eff at each kept resample, then CIs.
    results: Dict[str, BootModelResult] = {}
    bic_eff_samples: Dict[str, np.ndarray] = {}
    for name, (_fit_fn, k) in models.items():
        nmses = np.array([nmse_hist[name][i] for i in keep])
        bics = np.array([
            bic_score(max(float(v), _NMSE_FLOOR), k, n_galaxies) for v in nmses
        ])
        bic_eff_samples[name] = bics
        results[name] = BootModelResult(
            name=name,
            n_params=k,
            nmse_mean=float(np.mean(nmses)),
            nmse_ci95=(float(np.quantile(nmses, 0.025)),
                       float(np.quantile(nmses, 0.975))),
            bic_eff_mean=float(np.mean(bics)),
            bic_eff_ci95=(float(np.quantile(bics, 0.025)),
                          float(np.quantile(bics, 0.975))),
        )

    # Pairwise deltas vs ADCD at the effective scale.
    adcd_bics = bic_eff_samples[adcd_key]
    for name in models:
        if name == adcd_key:
            continue
        deltas = adcd_bics - bic_eff_samples[name]
        ci = (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))
        results[name].delta_eff_mean = float(np.mean(deltas))
        results[name].delta_eff_ci95 = ci
        results[name].delta_eff_zero_in_ci = bool(ci[0] <= 0.0 <= ci[1])

    ordered = sorted(results.values(), key=lambda r: r.bic_eff_mean)

    comp = results[primary_competitor_key]
    headline_zero = (
        None if comp.delta_eff_ci95 is None
        else bool(comp.delta_eff_ci95[0] <= 0.0 <= comp.delta_eff_ci95[1])
    )
    message = _build_reviewer_message(
        competitor=comp,
        n_bootstrap=n_bootstrap,
        n_failed=n_failed,
    )

    return BootSummary(
        n_galaxies=n_galaxies,
        n_points=n_points,
        n_bootstrap=n_bootstrap,
        n_failed_resamples=n_failed,
        seed=seed,
        models=ordered,
        adcd_key=adcd_key,
        headline_delta_mean=comp.delta_eff_mean,
        headline_delta_ci95=comp.delta_eff_ci95,
        headline_zero_in_ci=headline_zero,
        reviewer_message=message,
    )


def _build_reviewer_message(
    competitor: BootModelResult,
    n_bootstrap: int,
    n_failed: int,
) -> str:
    """Assemble the ready-to-quote rebuttal sentence (bic_eff style)."""
    if competitor.delta_eff_ci95 is None:
        return ""
    lo, hi = competitor.delta_eff_ci95
    interp = interpret_delta_bic(competitor.delta_eff_mean)
    bracket = (
        "brackets zero (statistically inconclusive)"
        if competitor.delta_eff_zero_in_ci
        else "excludes zero (statistically decisive)"
    )
    return (
        f"Cluster bootstrap on galaxies ({n_bootstrap - n_failed} usable "
        f"resamples of {n_bootstrap}): delta-BIC_eff vs {competitor.name} = "
        f"{competitor.delta_eff_mean:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] -- "
        f"{bracket}. Point estimate interpretation: {interp}."
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_bootstrap_report(summary: BootSummary) -> None:
    """Pretty-print the bootstrap table (mirrors bic_eff_report style)."""
    print("\n" + "=" * 74)
    print("Cluster Bootstrap on delta-BIC_eff (galaxy-level resampling)")
    print("=" * 74)
    print(
        f"N_galaxies={summary.n_galaxies}, N_points={summary.n_points}, "
        f"n_bootstrap={summary.n_bootstrap} "
        f"({summary.n_failed_resamples} failed), seed={summary.seed}"
    )
    print("-" * 74)
    hdr = (
        f"{'Model':<26} {'NMSE':>8} {'BIC_eff':>9} "
        f"{'dBIC_eff':>10} {'95% CI':>22}"
    )
    print(hdr)
    print("-" * 74)
    for m in summary.models:
        nm = f"{m.nmse_mean:>8.4f}"
        be = f"{m.bic_eff_mean:>9.2f}"
        if m.name == summary.adcd_key:
            dm = "(baseline)"
            ci = "—"
        else:
            dm = f"{m.delta_eff_mean:+.2f}"
            lo, hi = m.delta_eff_ci95
            flag = " *0*" if m.delta_eff_zero_in_ci else ""
            ci = f"[{lo:+.2f}, {hi:+.2f}]{flag}"
        print(f"{m.name:<26} {nm} {be} {dm:>10} {ci:>22}")
    print("-" * 74)
    print(f"Reviewer message: {summary.reviewer_message}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# High-level driver: bootstrap straight from the SPARC cache
# ---------------------------------------------------------------------------

def run_bootstrap_from_cache(
    cache_path: str = "data/sparc/MassModels_Lelli2016c.mrt",
    discovered_expr: Optional[str] = None,
    results_path: str = "results/sparc_discovery.json",
    output_path: str = "results/bic_bootstrap_delta.json",
    competitor_keys: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 2026,
    n_restarts_2param: int = 10,
    verbose: bool = True,
) -> BootSummary:
    """End-to-end driver: stack SPARC galaxies, build models, bootstrap, write JSON.

    Args:
        cache_path: Path to the SPARC MRT cache.
        discovered_expr: ADCD expression string. If None, read from
            ``results_path`` (the ``discovered_expr`` field of the discovery JSON).
        results_path: Where to find ``discovered_expr`` if not given explicitly.
        output_path: Where to write the summary JSON.
        competitor_keys: 2-param baselines to include. Defaults to Simple MOND
            2-param (headline) + RAR 2-param + Standard MOND 2-param.
        n_bootstrap: Number of galaxy-level resamples.
        seed: RNG seed.
        n_restarts_2param: L-BFGS-B restarts for each 2-param fit per resample.
        verbose: Print progress.
    """
    from adcd.experiments.mond_comparison import (
        nu_simple_mond_2param, nu_standard_mond_2param, nu_rar_2param,
    )
    from adcd.experiments.sparc_data import parse_sparc_mrt, stack_sparc_galaxies
    from adcd.experiments.sparc_stacking import _nmse  # noqa: F401  (consistency ref)

    if discovered_expr is None:
        with open(results_path, encoding="utf-8") as fh:
            discovered_expr = json.load(fh)["discovered_expr"]
        if verbose:
            print(f"Loaded discovered_expr from {results_path}:\n  {discovered_expr}")

    if competitor_keys is None:
        competitor_keys = [
            "Simple MOND (2-param)", "RAR (McGaugh, 2-param)",
            "Standard MOND (2-param)",
        ]

    competitor_fns = {
        "Simple MOND (2-param)": nu_simple_mond_2param,
        "Standard MOND (2-param)": nu_standard_mond_2param,
        "RAR (McGaugh, 2-param)": nu_rar_2param,
    }

    if verbose:
        print("Stacking SPARC galaxies (per-galaxy arrays for cluster bootstrap)...")
    df = parse_sparc_mrt(cache_path)
    galaxies_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, grp in df.groupby("galaxy"):
        x, nu, _, _, _ = stack_sparc_galaxies(grp)
        # stack_sparc_galaxies returns 5-tuple; if n_galaxies==0 for a lone
        # group it returns empty arrays -- skip those.
        if len(x) >= 3:
            galaxies_data.append((x, nu))
    n_galaxies = len(galaxies_data)
    n_points = sum(len(x) for x, _ in galaxies_data)
    if verbose:
        print(f"  {n_galaxies} galaxies, {n_points} points total.")

    # Build the model table.
    models: Dict[str, Tuple[ModelFitFn, int]] = {}
    adcd_fn, adcd_k = _fit_adcd_factory(discovered_expr, seed=seed)
    models["ADCD discovered"] = (adcd_fn, adcd_k)
    for key in competitor_keys:
        fn = competitor_fns[key]
        models[key] = _fit_2param_factory(fn, n_restarts=n_restarts_2param, seed=seed)

    summary = bootstrap_delta_bic(
        galaxies_data=galaxies_data,
        models=models,
        n_galaxies=n_galaxies,
        n_points=n_points,
        adcd_key="ADCD discovered",
        primary_competitor_key="Simple MOND (2-param)",
        n_bootstrap=n_bootstrap,
        seed=seed,
        verbose=verbose,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)
    if verbose:
        print_bootstrap_report(summary)
        print(f"\nWrote {output_path}")

    return summary


if __name__ == "__main__":
    # Standalone usage:
    #   python -m adcd.experiments.bic_bootstrap [n_bootstrap]
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_bootstrap_from_cache(n_bootstrap=n)
