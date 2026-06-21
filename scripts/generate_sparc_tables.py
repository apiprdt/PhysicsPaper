"""Generate LaTeX tables for the SPARC capstone section from result JSONs.

Reads:
  - results/sparc_discovery.json         (model comparison, CV, bootstrap, fitted baselines)
  - results/sparc_robustness_results.json (3 quality-cut scenarios)
  - results/bic_bootstrap_delta.json     (galaxy-level cluster bootstrap on delta-BIC_eff)

Writes:
  - paper/generated/tab_sparc_comparison.tex    (4-model NMSE/BIC/n_params)
  - paper/generated/tab_sparc_fitted.tex        (2-param fitted baseline comparison)
  - paper/generated/tab_sparc_validation.tex    (CV mean +/- SE)
  - paper/generated/tab_sparc_bootstrap.tex     (parameter 95% CI)
  - paper/generated/tab_sparc_robustness.tex    (3 quality cuts)
  - paper/generated/tab_sparc_bic_bootstrap.tex (cluster bootstrap delta-BIC_eff)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISC = json.loads((ROOT / "results" / "sparc_discovery.json").read_text())
ROBUST = json.loads((ROOT / "results" / "sparc_robustness_results.json").read_text())
BIC_BOOT = json.loads((ROOT / "results" / "bic_bootstrap_delta.json").read_text())
OUT = ROOT / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def f(x, n=3):
    return f"{x:.{n}f}"


# ---- Table 1: model comparison (sorted by NMSE ascending) -------------------
comp = sorted(DISC["mond_comparison"], key=lambda m: m["nmse"])
rows = []
for m in comp:
    name = m["name"]
    nmse = f(m["nmse"], 4)
    bic = f"{m['bic']:.2f}" if m["bic"] is not None else "---"
    nparams = m["n_params"]
    rows.append(f"  {name} & {nmse} & {bic} & {nparams} \\\\")
# ADCD discovered row should be bolded; find it and rebuild
rows = []
for m in comp:
    name = m["name"]
    is_adcd = name.lower().startswith("adcd")
    nmse = f(m["nmse"], 4)
    bic = f"{m['bic']:.2f}" if m["bic"] is not None else "---"
    nparams = m["n_params"]
    nmse_str = rf"\textbf{{{nmse}}}" if is_adcd else nmse
    rows.append(f"  {name} & {nmse_str} & {bic} & {nparams} \\\\")

tab1 = r"""\begin{table}[H]
\centering\footnotesize
\caption{SPARC stacked radial acceleration relation ($N_{\rm gal}=171$, $N_{\rm pts}=3342$).
NMSE evaluated on the stacked $(x, \nu_{\rm obs})$ dataset with
$x \equiv g_{\rm bar}/a_0$. The ADCD-discovered 2-parameter function achieves the
lowest NMSE and the lowest BIC.}
\label{tab:sparc_comparison}
\begin{tabular}{l c c c}
\toprule
\textbf{Model} & \textbf{NMSE} $\downarrow$ & \textbf{BIC} $\downarrow$ & \textbf{Free params} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_sparc_comparison.tex").write_text(tab1, encoding="utf-8")

# ---- Table 1b: 2-param fitted baselines (fair comparison) ---------------------
fitted_comp = DISC.get("fitted_baselines", [])
if fitted_comp:
    fitted_rows = []
    for m in fitted_comp:
        name = m["name"]
        is_adcd = name.lower().startswith("adcd")
        nmse = f(m["nmse"], 4)
        bic = f"{m['bic']:.2f}" if m["bic"] is not None else "---"
        ft = m.get("fitted_theta", {})
        t0_str = f(ft["theta_0"], 3) if ft else "---"
        t1_str = f(ft["theta_1"], 3) if ft else "---"
        # Highlight ADCD row with underline (not bold-lowest, since fitted baselines
        # are competitive — see honest caption below).
        if is_adcd:
            nmse_str = rf"\underline{{{nmse}}}"
            bic_str = rf"\underline{{{bic}}}"
        else:
            nmse_str = nmse
            bic_str = bic
        fitted_rows.append(
            f"  {name} & {t0_str} & {t1_str} & {nmse_str} & {bic_str} \\\\"
        )

    tab1b = r"""\begin{table}[H]
\centering\footnotesize
\caption{Fair parameter-matched comparison: all four models fitted with
exactly 2 free parameters via L-BFGS-B (20 random restarts) on the same
stacked SPARC dataset ($N_{\rm gal}=171$, $N_{\rm pts}=3342$).
Under matched parameter count, the fitted Simple-MOND and RAR forms attain
marginally lower in-sample NMSE than the ADCD-discovered form (a $\sim$1.5\%
difference, well within the bootstrap CI). Galaxy-level cross-validation
(Table~\ref{tab:sparc_cv}) shows the three are statistically
indistinguishable out-of-sample. The honest conclusion is therefore that
ADCD's correction-first search recovers an interpolating function
\emph{competitive with}—not superior to—well-known 2-param MOND forms, while
requiring no prior knowledge of the MOND/RAR functional family.}
\label{tab:sparc_fitted}
\begin{tabular}{l c c c c}
\toprule
\textbf{Model (2-param)} & $\hat\theta_0$ & $\hat\theta_1$ &
\textbf{NMSE} $\downarrow$ & \textbf{BIC} $\downarrow$ \\
\midrule
""" + "\n".join(fitted_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab_sparc_fitted.tex").write_text(tab1b, encoding="utf-8")
else:
    print("  (skipping tab_sparc_fitted.tex — no fitted_baselines in JSON)")

# ---- Table 2: cross-validation ---------------------------------------------
cv = DISC["cross_validation"]
order = [
    "Simple MOND", "Standard MOND", "RAR (McGaugh)", "ADCD discovered",
    "Simple MOND (2-param)", "Standard MOND (2-param)", "RAR (McGaugh, 2-param)",
]
# Only include models that are present in the CV results
order = [m for m in order if m in cv]
cvrows = []
for name in order:
    s = cv[name]
    is_adcd = name.lower().startswith("adcd")
    mean = f(s["mean_nmse"], 4)
    se = f(s["std_error"], 4)
    # Underline (not bold) the ADCD row: fitted baselines are competitive,
    # so we mark rather than emphasize ADCD as the unique winner.
    mean_str = rf"\underline{{{mean}}}" if is_adcd else mean
    cvrows.append(f"  {name} & {mean_str} $\\pm$ {se} \\\\")

tab2 = r"""\begin{table}[H]
\centering\footnotesize
\caption{Galaxy-level cross-validation (10 repeated 50/50 train/test splits by
galaxy). The top three rows are zero-parameter canonical forms; the middle row
is the ADCD-discovered 2-param form; the bottom three rows are the same MOND/RAR
families refit with 2 free parameters (fair parameter-matched comparison).
ADCD's out-of-sample NMSE is statistically indistinguishable from the best
2-param fitted baselines (overlapping $\pm$SE intervals), confirming the
correction-first search is competitive with hand-crafted MOND forms without
requiring prior knowledge of the interpolating-function family.}
\label{tab:sparc_cv}
\begin{tabular}{l c}
\toprule
\textbf{Model} & \textbf{Test NMSE} (mean $\pm$ SE) \\
\midrule
""" + "\n".join(cvrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_sparc_validation.tex").write_text(tab2, encoding="utf-8")

# ---- Table 3: bootstrap parameter CIs --------------------------------------
# Map internal symbol names (e.g. "theta_r1_0") to display symbols theta_0, theta_1, ...
_PARAM_DISPLAY = {}
def _param_display(name: str) -> str:
    if name not in _PARAM_DISPLAY:
        # Extract trailing integer index from names like "theta_r1_0", "theta_0", etc.
        idx = "".join(ch for ch in name if ch.isdigit())[-1] if any(ch.isdigit() for ch in name) else "0"
        _PARAM_DISPLAY[name] = rf"\theta_{{{idx}}}"
    return _PARAM_DISPLAY[name]

boot = DISC["bootstrap_ci"]
# Preserve canonical order theta_0 then theta_1 (sort by trailing index)
def _param_idx(name: str) -> int:
    digits = [ch for ch in name if ch.isdigit()]
    return int(digits[-1]) if digits else 0
bootrows = []
for param in sorted(boot.keys(), key=_param_idx):
    s = boot[param]
    mean = f(s["mean"], 3)
    lo = f(s["ci_lower"], 3)
    hi = f(s["ci_upper"], 3)
    bootrows.append(f"  ${_param_display(param)}$ & {mean} & [{lo}, {hi}] \\\\")

tab3 = r"""\begin{table}[H]
\centering\footnotesize
\caption{Bootstrap parameter estimates for the ADCD-discovered interpolating
function $\nu(x) = \theta_0(\sqrt{1+\theta_1/x}-1)+1$ (50 galaxy-level
bootstrap resamples; 95\% percentile CI).}
\label{tab:sparc_bootstrap}
\begin{tabular}{l c c}
\toprule
\textbf{Parameter} & \textbf{Mean} & \textbf{95\% CI} \\
\midrule
""" + "\n".join(bootrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_sparc_bootstrap.tex").write_text(tab3, encoding="utf-8")

# ---- Table 4: robustness across quality cuts --------------------------------
robrows = []
for sc in ROBUST:
    nmse_adcd = f(sc["nmse_adcd"], 4)
    nmse_rar = f(sc["nmse_rar"], 4)
    nmse_sm = f(sc["nmse_sm"], 4)
    t0 = f(sc["theta_0"], 3)
    t1 = f(sc["theta_1"], 3)
    robrows.append(
        f"  {sc['scenario']} & {sc['galaxies']} & {sc['points']} & "
        f"{t0} & {t1} & {nmse_adcd} & {nmse_rar} & {nmse_sm} \\\\"
    )

tab4 = r"""\begin{table}[H]
\centering\footnotesize
\caption{SPARC robustness across three quality cuts. The ADCD-discovered
function consistently outperforms Simple MOND and the McGaugh+16 RAR form
across increasingly strict galaxy-quality selection.}
\label{tab:sparc_robustness}
\begin{tabularx}{\textwidth}{L c c c c c c c}
\toprule
\textbf{Quality cut} & \textbf{Gal.} & \textbf{Pts} &
$\hat\theta_0$ & $\hat\theta_1$ &
\textbf{ADCD} & \textbf{RAR} & \textbf{Simple MOND} \\
& & & & & \multicolumn{3}{c}{NMSE} \\
\cmidrule(lr){6-8}
\midrule
""" + "\n".join(robrows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
(OUT / "tab_sparc_robustness.tex").write_text(tab4, encoding="utf-8")

# ---- Table 5: cluster bootstrap on delta-BIC_eff -----------------------------
# Renders the galaxy-level cluster bootstrap (1000 resamples) on the
# effective-sample-size-scaled BIC difference at N_eff = 171 independent
# galaxies. ADCD is the implicit baseline; each competitor row reports
# delta_eff = BIC_eff(competitor) - BIC_eff(ADCD), so negative delta = ADCD wins.
adcd_key = BIC_BOOT.get("adcd_key", "ADCD discovered")
n_gal = BIC_BOOT.get("n_galaxies")
n_boot = BIC_BOOT.get("n_bootstrap")
n_failed = BIC_BOOT.get("n_failed_resamples", 0)
n_used = n_boot - n_failed

# 4-column layout: Model | BIC_eff (95% CI) | delta | 95% CI | Verdict
boot_rows = []
for m in BIC_BOOT["models"]:
    name = m["name"]
    if name == adcd_key:
        bic_eff = f(m["bic_eff_mean"], 1)
        bic_lo = f(m["bic_eff_ci95"][0], 1)
        bic_hi = f(m["bic_eff_ci95"][1], 1)
        boot_rows.append(
            rf"  \underline{{{name}}} & {bic_eff} $\pm$ [{bic_lo}, {bic_hi}]"
            r" & --- & --- & (baseline) \\"
        )
        continue
    delta = m["delta_eff_mean"]
    delta_lo, delta_hi = m["delta_eff_ci95"]
    brackets = m.get("delta_eff_zero_in_ci")
    if brackets:
        verdict = r"Inconclusive (CI brackets $0$)"
    else:
        verdict = m.get("interpretation_eff", "")
    boot_rows.append(
        rf"  {name} & --- & {delta:+.1f} & [{delta_lo:+.1f}, {delta_hi:+.1f}]"
        rf" & {verdict} \\"
    )

tab5 = r"""\begin{table}[H]
\centering\footnotesize
\caption{Galaxy-level cluster bootstrap on the effective-sample-size-scaled BIC
difference $\delta{\rm BIC}_{\rm eff}$ at $N_{\rm eff}=""" + str(n_gal) + r"""$
independent galaxies (""" + f"{n_used:,}" + r""" usable resamples of """ + f"{n_boot:,}" + r""",
seed 2026). ADCD is the baseline; $\delta{\rm BIC}_{\rm eff}<0$ means ADCD wins.
Per Kass \& Rafferty~\cite{kass1995bayes}, $|\delta|<2$ is weak,
$2$--$6$ positive, $6$--$10$ strong, $>10$ very strong---\emph{but} if the
95\% bootstrap CI brackets zero the result is formally inconclusive regardless
of the point estimate. ADCD is statistically indistinguishable from the
2-param Simple-MOND and McGaugh RAR forms (CI brackets zero) and decisively
outperforms the 2-param Standard-MOND form (CI entirely below zero).}
\label{tab:sparc_bic_bootstrap}
\begin{tabular}{l c c c l}
\toprule
\textbf{Model (2-param)} & \textbf{BIC}$_{\rm eff}$ (95\% CI) &
$\delta{\rm BIC}_{\rm eff}$ & \textbf{95\% CI} & \textbf{Verdict} \\
\midrule
""" + "\n".join(boot_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_sparc_bic_bootstrap.tex").write_text(tab5, encoding="utf-8")

print("Generated SPARC tables:")
for p in sorted(OUT.glob("tab_sparc_*.tex")):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
