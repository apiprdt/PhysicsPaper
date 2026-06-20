"""Generate LaTeX tables for the SPARC capstone section from result JSONs.

Reads:
  - results/sparc_discovery.json         (model comparison, CV, bootstrap)
  - results/sparc_robustness_results.json (3 quality-cut scenarios)

Writes:
  - paper/generated/tab_sparc_comparison.tex   (4-model NMSE/BIC/n_params)
  - paper/generated/tab_sparc_validation.tex   (CV mean +/- SE)
  - paper/generated/tab_sparc_bootstrap.tex    (parameter 95% CI)
  - paper/generated/tab_sparc_robustness.tex   (3 quality cuts)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISC = json.loads((ROOT / "results" / "sparc_discovery.json").read_text())
ROBUST = json.loads((ROOT / "results" / "sparc_robustness_results.json").read_text())
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

# ---- Table 2: cross-validation ---------------------------------------------
cv = DISC["cross_validation"]
order = ["Simple MOND", "Standard MOND", "RAR (McGaugh)", "ADCD discovered"]
cvrows = []
for name in order:
    s = cv[name]
    is_adcd = name.lower().startswith("adcd")
    mean = f(s["mean_nmse"], 4)
    se = f(s["std_error"], 4)
    mean_str = rf"\textbf{{{mean}}}" if is_adcd else mean
    cvrows.append(f"  {name} & {mean_str} $\\pm$ {se} \\\\")

tab2 = r"""\begin{table}[H]
\centering\footnotesize
\caption{Galaxy-level cross-validation (10 repeated 50/50 train/test splits by
galaxy). ADCD-discovered function generalizes best out-of-sample.}
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
boot = DISC["bootstrap_ci"]
bootrows = []
for param in sorted(boot.keys()):
    s = boot[param]
    mean = f(s["mean"], 3)
    lo = f(s["ci_lower"], 3)
    hi = f(s["ci_upper"], 3)
    bootrows.append(f"  ${param}$ & {mean} & [{lo}, {hi}] \\\\")

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

print("Generated 4 SPARC tables:")
for p in sorted(OUT.glob("tab_sparc_*.tex")):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
