#!/usr/bin/env python3
"""
Generate PySR fair-comparison LaTeX tables for the paper.

Reads pysr_baseline_fair.json, pysr_profiles.py, scratch_correction_results.json,
gate_telemetry.json.

Run: py -3.11 scripts/generate_pysr_fair_tables.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pysr_profiles import PYSR_PROFILES

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]

# Primary benchmark = 9 standard scenarios (textbook / cross_domain / synthetic).
# Exclude blind (held-out) and multivariable (different benchmark family) so all
# PySR profiles are compared on the same residual targets as ADCD.
STD9_TIERS = ("textbook", "cross_domain", "synthetic")


def _filter_std9(results):
    """Restrict a result list to the 9-scenario primary benchmark."""
    return [r for r in results if r.get("tier") in STD9_TIERS]


def _class_at_noise(results, noise):
    sub = [r for r in results if abs(r["noise"] - noise) < 1e-9]
    m = sum(1 for r in sub if r.get("class_match"))
    return m, len(sub)


def generate_pysr_config_table() -> str:
    fair = PYSR_PROFILES["fair"]
    fast = PYSR_PROFILES["fast"]
    generous = PYSR_PROFILES["generous"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{PySR baseline configuration profiles. Primary comparison uses \texttt{fair} "
        r"(near-default search budget). \texttt{fast} is legacy wall-clock-matched; "
        r"\texttt{generous} is upper-bound ablation.}",
        r"\label{tab:pysr_config}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{l c c c c c c}",
        r"\toprule",
        r"\textbf{Profile} & \textbf{Iterations} & \textbf{Max Size} & \textbf{Timeout (s)} & "
        r"\textbf{Populations} & \textbf{Pop.\ Size} & \textbf{Seed} \\",
        r"\midrule",
        r"\texttt{fast} (legacy) & "
        f"{fast['niterations']} & {fast['maxsize']} & {fast['timeout_in_seconds']} & 8 & 20 & 42 \\\\",
        r"\texttt{fair} (primary) & "
        f"{fair['niterations']} & {fair['maxsize']} & {fair['timeout_in_seconds']} & 8 & 20 & 42 \\\\",
        r"\texttt{generous} & "
        f"{generous['niterations']} & {generous['maxsize']} & {generous['timeout_in_seconds']} & 8 & 20 & 42 \\\\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Shared: binary ops }$\{+,-,\times,\div\}$; "
        r"unary ops $\{\exp,\sin,\cos,\log,\tanh,\sqrt\}$; parsimony $=0.0032$; "
        r"deterministic $=$ True; same residual $y - y_{\mathrm{classical}}$ as ADCD.} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_baseline_comparison_table(adcd, pysr_fair, pysr_fast=None,
                                       pysr_generous=None) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Structural class match rate: ADCD (Mock Proposer, seed=42) vs.\ PySR on the same "
        r"9-scenario residual benchmark. PySR \texttt{fair} is the primary unconstrained baseline; "
        r"\texttt{generous} ($2\times$ iterations, $2\times$ timeout, larger \texttt{maxsize}) is an "
        r"upper-bound ablation showing that additional search budget does not close the gap.}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{0\%} & \textbf{1\%} & \textbf{5\%} & \textbf{10\%} \\",
        r"\midrule",
    ]
    for label, data in [("ADCD (ours)", adcd), ("PySR fair", pysr_fair),
                        ("PySR generous (upper bound)", pysr_generous)]:
        if not data:
            continue
        row = [label]
        for noise in NOISE_LEVELS:
            m, n = _class_at_noise(data, noise)
            row.append(f"{m}/{n} ({100*m/n:.1f}\\%)")
        lines.append(" & ".join(row) + r" \\")
    if pysr_fast:
        row = ["PySR fast (legacy)"]
        for noise in NOISE_LEVELS:
            m, n = _class_at_noise(pysr_fast, noise)
            row.append(f"{m}/{n} ({100*m/n:.1f}\\%)")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_search_funnel_table(gate: dict) -> str:
    inp = gate.get("input_count", 0)
    stages = [
        ("Proposed (Stage 0b)", gate.get("total_proposed", inp)),
        ("After parse", inp - gate.get("parse_fail", 0)),
        ("After AST", inp - gate.get("parse_fail", 0) - gate.get("ast_reject", 0)),
        ("After dimensional", inp - gate.get("parse_fail", 0) - gate.get("ast_reject", 0) - gate.get("dim_reject", 0)),
        ("After transcendental", inp - gate.get("parse_fail", 0) - gate.get("ast_reject", 0)
         - gate.get("dim_reject", 0) - gate.get("transcendental_reject", 0)),
        ("After ARC (Stage 1 out)", gate.get("output_count", 0)),
        ("JAX optimized (Stage 2)", gate.get("total_optimized", 0)),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{ADCD search-space funnel (aggregate over 9 scenarios $\times$ 4 noise levels, "
        r"Mock Proposer seed=42). Telemetry from \texttt{gate\_telemetry.json}.}",
        r"\label{tab:search_funnel}",
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Candidates} & \textbf{Survival (\%)} \\",
        r"\midrule",
    ]
    base = stages[0][1] or 1
    for name, count in stages:
        pct = 100.0 * count / base
        lines.append(f"{name} & {count:,} & {pct:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_runtime_table(adcd, pysr_fair, pysr_generous=None) -> str:
    adcd_t = sum(r.get("time_seconds", 0) for r in adcd) / len(adcd) if adcd else 0
    pysr_t = sum(r.get("time_seconds", 0) for r in pysr_fair) / len(pysr_fair) if pysr_fair else 0
    adcd_opt = sum(r.get("total_candidates_optimized", 0) for r in adcd)
    pysr_expr = sum(r.get("expressions_evaluated", 0) for r in pysr_fair)
    pysr_g_t = (sum(r.get("time_seconds", 0) for r in pysr_generous) / len(pysr_generous)
                if pysr_generous else None)
    pysr_g_expr = (sum(r.get("expressions_evaluated", 0) for r in pysr_generous) / len(pysr_generous)
                   if pysr_generous else None)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Compute budget comparison (9 scenarios $\times$ 4 noise levels).}",
        r"\label{tab:runtime}",
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Mean wall-clock / run} & \textbf{Expressions explored} \\",
        r"\midrule",
        f"ADCD (Mock) & {adcd_t:.1f}\\,s & {adcd_opt:,} JAX optim. calls \\\\",
        f"PySR (fair) & {pysr_t:.1f}\\,s & $\\sim${pysr_expr // len(pysr_fair) if pysr_fair else 0} hall-of-fame / run \\\\",
    ]
    if pysr_g_t is not None:
        lines.append(
            f"PySR (generous) & {pysr_g_t:.1f}\\,s & "
            f"$\\sim${pysr_g_expr:.0f} hall-of-fame / run \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    out = ROOT / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)

    adcd_path = ROOT / "scratch_correction_results.json"
    fair_path = ROOT / "pysr_baseline_fair.json"
    fast_path = ROOT / "pysr_baseline_results.json"
    generous_path = ROOT / "pysr_baseline_generous.json"
    gate_path = ROOT / "gate_telemetry.json"
    repro_path = ROOT / "reproducibility_results.json"

    if adcd_path.exists():
        adcd = json.load(open(adcd_path))
    elif repro_path.exists():
        # scratch_correction_results.json is gitignored and may be absent in a
        # fresh checkout; the same seed=42 primary-benchmark data is preserved
        # inside reproducibility_results.json, so fall back to it.
        repro = json.load(open(repro_path))
        adcd = [r for r in repro if r.get("seed") == 42]
        print(f"NOTE: {adcd_path.name} missing — using seed=42 slice from "
              f"{repro_path.name} ({len(adcd)} entries).")
    else:
        adcd = []
    std9 = _filter_std9(adcd)
    # Filter every PySR profile to the 9-scenario primary benchmark so that all
    # profiles (which may have picked up extra tiers, e.g. multivariable) are
    # compared on the same residual targets as ADCD.
    pysr_fair = _filter_std9(json.load(open(fair_path))) if fair_path.exists() else []
    pysr_fast = _filter_std9(json.load(open(fast_path))) if fast_path.exists() else None
    pysr_generous = (_filter_std9(json.load(open(generous_path)))
                     if generous_path.exists() else None)
    gate = json.load(open(gate_path)) if gate_path.exists() else {}

    tables = {
        "tab_pysr_config.tex": generate_pysr_config_table(),
        "tab_baseline_comparison.tex": generate_baseline_comparison_table(
            std9, pysr_fair, pysr_fast, pysr_generous),
        "tab_search_funnel.tex": generate_search_funnel_table(gate),
        "tab_runtime.tex": generate_runtime_table(std9, pysr_fair, pysr_generous),
    }
    for name, content in tables.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
