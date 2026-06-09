#!/usr/bin/env python3
"""
Generate real-world validation tables for the paper:
  1. Parameter recovery (structural + quantitative)
  2. Template leakage disclosure
  3. Binary pulsar sensitivity (requires binary_pulsar_sensitivity.json)

Run: py -3.11 scripts/generate_real_world_tables.py
Prereq: py -3.11 run_real_data_benchmark.py
         py -3.11 run_binary_pulsar_sensitivity.py  (optional)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adcd.real_scenarios import get_real_scenarios

# ── Table 3: static template leakage analysis ────────────────────────────────

LEAKAGE_ROWS = [
    ("Real: Mercury Perihelion", r"\theta_0 \cdot (v/c)^2", "Yes", "Yes", "polynomial"),
    ("Real: Hydrogen Lamb Shift", r"\theta_0 / n^3", "No", "Yes", "power\_law"),
    ("Real: Blackbody Radiation", r"\exp(-hf/k_BT)", "No", "Yes", "exponential"),
    ("Real: Muon g-2", r"\theta_0 \cdot \alpha / \pi", "No", "Yes", "polynomial"),
    (
        "Real: Binary Pulsar Decay",
        r"\theta_0 \cdot P^{-5/3}",
        "No",
        "Yes",
        "power\_law",
    ),
]


def _extract_exponent(expr: str, fitted: dict) -> str:
    """Best-effort recovered exponent for power-law forms."""
    if not expr:
        return "—"
    m = re.search(r"P\*\*\(-([\d./]+)\)", expr.replace(" ", ""))
    if m:
        return m.group(1)
    m = re.search(r"P\*\*\(-theta_1\)", expr.replace(" ", ""))
    if m and "theta_1" in fitted:
        return f"{fitted['theta_1']:.3f}"
    m = re.search(r"/P\*\*theta_1", expr.replace(" ", ""))
    if m and "theta_1" in fitted:
        return f"{fitted['theta_1']:.3f}"
    return "—"


def _latex_expr(s: str) -> str:
    """Minimal SymPy-like string → LaTeX for table cells."""
    s = s.replace("**", "^")
    s = s.replace("*", " \\cdot ")
    s = s.replace("theta_0", r"\theta_0").replace("theta_1", r"\theta_1").replace("theta_2", r"\theta_2")
    s = s.replace("vc2", r"(v/c)^2").replace("exp", r"\exp").replace("log", r"\log")
    s = s.replace("alpha", r"\alpha").replace("pi", r"\pi")
    return s


def _recovery_notes(r: dict) -> str:
  if "Pulsar" in r["scenario"]:
      fitted = r.get("fitted_theta") or {}
      exp = _extract_exponent(r.get("discovered_expr", ""), fitted)
      if exp != "—":
          return f"exponent $\\approx {exp}$ (true $5/3$)"
  ast = r.get("ast_edit_distance", 999)
  if ast == 0:
      return "exact AST match"
  if ast <= 4:
      return f"AST dist $={ast}$"
  return "structural class only"


def generate_parameter_recovery_table(real_results: list) -> str:
    scenarios = {s.name: s for s in get_real_scenarios()}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Real-World Structural and Quantitative Recovery (Mock Proposer, seed=42). "
        r"Quantitative success: NMSE $< 10^{-4}$. Asterisk: structural match only.}",
        r"\label{tab:real_param_recovery}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{l l l c c c}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{True $\Delta$} & \textbf{Recovered $\Delta$} & "
        r"\textbf{Class} & \textbf{NMSE} & \textbf{Recovery Notes} \\",
        r"\midrule",
    ]
    for r in real_results:
        sc = scenarios.get(r["scenario"])
        true_expr = _latex_expr(sc.correction_expr if sc else "—")
        recovered = _latex_expr(r.get("discovered_expr", "—"))
        cls = "Yes" if r.get("class_match") else "No"
        nmse = r.get("nmse_full", float("nan"))
        star = "" if nmse < 1e-4 else "^{*}"
        nmse_s = f"${nmse:.2e}{star}$"
        notes = _recovery_notes(r)
        lines.append(
            f"{r['scenario'].replace('Real: ', '')} & ${true_expr}$ & ${recovered}$ & "
            f"{cls} & {nmse_s} & {notes} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_leakage_table() -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Template Leakage Disclosure for Real-World Benchmark (Mock Proposer extended mode). "
        r"``Exact'' = ground-truth functional form appears verbatim in the extended template bank.}",
        r"\label{tab:template_leakage}",
        r"\begin{tabularx}{\textwidth}{l l c c l}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Ground Truth} & \textbf{Exact Template?} & "
        r"\textbf{Family in Bank?} & \textbf{Expected Class} \\",
        r"\midrule",
    ]
    for name, truth, exact, family, cls in LEAKAGE_ROWS:
        lines.append(
            f"{name.replace('Real: ', '')} & ${truth}$ & {exact} & {family} & \\texttt{{{cls}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def generate_sensitivity_table(sensitivity: list) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Binary Pulsar Sensitivity: structural recovery vs.\ formulation difficulty. "
        r"v2.1 default uses the \texttt{P\_only} reduced-variable form.}",
        r"\label{tab:pulsar_sensitivity}",
        r"\begin{tabularx}{\textwidth}{l l l c c}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Free Variables} & \textbf{Discovered $\Delta$} & "
        r"\textbf{Class Match} & \textbf{NMSE} \\",
        r"\midrule",
    ]
    for r in sensitivity:
        vars_s = ", ".join(r.get("variables", []))
        match = "Yes" if r.get("class_match") else "No"
        expr = _latex_expr(r.get("discovered_expr", "—"))
        nmse = r.get("nmse_full", float("nan"))
        lines.append(
            f"\\texttt{{{r['variant']}}} & ${vars_s}$ & ${expr}$ & {match} & ${nmse:.2e}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    real_path = ROOT / "real_data_results.json"
    sens_path = ROOT / "binary_pulsar_sensitivity.json"
    out_dir = ROOT / "paper" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not real_path.exists():
        print("Missing real_data_results.json — run run_real_data_benchmark.py first")
        return 1

    with open(real_path) as f:
        real_results = json.load(f)

    tables = {
        "tab_real_param_recovery.tex": generate_parameter_recovery_table(real_results),
        "tab_template_leakage.tex": generate_leakage_table(),
    }

    if sens_path.exists():
        with open(sens_path) as f:
            tables["tab_pulsar_sensitivity.tex"] = generate_sensitivity_table(json.load(f))
    else:
        print("Note: binary_pulsar_sensitivity.json not found — skipping sensitivity table")

    for name, content in tables.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")

    # Summary stats for paper inline text
    n_struct = sum(1 for r in real_results if r.get("class_match"))
    n_quant = sum(1 for r in real_results if r.get("nmse_full", 1) < 1e-4)
    n_conv = sum(1 for r in real_results if r.get("converged"))
    summary = {
        "structural_matches": f"{n_struct}/{len(real_results)}",
        "quantitative_nmse_lt_1e4": f"{n_quant}/{len(real_results)}",
        "optimizer_converged": f"{n_conv}/{len(real_results)}",
    }
    (out_dir / "real_world_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
