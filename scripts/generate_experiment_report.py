#!/usr/bin/env python3
"""
Generate experiment_results.md from canonical JSON result files.
Run after benchmarks to keep documentation synchronized with measured data.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _class_match_rate(results, noise=None, tier=None):
    subset = results
    if noise is not None:
        subset = [r for r in subset if abs(r["noise"] - noise) < 1e-9]
    if tier is not None:
        subset = [r for r in subset if r.get("tier") == tier]
    if not subset:
        return 0, 0, 0.0
    matched = sum(1 for r in subset if r.get("class_match"))
    return matched, len(subset), matched / len(subset) * 100


def _aggregate_gate_stats(results):
  totals = {
      "input_count": 0,
      "parse_fail": 0,
      "ast_reject": 0,
      "dim_reject": 0,
      "transcendental_reject": 0,
      "arc_reject": 0,
      "coarse_reject": 0,
      "output_count": 0,
      "proposed": 0,
      "survived_stage1": 0,
      "optimized": 0,
  }
  for r in results:
      gs = r.get("gate_stats") or {}
      for k in ("input_count", "parse_fail", "ast_reject", "dim_reject",
                "transcendental_reject", "arc_reject", "coarse_reject", "output_count"):
          totals[k] += gs.get(k, 0)
      totals["proposed"] += r.get("total_candidates_proposed", 0)
      totals["survived_stage1"] += r.get("total_candidates_survived_stage1", 0)
      totals["optimized"] += r.get("total_candidates_optimized", 0)
  return totals


def generate():
    adcd = _load(ROOT / "scratch_correction_results.json")
    real = _load(ROOT / "real_data_results.json")
    repro = _load(ROOT / "reproducibility_results.json")
    pysr_fast = _load(ROOT / "pysr_baseline_results.json")
    pysr_fair = _load(ROOT / "pysr_baseline_fair.json")

    lines = [
        "# Experiment Report: ADCD Correction Discovery",
        "",
        "> **AUTO-GENERATED** — do not edit manually.",
        f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | git: `{_git_commit()}`",
        "",
        "Regenerate: `python scripts/generate_experiment_report.py`",
        "",
    ]

    if adcd:
        std9 = [r for r in adcd if r.get("tier") in ("textbook", "cross_domain", "synthetic")]
        m, n, pct = _class_match_rate(std9)
        lines += [
            "## Standard Benchmark (9 scenarios × 4 noise levels)",
            "",
            f"- **Class match (9-scenario subset):** {m}/{n} ({pct:.1f}%)",
        ]
        for noise in [0.0, 0.01, 0.05, 0.10]:
            m, n, pct = _class_match_rate(std9, noise=noise)
            lines.append(f"- **{noise*100:.0f}% noise:** {m}/{n} ({pct:.1f}%)")

        agg = _aggregate_gate_stats(adcd)
        if agg["input_count"] > 0:
            lines += [
                "",
                "### Search Space Telemetry (ADCD)",
                "",
                "| Metric | Count |",
                "|--------|------:|",
                f"| Candidates proposed | {agg['proposed']:,} |",
                f"| Gate pipeline inputs | {agg['input_count']:,} |",
                f"| Parse failures | {agg['parse_fail']:,} |",
                f"| AST rejections | {agg['ast_reject']:,} |",
                f"| Dimensional rejections | {agg['dim_reject']:,} |",
                f"| Transcendental rejections | {agg['transcendental_reject']:,} |",
                f"| ARC rejections | {agg['arc_reject']:,} |",
                f"| Coarse NMSE rejections | {agg['coarse_reject']:,} |",
                f"| Stage 1 survivors (unique per iter) | {agg['survived_stage1']:,} |",
                f"| Stage 2 optimization calls | {agg['optimized']:,} |",
                f"| Gate output (passed all) | {agg['output_count']:,} |",
            ]
            if agg["input_count"]:
                lines.append(
                    f"- **Overall gate survival:** {agg['output_count']/agg['input_count']*100:.1f}%"
                )
        lines.append("")

    if real:
        lines += ["## Real-World Validation (synthetic-real hybrid)", ""]
        lines.append("| Scenario | Class Match | Converged | Full NMSE |")
        lines.append("|:---|:---:|:---:|:---:|")
        converged_count = 0
        for r in real:
            conv = r.get("converged", False)
            if conv:
                converged_count += 1
            nmse = r.get("nmse_full", float("nan"))
            lines.append(
                f"| {r['scenario']} | {'✓' if r.get('class_match') else '✗'} | "
                f"{'✓' if conv else '—'} | {nmse:.2e} |"
            )
        n_struct = sum(1 for r in real if r.get("class_match"))
        n_quant = sum(1 for r in real if r.get("nmse_full", 1) < 1e-4)
        lines += [
            "",
            f"**Summary:** {n_struct}/{len(real)} structural class matches; "
            f"{n_quant}/{len(real)} quantitative (NMSE $< 10^{{-4}}$); "
            f"{converged_count}/{len(real)} optimizer converged ($< 10^{{-5}}$).",
            "",
            "> Data are synthetic-real hybrid (JPL/NIST/CODATA constants), not raw instrument archives.",
            "> Template-assisted (mock) vs zero-shot (hybrid/gemini) results must be reported separately.",
            "> Binary pulsar v2.1 uses reduced-variable formulation (P only); see sensitivity study.",
            "",
        ]

    if pysr_fast or pysr_fair:
        lines += ["## PySR Baseline Comparison", ""]
        for label, data in [("fast (legacy)", pysr_fast), ("fair", pysr_fair)]:
            if not data:
                continue
            m, n, pct = _class_match_rate(data)
            times = [r.get("time_seconds", 0) for r in data]
            exprs = [r.get("expressions_evaluated", 0) for r in data]
            mean_t = sum(times) / len(times) if times else 0
            mean_e = sum(exprs) / len(exprs) if exprs else 0
            lines.append(
                f"- **PySR {label}:** {m}/{n} class matches ({pct:.1f}%), "
                f"mean {mean_t:.1f}s/scenario, mean hall-of-fame size {mean_e:.0f}"
            )
        lines.append("")

    if repro:
        lines += [
            "## Reproducibility",
            "",
            f"- Entries in `reproducibility_results.json`: {len(repro)}",
            "",
        ]

    out = ROOT / "experiment_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(generate())
