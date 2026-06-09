#!/usr/bin/env python3
"""
C2: Generate ADCD vs PySR efficiency comparison table (Markdown + LaTeX rows).

Reads gate_telemetry.json, scratch_correction_results.json, pysr_baseline_*.json

Usage:
    python scripts/generate_efficiency_table.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mean_time(data):
    if not data:
        return 0.0
    return sum(r.get("time_seconds", 0) for r in data) / len(data)


def _class_rate(data):
    if not data:
        return 0, 0, 0.0
    m = sum(1 for r in data if r.get("class_match"))
    return m, len(data), 100 * m / len(data)


def main():
    adcd_path = ROOT / "scratch_correction_results.json"
    gate_path = ROOT / "gate_telemetry.json"
    lines = ["## Efficiency Comparison (auto-generated)", ""]

    if adcd_path.exists():
        adcd = json.load(open(adcd_path))
        std9 = [r for r in adcd if r.get("tier") in ("textbook", "cross_domain", "synthetic")]
        m, n, pct = _class_rate(std9)
        mean_t = _mean_time(std9)
        proposed = sum(r.get("total_candidates_proposed", 0) for r in adcd)
        optimized = sum(r.get("total_candidates_optimized", 0) for r in adcd)
        lines.append("| Method | Profile | Class Match | Mean Time | Expr. Proposed | Optim Calls |")
        lines.append("|--------|---------|-------------|-----------|----------------|-------------|")
        lines.append(
            f"| ADCD | mock | {m}/{n} ({pct:.1f}%) | {mean_t:.1f}s | {proposed:,} | {optimized:,} |"
        )

    if gate_path.exists():
        gate = json.load(open(gate_path))
        lines.append("")
        lines.append("### Gate Funnel (aggregate)")
        lines.append("")
        lines.append("| Stage | Count |")
        lines.append("|-------|------:|")
        for k in ("input_count", "parse_fail", "ast_reject", "dim_reject",
                  "transcendental_reject", "arc_reject", "coarse_reject", "output_count"):
            if k in gate:
                lines.append(f"| {k} | {gate[k]:,} |")
        if gate.get("input_count"):
            lines.append(f"| overall survival | {gate['output_count']/gate['input_count']*100:.1f}% |")

    for profile in ("fast", "fair", "generous"):
        path = ROOT / f"pysr_baseline_{profile}.json"
        if not path.exists() and profile == "fast":
            path = ROOT / "pysr_baseline_results.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        m, n, pct = _class_rate(data)
        mean_t = _mean_time(data)
        exprs = [r.get("expressions_evaluated", 0) for r in data]
        mean_e = sum(exprs) / len(exprs) if exprs else 0
        lines.append(
            f"| PySR | {profile} | {m}/{n} ({pct:.1f}%) | {mean_t:.1f}s | ~{mean_e:.0f} hall-of-fame | — |"
        )

    out = ROOT / "efficiency_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
