"""
ADCD Pareto Frontier Plotter (PhySO / PySR Style)
=================================================
Generates publication-quality PDF and PNG plots of discovered Pareto frontiers
for inclusion into LaTeX physics papers and manuscripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_validation_pareto_fronts(
    json_report_path: str | Path = "run_outputs/adcd_v3_taxonomy_validation_report.json",
    output_dir: str | Path = "run_outputs",
) -> list[str]:
    """
    Generate publication-ready PDF & PNG figures for each scenario's Pareto front.
    """
    json_path = Path(json_report_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[WARN] Report file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    generated_files = []

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    for scenario_name, report in data.items():
        checks = report.get("checks", {})
        ps = checks.get("primary_search", {})
        pareto = ps.get("pareto_front", [])
        if not pareto:
            continue

        tier = report.get("tier", "UNKNOWN")
        slug = scenario_name.lower().replace(" ", "_")

        ranks = list(range(1, len(pareto) + 1))
        nmses = [c.get("nmse", float("nan")) for c in pareto]
        bics = [c.get("bic", float("nan")) for c in pareto]
        labels = [c.get("class", "model") for c in pareto]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

        # Plot 1: Pareto Rank vs NMSE (Log Scale)
        ax1.plot(ranks, nmses, "o-", color="#1f77b4", linewidth=1.8, markersize=7, label="Pareto Frontier")
        ax1.scatter([ranks[0]], [nmses[0]], color="#d62728", s=120, zorder=5, label=f"Rank 1 ({labels[0]})")
        ax1.set_yscale("log")
        ax1.set_xlabel("Pareto Rank")
        ax1.set_ylabel("NMSE (Residual)")
        ax1.set_title("Fitness Curve")
        ax1.set_xticks(ranks)
        ax1.legend(frameon=True)

        # Plot 2: Pareto Rank vs BIC
        ax2.plot(ranks, bics, "s-", color="#2ca02c", linewidth=1.8, markersize=7, label="Extended Info Criterion")
        ax2.scatter([ranks[0]], [bics[0]], color="#d62728", s=120, zorder=5, label=f"Rank 1 (EBIC={bics[0]:.2f})")
        ax2.set_xlabel("Pareto Rank")
        ax2.set_ylabel("EBIC Score")
        ax2.set_title(f"Extended Bayesian Information Criterion\n(Tier: {tier})")
        ax2.set_xticks(ranks)
        ax2.legend(frameon=True)

        fig.suptitle(f"ADCD Discovered Pareto Frontier: {scenario_name}", y=1.03, weight="bold")
        fig.tight_layout()

        pdf_path = out_dir / f"pareto_{slug}.pdf"
        png_path = out_dir / f"pareto_{slug}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

        generated_files.extend([str(pdf_path), str(png_path)])
        print(f"Generated Pareto Plots: {pdf_path.name} & {png_path.name}")

    return generated_files


if __name__ == "__main__":
    plot_validation_pareto_fronts()
