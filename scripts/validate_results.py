#!/usr/bin/env python3
"""
Validate consistency between benchmark JSON files and documented claims.
Exit code 0 = pass, 1 = inconsistencies found.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ERRORS = []
WARNINGS = []


def err(msg: str):
    ERRORS.append(msg)


def warn(msg: str):
    WARNINGS.append(msg)


def load(path: Path):
    if not path.exists():
        warn(f"Missing optional file: {path.name}")
        return None
    with open(path) as f:
        return json.load(f)


def validate_real_data(real):
    if not real:
        return
    for r in real:
        name = r.get("scenario", "?")
        nmse = r.get("nmse_full", 1.0)
        converged = r.get("converged", False)
        if name == "Real: Mercury Perihelion":
            if converged and nmse > 1e-3:
                err(f"Mercury marked converged but NMSE={nmse:.4f} > 1e-3")
            if nmse > 0.1 and converged:
                err(f"Mercury converged flag inconsistent with NMSE={nmse:.4f}")


def validate_adcd_telemetry(adcd):
    if not adcd:
        return
    missing = sum(1 for r in adcd if "gate_stats" not in r)
    if missing:
        warn(
            f"{missing}/{len(adcd)} ADCD entries lack gate_stats — "
            "re-run run_correction_discovery.py to refresh telemetry"
        )


def validate_pysr_profiles():
    fair = ROOT / "pysr_baseline_fair.json"
    if not fair.exists():
        warn("pysr_baseline_fair.json missing — run: py run_pysr_baseline.py --profile fair")


def validate_experiment_report():
    report = ROOT / "experiment_results.md"
    if not report.exists():
        warn("experiment_results.md missing")
        return
    text = report.read_text(encoding="utf-8")
    if "AUTO-GENERATED" not in text:
        warn("experiment_results.md is not auto-generated — run scripts/generate_experiment_report.py")
    if "1.34e-28" in text or "1.34e-28" in text.replace("-", ""):
        err("experiment_results.md contains stale Mercury NMSE claim (~1e-28)")


def main():
    validate_real_data(load(ROOT / "real_data_results.json"))
    validate_adcd_telemetry(load(ROOT / "scratch_correction_results.json"))
    validate_pysr_profiles()
    validate_experiment_report()

    for w in WARNINGS:
        print(f"WARNING: {w}")
    for e in ERRORS:
        print(f"ERROR: {e}")

    if ERRORS:
        print(f"\nFAILED: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
        return 1
    print(f"\nPASSED with {len(WARNINGS)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
