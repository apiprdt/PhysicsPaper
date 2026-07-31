"""
Contamination Check Script for ADCD Pre-Registration Lock.
Scans the codebase to ensure zero pre-registered benchmark equations appear in any Python files.
"""

import sys
import os
import re

BENCHMARK_SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "preregistration", "benchmark_spec.txt")
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")


def run_contamination_check() -> bool:
    if not os.path.exists(BENCHMARK_SPEC_PATH):
        print(f"[ERROR] Benchmark spec not found at: {BENCHMARK_SPEC_PATH}")
        return False

    with open(BENCHMARK_SPEC_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    equations = []
    for line in lines:
        if ":" in line and not line.startswith("#"):
            eq = line.split(":", 1)[1].strip()
            if eq:
                equations.append(eq)

    print(f"[INFO] Loaded {len(equations)} benchmark equations for contamination check.")

    contaminated = False
    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                for eq in equations:
                    # Clean string comparison
                    eq_clean = eq.replace(" ", "")
                    if len(eq_clean) > 8 and eq_clean in content.replace(" ", ""):
                        print(f"[WARNING] Potential contamination detected in {file} for equation: {eq}")
                        contaminated = True

    if not contaminated:
        print("[SUCCESS] Zero contamination detected across codebase for pre-registered benchmarks!")
        return True
    else:
        print("[FAIL] Contamination check failed!")
        return False


if __name__ == "__main__":
    success = run_contamination_check()
    sys.exit(0 if success else 1)
