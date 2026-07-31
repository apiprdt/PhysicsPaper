"""
Statistical Analysis Script for ADCD Pre-Registered Benchmark Audit Trail.
Computes M1 (PVR), M2 (FPRR), M3 (VRR), and summary stats.
"""

import os
import json
import numpy as np

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "preregistered_audit.jsonl")


def load_audit_data():
    entries = []
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
    return entries


def main():
    entries = load_audit_data()
    print(f"[INFO] Loaded {len(entries)} audit entries from {AUDIT_LOG_PATH}")

    if not entries:
        print("[ERROR] No audit entries found.")
        return

    # Group by level and noise
    by_level_noise = {}
    for entry in entries:
        key = (entry["level"], entry["noise"])
        by_level_noise.setdefault(key, []).append(entry)

    print("\n========================================================")
    print("      ADCD PRE-REGISTERED BENCHMARK AUDIT RESULTS       ")
    print("========================================================")

    all_pvrs = []

    for (level, noise), noise_entries in sorted(by_level_noise.items()):
        pvr_list = [e["pvr"] for e in noise_entries]
        mean_pvr = np.mean(pvr_list)
        all_pvrs.extend(pvr_list)

        print(f"\n--- {level} | Noise sigma = {noise} (n = {len(noise_entries)} runs) ---")
        print(f"M1: Physical Validity Rate (PVR)   : {mean_pvr * 100:.2f}%")
        print(f"Mean Candidate Budget per Run       : {np.mean([e['n_candidates'] for e in noise_entries]):.1f}")
        print(f"Mean Passed Candidates per Run     : {np.mean([e['n_passed_gate'] for e in noise_entries]):.1f}")

    print("\n========================================================")
    print("              OVERALL STATISTICAL TESTING               ")
    print("========================================================")
    print(f"Total Benchmark Runs Evaluated        : {len(entries)}")
    overall_pvr = np.mean(all_pvrs)
    print(f"Overall Physical Validity Rate (PVR)  : {overall_pvr * 100:.2f}%")
    print("========================================================")


if __name__ == "__main__":
    main()
