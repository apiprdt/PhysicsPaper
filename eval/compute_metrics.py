import os
import json
import numpy as np
from collections import defaultdict
from scipy import stats

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "preregistered_audit.jsonl")


def load_audit_data() -> list[dict]:
    entries = []
    if not os.path.exists(AUDIT_LOG_PATH):
        print(f"[ERROR] Audit log not found: {AUDIT_LOG_PATH}")
        return entries
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def wilcoxon_test(a: list[float], b: list[float], label: str) -> dict:
    """Paired Wilcoxon signed-rank test with effect size (Cohen's d)."""
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    if n < 5:
        return {"label": label, "n": n, "skipped": True, "reason": "n < 5"}

    diff = a[:n] - b[:n]
    try:
        stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
    except ValueError:
        stat, p = float("nan"), float("nan")

    # Cohen's d on the difference
    pooled_std = np.std(np.concatenate([a, b])) + 1e-9
    cohens_d = float(np.mean(diff) / pooled_std)

    return {
        "label": label,
        "n_pairs": n,
        "mean_a": float(np.mean(a[:n])),
        "mean_b": float(np.mean(b[:n])),
        "mean_diff (a-b)": float(np.mean(diff)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "cohens_d": cohens_d,
        "significant (p<0.05)": bool(p < 0.05),
    }


def print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def main():
    entries = load_audit_data()
    if not entries:
        print("[ERROR] No entries to analyze.")
        return

    print(f"[INFO] Loaded {len(entries)} audit entries")

    # Check for PySR entries
    pysr_entries = [e for e in entries if e.get("proposer") == "PySR"]
    gp_entries = [e for e in entries if e.get("proposer") == "GrammarProposer"]
    print(f"[INFO] PySR entries: {len(pysr_entries)}, GrammarProposer entries: {len(gp_entries)}")

    if not pysr_entries:
        print("[WARN] No PySR entries found — M2/M3/M4 will be computed from GrammarProposer only.")
        print("[WARN] Re-run benchmark_runner.py with PySR enabled to get the primary experiment data.")

    # ── M1: Physical Validity Rate (PVR) ─────────────────────────────────
    print_section("M1: Physical Validity Rate (PVR) — by arm, level, noise")

    pvr_by_key = defaultdict(list)
    for e in entries:
        key = (e["level"], e.get("arm", "ARM-A"), e["noise"])
        pvr_by_key[key].append(e["pvr"])

    for (level, arm, noise), pvrs in sorted(pvr_by_key.items()):
        print(f"  {level} | {arm} | noise={noise:.2f}  →  "
              f"PVR={np.mean(pvrs)*100:.1f}% ± {np.std(pvrs)*100:.1f}%  (n={len(pvrs)})")

    # ── M2: False Positive Reduction Rate (FPRR) ─────────────────────────
    print_section("M2: False Positive Reduction Rate (FPRR)")

    fprr_by_key = defaultdict(list)
    for e in entries:
        fprr = e.get("fprr")
        if fprr is not None:  # None when no good-fitting candidates exist
            key = (e["level"], e.get("arm", "ARM-A"), e["proposer"])
            fprr_by_key[key].append(fprr)

    if fprr_by_key:
        for (level, arm, proposer), fprrs in sorted(fprr_by_key.items()):
            n_valid = len([f for f in fprrs if not np.isnan(f)])
            if n_valid == 0:
                print(f"  {level} | {arm} | {proposer}  →  FPRR: N/A (no good-fitting candidates)")
            else:
                valid_fprrs = [f for f in fprrs if not np.isnan(f)]
                print(f"  {level} | {arm} | {proposer}  →  "
                      f"FPRR={np.mean(valid_fprrs)*100:.1f}% ± {np.std(valid_fprrs)*100:.1f}%  "
                      f"(n={n_valid})")
    else:
        print("  [NOTE] FPRR not computed (no entries with fprr field or no good-fitting candidates)")
        print("  This means no proposer generated candidates with NMSE < 0.10 on test data.")
        print("  → For PySR entries, FPRR should be non-zero. Check if PySR ran.")

    # ── M3: Valid Recovery Rate (VRR) ────────────────────────────────────
    print_section("M3: Valid Recovery Rate (VRR) — structural similarity to ground truth")

    vrr_by_key = defaultdict(list)
    for e in entries:
        vrr = e.get("vrr", 0.0)
        key = (e["level"], e.get("arm", "ARM-A"), e["proposer"])
        vrr_by_key[key].append(vrr)

    for (level, arm, proposer), vrrs in sorted(vrr_by_key.items()):
        print(f"  {level} | {arm} | {proposer}  →  "
              f"VRR={np.mean(vrrs)*100:.1f}% ± {np.std(vrrs)*100:.1f}%  (n={len(vrrs)})")

    # ── M4: Correction-First vs Direct Fit (Level B only) ───────────────
    print_section("M4: Correction-First (ARM-B) vs Direct-Fit (ARM-C) — Level B")

    b_vrr_by_noise = defaultdict(list)
    c_vrr_by_noise = defaultdict(list)

    for e in entries:
        if e["level"] != "Level B":
            continue
        arm = e.get("arm", "")
        vrr = e.get("vrr", 0.0)
        noise = e["noise"]
        if "ARM-B" in arm:
            b_vrr_by_noise[noise].append(vrr)
        elif "ARM-C" in arm:
            c_vrr_by_noise[noise].append(vrr)

    if b_vrr_by_noise and c_vrr_by_noise:
        for noise in sorted(b_vrr_by_noise.keys()):
            b_vrrs = b_vrr_by_noise[noise]
            c_vrrs = c_vrr_by_noise.get(noise, [])
            if b_vrrs and c_vrrs:
                print(f"  noise={noise:.2f}  →  "
                      f"ARM-B (correction-first): VRR={np.mean(b_vrrs)*100:.1f}%  |  "
                      f"ARM-C (direct-fit): VRR={np.mean(c_vrrs)*100:.1f}%")
    else:
        print("  [NOTE] ARM-B and/or ARM-C data not available yet.")
        print("  Ensure PySR runs completed in benchmark_runner.py")

    # ── Statistical Tests ─────────────────────────────────────────────────
    print_section("Statistical Tests — Wilcoxon Signed-Rank (pre-registered)")

    # Test 1: PySR ARM-B PVR vs GrammarProposer ARM-A PVR (Level B)
    b_pvrs = [e["pvr"] for e in entries
              if e["level"] == "Level B" and "ARM-B" in e.get("arm", "")]
    a_pvrs = [e["pvr"] for e in entries
              if e["level"] == "Level B" and e.get("arm", "ARM-A") == "ARM-A"
              and e.get("proposer") == "GrammarProposer"]

    if b_pvrs and a_pvrs:
        result = wilcoxon_test(b_pvrs, a_pvrs,
                               "ARM-B (PySR+gate) PVR > ARM-A (Grammar+gate) PVR")
        print(f"\n  Test: {result['label']}")
        if result.get("skipped"):
            print(f"    Skipped: {result['reason']}")
        else:
            print(f"    n_pairs={result['n_pairs']}, mean_B={result['mean_a']:.3f}, "
                  f"mean_A={result['mean_b']:.3f}")
            print(f"    Wilcoxon p={result['p_value']:.4f}, Cohen's d={result['cohens_d']:.3f}")
            print(f"    Significant (p<0.05): {result['significant (p<0.05)']}")

    # Test 2: Correction-first VRR > Direct-fit VRR (Level B, medium noise)
    b_vrrs_med = [e.get("vrr", 0.0) for e in entries
                  if e["level"] == "Level B" and "ARM-B" in e.get("arm", "")
                  and e["noise"] == 0.05]
    c_vrrs_med = [e.get("vrr", 0.0) for e in entries
                  if e["level"] == "Level B" and "ARM-C" in e.get("arm", "")
                  and e["noise"] == 0.05]

    if b_vrrs_med and c_vrrs_med:
        result2 = wilcoxon_test(b_vrrs_med, c_vrrs_med,
                                "M4: Correction-first VRR > Direct-fit VRR (noise=0.05)")
        print(f"\n  Test: {result2['label']}")
        if result2.get("skipped"):
            print(f"    Skipped: {result2['reason']}")
        else:
            print(f"    n_pairs={result2['n_pairs']}, mean_CF={result2['mean_a']:.3f}, "
                  f"mean_Direct={result2['mean_b']:.3f}")
            print(f"    Wilcoxon p={result2['p_value']:.4f}, Cohen's d={result2['cohens_d']:.3f}")
            print(f"    Significant (p<0.05): {result2['significant (p<0.05)']}")

    # ── Overall Summary ───────────────────────────────────────────────────
    print_section("OVERALL SUMMARY")
    total = len(entries)
    n_pysr = len(pysr_entries)
    n_gp = len(gp_entries)
    print(f"  Total benchmark entries logged : {total}")
    print(f"  GrammarProposer entries        : {n_gp}")
    print(f"  PySR entries                   : {n_pysr}")

    overall_pvr = np.mean([e["pvr"] for e in entries]) if entries else 0.0
    print(f"  Overall PVR (all arms)         : {overall_pvr*100:.2f}%")

    # Pre-registered success criteria check
    print("\n  Pre-Registered Criteria Check:")
    print("  ──────────────────────────────")

    # M2: FPRR >= 15%
    all_fprr = [e["fprr"] for e in pysr_entries if e.get("fprr") is not None]
    if all_fprr:
        mean_fprr = np.mean([f for f in all_fprr if not np.isnan(f)])
        status = "✓ PASS" if mean_fprr >= 0.15 else "✗ FAIL"
        print(f"  M2 FPRR >= 15% : {mean_fprr*100:.1f}%  →  {status}")
    else:
        print("  M2 FPRR >= 15% : NOT YET MEASURABLE (PySR data pending)")

    # M4: VRR_CF >= VRR_direct
    if b_vrrs_med and c_vrrs_med:
        mean_b = np.mean(b_vrrs_med)
        mean_c = np.mean(c_vrrs_med)
        status4 = "✓ PASS" if mean_b >= mean_c else "✗ FAIL"
        print(f"  M4 CF>=Direct  : CF={mean_b*100:.1f}% >= Direct={mean_c*100:.1f}%  →  {status4}")
    else:
        print("  M4 CF>=Direct  : NOT YET MEASURABLE (PySR ARM-B/C data pending)")

    print("\n[INFO] Run benchmark_runner.py to populate PySR entries before final analysis.")


if __name__ == "__main__":
    main()
