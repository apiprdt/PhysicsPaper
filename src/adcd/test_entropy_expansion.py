import os
import sys

# Add src to path so we can run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from adcd.run_adcd_v3_validation import run_full_protocol

if __name__ == "__main__":
    print("Testing Entropy Expansion scenario (ground truth = D_log)")
    try:
        results = run_full_protocol(
            scenario_name="Entropy Expansion",
            ratio_symbol="dV / V_i",
            ground_truth_primitive="D_log",
            seed=42
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
