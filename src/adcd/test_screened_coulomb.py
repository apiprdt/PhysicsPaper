import sys
import json
from adcd.run_adcd_v3_validation import run_full_protocol

if __name__ == "__main__":
    results = run_full_protocol(
        scenario_name="Screened Coulomb",
        ratio_symbol="r / theta_1",
        ground_truth_primitive="D_exp",
        seed=42
    )
