
import sys
sys.path.insert(0, "e:\\ADCD\\PhysicsPaper\\src")
from adcd.run_adcd_v3_validation_blind import run_full_validation_suite

# Run just positive control with julia
run_full_validation_suite(n_candidates=10, mode="positive_control", engine="julia")
