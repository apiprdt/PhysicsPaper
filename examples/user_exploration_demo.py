#!/usr/bin/env python3
"""
examples/user_exploration_demo.py
==============================================================================
Demonstration: How a User / Physicist Can Tinker with ADCD & Test Wild Hypotheses.
==============================================================================
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import adcd

print("=" * 80)
print(" ADCD USER EXPLORATION DEMO")
print("=" * 80)

# -------------------------------------------------------------------------
# Step 1: List existing physical primitives
# -------------------------------------------------------------------------
print("\n[1] Listing Currently Registered Primitives in ADCD:")
for name, desc in adcd.list_primitives().items():
    print(f"  - {name:<15}: {desc}")

# -------------------------------------------------------------------------
# Step 2: Register a WILD Custom Hypothesis (e.g. Quantum Vacuum Friction)
# -------------------------------------------------------------------------
print("\n[2] Registering a Custom User Hypothesis:")
custom_prim = adcd.register_primitive(
    name="D_quantum_vacuum",
    template="(1.0 - exp(-Abs({u}))) * log(1.0 + ({u})**2)",
    token_cost=6,
    domain_note="Quantum vacuum friction hypothesis with logarithmic saturation",
    enforce_zero_limit=True  # Automatically checks lim_{u->0} D(u) == 0
)
print(f"  -> Successfully registered: {custom_prim.name} with template {custom_prim.string_template}")

# -------------------------------------------------------------------------
# Step 3: Run Discovery on Synthetic Observational Data
# -------------------------------------------------------------------------
print("\n[3] Running Discovery on Observational Data (Time Dilation):")
scenarios = {s.name: s for s in adcd.get_all_scenarios()}
time_dilation = scenarios["Time Dilation"]

# Run discovery with the new custom primitive in the dictionary
result = adcd.discover(
    scenario_or_data=time_dilation,
    seed=42,
    noise_level=0.01,
    engine="julia",
    use_taxonomy_prior=False,  # Full unconstrained exploration across all primitives
)

# -------------------------------------------------------------------------
# Step 4: Display the Discovery Report
# -------------------------------------------------------------------------
result.summary()
