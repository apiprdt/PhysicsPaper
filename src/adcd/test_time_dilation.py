from adcd.run_adcd_v3_validation import run_full_protocol

if __name__ == "__main__":
    print("Testing Time Dilation scenario (ground truth = D_sqrt_inv)")
    try:
        results = run_full_protocol(
            scenario_name="Time Dilation",
            ratio_symbol="-(v / c)**2",
            ground_truth_primitive="D_sqrt_inv",
            seed=42
        )
    except Exception as e:
        print(f"Error running protocol: {e}")
