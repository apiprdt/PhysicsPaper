import os

base_dir = r"e:\ADCD"
output_file = r"e:\ADCD\PhysicsPaper\allcode.txt"

files = [
    # Julia Engine
    (r"ADCDEngine\src\ADCDEngine.jl", "Main module file for the Julia engine"),
    (r"ADCDEngine\src\ADCDDimensions.jl", "Dimensional analysis and units handling"),
    (r"ADCDEngine\src\PrimitiveRegistry.jl", "Registry of mathematical primitives/functions"),
    (r"ADCDEngine\src\CorrectionProposer.jl", "Proposes mathematical corrections"),
    (r"ADCDEngine\src\ConstantFitter.jl", "Fits constants in symbolic expressions"),
    (r"ADCDEngine\src\IdentifiabilityGate.jl", "Checks identifiability of proposed expressions"),
    (r"ADCDEngine\src\FilterCascade.jl", "Cascaded filtering of proposed models"),
    
    # Julia Engine Test
    (r"ADCDEngine\test\runtests.jl", "Test suite for the Julia engine"),
    
    # Julia Project Config
    (r"ADCDEngine\Project.toml", "Julia project dependencies and metadata"),
    
    # Python ADCD Package
    (r"PhysicsPaper\src\adcd\__init__.py", "Package initialization"),
    (r"PhysicsPaper\src\adcd\anomaly_scenarios.py", "Anomaly detection scenarios"),
    (r"PhysicsPaper\src\adcd\arc_scorer.py", "Scoring mechanisms for models"),
    (r"PhysicsPaper\src\adcd\asymptotic_dictionary_proposer_v3.py", "Asymptotic dictionary proposer logic"),
    (r"PhysicsPaper\src\adcd\auto_scenario.py", "Automated scenario generation/handling"),
    (r"PhysicsPaper\src\adcd\bayesian_ranker.py", "Bayesian ranking of models"),
    (r"PhysicsPaper\src\adcd\budget_sweep.py", "Computation budget sweeping utilities"),
    (r"PhysicsPaper\src\adcd\coarse_evaluator.py", "Coarse evaluation of candidates"),
    (r"PhysicsPaper\src\adcd\constants.py", "Package-wide constants"),
    (r"PhysicsPaper\src\adcd\context.py", "Context management for ADCD pipeline"),
    (r"PhysicsPaper\src\adcd\dimensional_checker.py", "Python-side dimensional checking"),
    (r"PhysicsPaper\src\adcd\grammar_proposer_v3.py", "Grammar-based model proposer"),
    (r"PhysicsPaper\src\adcd\identifiability.py", "Python-side identifiability checks"),
    (r"PhysicsPaper\src\adcd\jax_optimizer.py", "JAX-based optimization routines"),
    (r"PhysicsPaper\src\adcd\julia_bridge.py", "Bridge to communicate with Julia engine"),
    (r"PhysicsPaper\src\adcd\metrics.py", "Evaluation metrics"),
    (r"PhysicsPaper\src\adcd\mode_detection.py", "Mode detection for datasets"),
    (r"PhysicsPaper\src\adcd\pipeline.py", "Main execution pipeline"),
    (r"PhysicsPaper\src\adcd\quickfit.py", "Quick fitting routines"),
    (r"PhysicsPaper\src\adcd\real_data_loader.py", "Loader for real-world datasets"),
    (r"PhysicsPaper\src\adcd\real_scenarios.py", "Definitions of real-world scenarios"),
    (r"PhysicsPaper\src\adcd\residual_features.py", "Feature extraction from residuals"),
    (r"PhysicsPaper\src\adcd\run_adcd_v3_validation_blind.py", "Validation script for V3 pipeline"),
    
    # Python Scripts
    (r"PhysicsPaper\test_sparc_julia.py", "Testing script for SPARC/Julia integration")
]

# Write out the contents
with open(output_file, "w", encoding="utf-8") as out:
    # Header
    out.write(f"# ADCD Engine v2 — All Core Source Code\n")
    out.write(f"# Generated: 2026-08-20T14:34:33+07:00\n")
    out.write(f"# Architecture: Python UI/validation (adcd/) <-> Julia compute engine (ADCDEngine/)\n\n")
    
    # Table of contents
    out.write("TABLE OF CONTENTS\n")
    for filepath, role in files:
        out.write(f"- {filepath.replace(os.sep, '/')}: {role}\n")
    
    out.write("\n")
    
    total_lines = 0
    total_files = 0
    
    for filepath, role in files:
        abs_path = os.path.join(base_dir, filepath)
        if not os.path.exists(abs_path):
            print(f"Warning: File not found - {abs_path}")
            continue
            
        total_files += 1
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            lines = content.split('\n')
            total_lines += len(lines)
            
        out.write("================================================================\n")
        out.write(f"FILE: {filepath.replace(os.sep, '/')}\n")
        out.write(f"ROLE: {role}\n")
        out.write("================================================================\n")
        out.write(content)
        if not content.endswith('\n'):
            out.write('\n')
        out.write("\n")

size_kb = os.path.getsize(output_file) / 1024
print(f"Total files processed: {total_files}")
print(f"Total lines written: {total_lines}")
print(f"Output file size: {size_kb:.2f} KB")
