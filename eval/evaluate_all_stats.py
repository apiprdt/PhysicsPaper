import json
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.asymptotic_dictionary_proposer_v3 import AsymptoticDictionaryProposerV3

def get_stats():
    scenarios_cfg = [
        {"name": "Time Dilation", "ratio": "(v/c)**2"},
        {"name": "Screened Coulomb", "ratio": "r/theta_1"},
        {"name": "Entropy Expansion", "ratio": "(dV/V_i)"},
    ]
    
    all_scens = get_all_scenarios()
    stats_dict = {}
    
    for cfg in scenarios_cfg:
        scen = next(s for s in all_scens if s.name == cfg["name"])
        proposer = AsymptoticDictionaryProposerV3(ratio_symbol=cfg["ratio"])
        
        from adcd.pipeline import Stage1Pipeline
        from adcd.dimensional_checker import ASTValidator, DimensionalChecker
        from adcd.arc_scorer import ARCScorer, build_arc_regimes
        from adcd.jax_optimizer import JAXOptimizer
        from adcd.correction_orchestrator import CorrectionOrchestrator
        
        validator = ASTValidator(max_depth=9, max_tokens=40)
        checker = DimensionalChecker()
        regimes = build_arc_regimes(scen.classical_limit_variable, scen.classical_limit_direction)
        scorer = ARCScorer(regimes=regimes)
        pipeline = Stage1Pipeline(validator, checker, scorer)
        optimizer = JAXOptimizer()
        orchestrator = CorrectionOrchestrator(proposer=proposer, pipeline=pipeline, optimizer=optimizer, max_iterations=1, verbose=False)
        
        res = orchestrator.search_correction(scen, noise_level=0.01, seed=42)
        if res.gate_stats:
            stats_dict[cfg["name"]] = res.gate_stats.to_dict()
        else:
            stats_dict[cfg["name"]] = {}
        
    print(json.dumps(stats_dict, indent=2))

if __name__ == "__main__":
    get_stats()
