import numpy as np
import pytest
from adcd.iadcd_orchestrator import iADCDOrchestrator

def test_iadcd_orchestrator_synthetic():
    # Generate synthetic target that has a two-step correction series:
    # y = y_classical + c1 * x + c2 * x**2
    # Where y_classical = x**3
    np.random.seed(42)
    x = np.linspace(0.1, 1.0, 100)
    y_classical = x**3
    
    # Target coefficients
    c1 = 0.5
    c2 = 0.8
    y_obs = y_classical + c1 * x + c2 * (x**2)
    
    # We build our independent variable dictionary
    X = {"x": x}
    
    # Instantiate the orchestrator with max 3 rounds
    orchestrator = iADCDOrchestrator(max_rounds=2, convergence_nmse=1e-5, min_snr=0.1, verbose=True)
    
    # Run iADCD
    res = orchestrator.run_iterative_discovery(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable="x",
        limit_direction="0",
        classical_expr="x**3",
        proposer="mock",
        seed=42
    )
    
    # Check that we completed at least one round and reached low NMSE
    assert len(res.rounds) > 0
    assert res.final_nmse_full < 1e-2
    assert "x" in res.final_expr
