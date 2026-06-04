import pytest
import sympy as sp
import numpy as np
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline

def test_complete_cascading_pipeline_structural():
    # Setup dependencies
    m, v = sp.symbols('m v')
    regimes = [
        AsymptoticRegime(variable=v, limit_target=0, ground_truth_expr="0", weight=1.0)
    ]
    
    validator = ASTValidator(max_depth=5, max_tokens=15)
    checker = DimensionalChecker()
    scorer = ARCScorer(regimes=regimes)
    
    pipeline = Stage1Pipeline(validator, checker, scorer)
    
    # Mix of pristine, dimensionally invalid, and complex bloated formulas
    candidates = [
        "0.5 * m * v**2",               # Perfect candidate
        "m * v",                        # Dimension mismatch (Momentum instead of Energy)
        "m + r",                        # Unit clash crash
        "0.5*m*v**2 + m*v**3/c - r/t",  # Bloated equation
    ]
    
    results = pipeline.execute(candidates, target_dimension_key="E")
    
    # Only the valid kinetic energy should break past all 3 gates
    assert len(results) == 1
    assert results[0][0] == "0.5 * m * v**2"
    assert results[0][3] == pytest.approx(1.0, abs=1e-5)  # ARC Score is index 3

def test_pipeline_empirical_and_scale_invariance():
    # 1. Setup Synthetic Noisy Dataset (Kinetic Energy)
    rng = np.random.default_rng(42)
    m_data = rng.uniform(0.1, 10.0, size=100)
    v_data = rng.uniform(0.1, 5.0, size=100)
    
    # Ground Truth: E_k = 0.5 * m * v^2 + small noise
    y_true = 0.5 * m_data * v_data**2
    noise = rng.normal(0, 0.01, size=100)
    y_obs = y_true + noise
    
    X = {"m": m_data, "v": v_data}
    
    # 2. Setup Stage 1 Pipeline components
    v = sp.Symbol('v')
    regimes = [
        AsymptoticRegime(variable=v, limit_target=0, ground_truth_expr="0", weight=1.0)
    ]
    
    validator = ASTValidator(max_depth=5, max_tokens=15)
    checker = DimensionalChecker()
    scorer = ARCScorer(regimes=regimes)
    
    pipeline = Stage1Pipeline(validator, checker, scorer)
    
    candidates = [
        "0.5 * m * v**2",                # Perfect candidate
        "0.53 * m * v**2",               # Slightly off coefficient
        "0.5 * m * v**2 + m * v**3 / c", # Dimensionally correct but fits poorly (uses high c)
    ]
    
    # Execute pipeline on original data (Joule scale)
    results_joule = pipeline.execute(candidates, target_dimension_key="E", X=X, y_obs=y_obs, beta=1.0, constants={"c": 1.0})
    
    assert len(results_joule) == 3
    # Perfect candidate must have the highest combined score (index 1)
    assert results_joule[0][0] == "0.5 * m * v**2"
    # Off coefficient should be second
    assert results_joule[1][0] == "0.53 * m * v**2"
    
    # Combined score of perfect candidate is very close to 1.0
    perfect_score_joule = results_joule[0][1]
    assert perfect_score_joule > 0.95
    
    # 3. Prove Scale Invariance: Multiply observed target by 1000 (milli-Joule scale)
    y_milli = y_obs * 1000
    
    # We must also multiply variable m or scale appropriately, but wait!
    # If the candidate expression is "0.5 * m * v**2" and we evaluate it on the same X,
    # the predictions are in Joules, but y_milli is in milli-Joules. This would cause high MSE.
    # To compare same units, if y_obs is multiplied by 1000, we should multiply the candidate coefficient
    # (or variable arrays) appropriately to predict milli-Joules, or we scale the input.
    # Actually, a much cleaner way to show scale-invariance is to scale BOTH the dataset variables
    # (e.g. changing mass unit from kg to g, which multiplies mass by 1000) and target y (multiplied by 1000).
    # Let's see: if m is in grams, m_g = m_data * 1000.
    # If the target is in milli-Joules, y_milli = y_obs * 1000.
    # The candidate "0.0005 * m * v**2" (which accounts for mass in grams to predict milli-Joules) should have
    # EXACTLY the same combined score as "0.5 * m * v**2" on original data!
    # Let's verify this mathematically:
    # y_pred_milli = 0.0005 * m_g * v**2 = 0.0005 * (m_data * 1000) * v**2 = 0.5 * m_data * v**2 = y_pred_joule.
    # y_obs_milli = y_obs * 1000.
    # So y_pred_milli - y_obs_milli = 1000 * (y_pred_joule - y_obs_joule).
    # Thus, MSE_milli = 1,000,000 * MSE_joule.
    # But Var(y_milli) = 1,000,000 * Var(y_obs).
    # Therefore, NMSE_milli = MSE_milli / Var(y_milli) = NMSE_joule!
    # This means Combined Score remains EXACTLY identical! Let's test this beautifully!
    
    X_milli = {"m": m_data * 1000.0, "v": v_data}
    candidates_milli = [
        "0.5 * m * v**2"
    ]
    
    results_milli = pipeline.execute(candidates_milli, target_dimension_key="E", X=X_milli, y_obs=y_milli, beta=1.0, constants={"c": 1.0})
    
    perfect_score_milli = results_milli[0][1]
    
    # Combined scores must be virtually identical (scale-invariant NMSE)
    assert perfect_score_milli == pytest.approx(perfect_score_joule, abs=1e-7)


def test_pipeline_no_target_dimension():
    # Setup dependencies
    m, v = sp.symbols('m v')
    regimes = [
        AsymptoticRegime(variable=v, limit_target=0, ground_truth_expr="0", weight=1.0)
    ]
    
    validator = ASTValidator(max_depth=5, max_tokens=15)
    checker = DimensionalChecker()
    scorer = ARCScorer(regimes=regimes)
    
    pipeline = Stage1Pipeline(validator, checker, scorer)
    
    # When target_dimension_key is None, dimensional checks are bypassed
    candidates = [
        "0.5 * m * v**2",
        "m * v",
    ]
    
    results = pipeline.execute(candidates, target_dimension_key=None)
    
    # Both candidates should pass because dimensional check is bypassed
    assert len(results) == 2
    assert {r[0] for r in results} == {"0.5 * m * v**2", "m * v"}
