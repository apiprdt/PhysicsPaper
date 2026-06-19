import time
import logging
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from adcd.anomaly_scenarios import AnomalyScenario
from adcd.api import CustomAnomalyScenario, fit
from adcd.experiments._fit import fit_with_proposer
from adcd.llm_proposer import BaseProposer
from adcd.result import ADCDResult
from adcd.metrics import evaluate_correction

logger = logging.getLogger(__name__)

@dataclass
class iADCDRoundResult:
    round_idx: int
    discovered_expr: str
    discovered_theta: Dict[str, float]
    nmse_residual: float
    nmse_full: float
    correction_mode: str
    adcd_result: ADCDResult

@dataclass
class iADCDResult:
    rounds: List[iADCDRoundResult]
    final_expr: str
    final_nmse_full: float
    converged: bool
    total_time_seconds: float


class iADCDOrchestrator:
    """
    Iterative Anomaly-Driven Correction Discovery (iADCD).
    
    Iteratively runs ADCD on the residual anomalies to build a series of corrections:
    Round 1: Discover Δ_1 from initial anomaly -> subtract correction
    Round 2: Discover Δ_2 from the residual -> subtract correction
    Round 3: Discover Δ_3 from the next residual, etc.
    """
    def __init__(
        self,
        max_rounds: int = 3,
        convergence_nmse: float = 1e-4,
        min_snr: float = 1.0,
        verbose: bool = True,
    ):
        self.max_rounds = max_rounds
        self.convergence_nmse = convergence_nmse
        self.min_snr = min_snr
        self.verbose = verbose

    def run_iterative_discovery(
        self,
        X: Dict[str, np.ndarray],
        y_obs: np.ndarray,
        y_classical: np.ndarray,
        limit_variable: Optional[str] = None,
        limit_direction: str = "0",
        classical_expr: str = "0",
        variables_with_units: Optional[Dict[str, str]] = None,
        proposer: str = "mock",
        proposer_obj: Optional[BaseProposer] = None,
        round_proposers: Optional[List[BaseProposer]] = None,
        api_key: Optional[str] = None,
        seed: int = 42,
    ) -> iADCDResult:
        start_time = time.time()
        
        rounds = []
        current_y_classical = y_classical.copy()
        
        # We start with the base classical expression
        current_expr_sym = sp.sympify(classical_expr)
        
        # Track parameters across rounds
        all_thetas: Dict[str, float] = {}
        
        converged = False
        
        for round_idx in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n=== [iADCD Round {round_idx}/{self.max_rounds}] ===")
            
            # Check noise/signal level to avoid overfitting residuals
            residual = y_obs - current_y_classical
            residual_variance = np.var(residual)
            obs_variance = np.var(y_obs)
            snr = obs_variance / max(residual_variance, 1e-15)
            
            if self.verbose:
                print(f"Current Residual NMSE: {residual_variance / max(obs_variance, 1e-15):.6f} | SNR: {snr:.2f}")
            
            if residual_variance / max(obs_variance, 1e-15) < self.convergence_nmse:
                if self.verbose:
                    print(f"Convergence threshold met: NMSE < {self.convergence_nmse}")
                converged = True
                break
                
            if snr < self.min_snr:
                if self.verbose:
                    print(f"SNR ({snr:.2f}) below threshold ({self.min_snr}). Stopping to prevent fitting noise.")
                break
            
            # Run ADCD to find correction Δ_round_idx on current baseline
            round_mode = "auto" if round_idx == 1 else "additive"
            
            round_mode = "auto" if round_idx == 1 else "additive"
            active_proposer = (
                round_proposers[round_idx - 1]
                if round_proposers and round_idx - 1 < len(round_proposers)
                else proposer_obj
            )

            if active_proposer is not None:
                adcd_res = fit_with_proposer(
                    X=X,
                    y_obs=y_obs,
                    y_classical=current_y_classical,
                    proposer_obj=active_proposer,
                    limit_variable=limit_variable,
                    limit_direction=limit_direction,
                    classical_expr=str(current_expr_sym),
                    variables_with_units=variables_with_units,
                    correction_mode=round_mode,
                    verbose=self.verbose,
                    seed=seed + round_idx,
                    scenario_name=f"iADCD round {round_idx}",
                )
            else:
                adcd_res = fit(
                    X=X,
                    y_obs=y_obs,
                    y_classical=current_y_classical,
                    limit_variable=limit_variable,
                    limit_direction=limit_direction,
                    classical_expr=str(current_expr_sym),
                    variables_with_units=variables_with_units,
                    correction_mode=round_mode,
                    proposer=proposer,
                    api_key=api_key,
                    verbose=self.verbose,
                    seed=seed + round_idx,
                )
            
            if not adcd_res.search_result.best_expr:
                if self.verbose:
                    print(f"No correction found in Round {round_idx}.")
                break
                
            # Discovered expression
            best_expr = adcd_res.search_result.best_expr
            best_theta = adcd_res.search_result.best_theta
            
            # Re-key theta parameters to prevent conflicts across rounds
            # e.g., theta_0 -> theta_r1_0
            round_theta = {}
            renamed_expr = best_expr
            for k, val in best_theta.items():
                if k.startswith("theta_"):
                    new_key = f"theta_r{round_idx}_{k[6:]}"
                    round_theta[new_key] = val
                    renamed_expr = renamed_expr.replace(k, new_key)
                else:
                    round_theta[k] = val
            
            all_thetas.update(round_theta)
            
            # Evaluate new predictions
            actual_mode = adcd_res.scenario.correction_type
            
            # Reconstruct the symbolic expression
            disc_sym = sp.sympify(renamed_expr)
            if actual_mode == "multiplicative":
                current_expr_sym = current_expr_sym * (1 + disc_sym)
            else:
                current_expr_sym = current_expr_sym + disc_sym
                
            # Evaluate predictions numerically using the fitted parameters
            local_dict = {**X}
            for k, v in all_thetas.items():
                local_dict[k] = v
                
            # Compile mathematical function using sympy to compute predictions
            free_syms = list(current_expr_sym.free_symbols)
            expr_lambdified = sp.lambdify([sp.Symbol(s.name) for s in free_syms], current_expr_sym, "numpy")
            
            # Calculate new classical predictions
            args = []
            for s in free_syms:
                if s.name in local_dict:
                    args.append(local_dict[s.name])
                else:
                    args.append(1.0) # default fallback
                    
            current_y_classical = expr_lambdified(*args)
            if isinstance(current_y_classical, (int, float)):
                current_y_classical = np.full_like(y_obs, current_y_classical)
                
            # Compute updated residual and full NMSE
            new_res_var = np.var(y_obs - current_y_classical)
            nmse_res = new_res_var / max(obs_variance, 1e-15)
            
            round_res = iADCDRoundResult(
                round_idx=round_idx,
                discovered_expr=renamed_expr,
                discovered_theta=round_theta,
                nmse_residual=adcd_res.search_result.best_nmse_residual,
                nmse_full=nmse_res,
                correction_mode=actual_mode,
                adcd_result=adcd_res
            )
            rounds.append(round_res)
            
            if self.verbose:
                print(f"Round {round_idx} Discovered: {renamed_expr}")
                print(f"Round {round_idx} Discovered Theta: {round_theta}")
                print(f"Accumulated Expression: {current_expr_sym}")
                print(f"Updated Full NMSE: {nmse_res:.6f}")
                
            if nmse_res < self.convergence_nmse:
                if self.verbose:
                    print(f"Full convergence reached: Full NMSE {nmse_res:.6e} < {self.convergence_nmse:.6e}")
                converged = True
                break
                
        total_time = time.time() - start_time
        final_nmse_full = np.var(y_obs - current_y_classical) / max(np.var(y_obs), 1e-15)
        
        return iADCDResult(
            rounds=rounds,
            final_expr=str(current_expr_sym),
            final_nmse_full=final_nmse_full,
            converged=converged,
            total_time_seconds=total_time
        )
