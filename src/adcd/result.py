import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, Any, Optional
from adcd.correction_orchestrator import CorrectionSearchResult

class ADCDResult:
    """
    A scientist-friendly result wrapper for ADCD correction discovery runs.
    
    Provides rich visualization, Jupyter Notebook rendering, LaTeX export,
    and structured summaries.
    """
    def __init__(self, search_result: CorrectionSearchResult, scenario: Any, X: Dict[str, np.ndarray], y_obs: np.ndarray, y_classical: np.ndarray):
        self.search_result = search_result
        self.scenario = scenario
        self.X = X
        self.y_obs = y_obs
        self.y_classical = y_classical
        
        # Calculate residual
        if self.scenario.correction_type == "multiplicative":
            safe_classical = np.where(self.y_classical == 0, 1e-15, self.y_classical)
            self.residual = self.y_obs / safe_classical - 1.0
        else:
            self.residual = self.y_obs - self.y_classical

    @property
    def best_expr(self) -> str:
        """The symbolic expression of the best discovered correction term."""
        return self.search_result.best_expr

    @property
    def best_theta(self) -> Dict[str, float]:
        """The optimized parameter values (theta_0, theta_1, etc.)."""
        return self.search_result.best_theta

    @property
    def best_nmse_residual(self) -> float:
        """Normalized Mean Squared Error on the residual data."""
        return self.search_result.best_nmse_residual

    @property
    def best_nmse_full(self) -> float:
        """Normalized Mean Squared Error of the full corrected model on observed data."""
        return self.search_result.best_nmse_full

    @property
    def converged(self) -> bool:
        """Whether the search converged below the target threshold."""
        return self.search_result.converged

    def summary(self) -> str:
        """Returns a formatted plain-text summary of the discovery run."""
        lines = [
            "==================================================",
            "          ADCD DISCOVERY RUN SUMMARY              ",
            "==================================================",
            f"Scenario Name:      {self.scenario.name}",
            f"Domain:             {self.scenario.domain}",
            f"Correction Type:    {self.scenario.correction_type}",
            f"Asymptotic Regime:  {self.scenario.classical_limit_variable} -> {self.scenario.classical_limit_direction}",
            f"Convergence Status: {'CONVERGED' if self.converged else 'NOT CONVERGED'}",
            "--------------------------------------------------",
            f"Discovered Correction: {self.best_expr}",
            "Optimized Parameters (theta):"
        ]
        if self.best_theta:
            for k, v in self.best_theta.items():
                lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append("  None")
        
        lines.extend([
            "Metrics:",
            f"  Residual NMSE:   {self.best_nmse_residual:.6e}",
            f"  Full Model NMSE: {self.best_nmse_full:.6e}",
            f"  BIC Score:       {self.search_result.evaluation.bic:.2f}" if self.search_result.evaluation else "  BIC Score:       N/A",
            "Search Statistics:",
            f"  Iterations:      {len(self.search_result.history)}",
            f"  Total proposed:  {self.search_result.total_candidates_proposed}",
            f"  Survived Stage 1:{self.search_result.total_candidates_survived_stage1}",
            f"  Execution time:  {self.search_result.total_time_seconds:.2f} seconds",
            "=================================================="
        ])
        return "\n".join(lines)

    def export_latex(self) -> str:
        """Exports the best discovered correction term as a LaTeX string."""
        if not self.best_expr:
            return "No correction discovered."
        
        try:
            expr = sp.sympify(self.best_expr)
            # Replace theta_N with their optimized values in LaTeX representation if requested,
            # but usually scientists want the parameterized LaTeX form with a legend.
            # Let's export the parameterized formula.
            latex_str = sp.latex(expr)
            return f"\\Delta = {latex_str}"
        except Exception as e:
            return f"Error formatting LaTeX: {str(e)}"

    def show_candidates(self, top_k: int = 5) -> Any:
        """Prints or displays the top candidate equations from the search history."""
        # Collate all optimized candidates across all iterations
        all_candidates = []
        seen = set()
        
        for step in self.search_result.history:
            if hasattr(step, 'top_5') and step.top_5:
                for expr, nmse in step.top_5:
                    if expr not in seen:
                        seen.add(expr)
                        all_candidates.append((expr, nmse))
        
        # Sort by NMSE ascending
        all_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
        
        # Detect environment: return HTML table in Jupyter, otherwise print plain text
        try:
            from IPython.display import HTML
            html_content = [
                "<table style='border-collapse: collapse; width: 100%; font-family: sans-serif;'>",
                "<tr style='background-color: #f2f2f2; text-align: left;'>",
                "<th style='padding: 8px; border: 1px solid #ddd;'>Rank</th>",
                "<th style='padding: 8px; border: 1px solid #ddd;'>Candidate Expression</th>",
                "<th style='padding: 8px; border: 1px solid #ddd;'>Residual NMSE</th>",
                "</tr>"
            ]
            for idx, (expr, nmse) in enumerate(all_candidates, 1):
                bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
                html_content.append(
                    f"<tr style='background-color: {bg};'>"
                    f"<td style='padding: 8px; border: 1px solid #ddd;'>{idx}</td>"
                    f"<td style='padding: 8px; border: 1px solid #ddd;'><code>{expr}</code></td>"
                    f"<td style='padding: 8px; border: 1px solid #ddd;'>{nmse:.6e}</td>"
                    f"</tr>"
                )
            html_content.append("</table>")
            return HTML("".join(html_content))
        except ImportError:
            # Fallback to plain text
            lines = [
                f"{'Rank':<5} | {'Candidate Expression':<50} | {'Residual NMSE':<15}",
                "-" * 76
            ]
            for idx, (expr, nmse) in enumerate(all_candidates, 1):
                lines.append(f"{idx:<5} | {expr:<50} | {nmse:.6e}")
            print("\n".join(lines))
            return None

    def plot_residuals(self, save_path: Optional[str] = None):
        """
        Plots a high-quality visualization of the discovery:
        1. Residual vs. Independent Variable, overlaid with the best-fit correction.
        2. Observed vs. Corrected Model.
        """
        primary_var = self.scenario.classical_limit_variable
        if primary_var not in self.X:
            # Fallback if the limit variable is not in X (e.g. constant/scalar scenarios)
            primary_var = list(self.X.keys())[0]

        x_vals = self.X[primary_var]
        
        # Sort values for smooth plotting curves
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        self.residual[sort_idx]  # noqa: ensure sorted residual exists
        
        # Evaluate discovered correction on sorted data
        # Build evaluation dict
        eval_dict = {k: v[sort_idx] for k, v in self.X.items()}
        for k, v in self.scenario.classical_constants.items():
            eval_dict[k] = v
        if self.best_theta:
            for k, v in self.best_theta.items():
                eval_dict[k] = v

        try:
            # Safely evaluate correction expression
            # Use sympy to lambdify the expression to avoid unsafe eval
            expr = sp.sympify(self.best_expr)
            free_symbols = [str(s) for s in expr.free_symbols]
            
            # Filter symbols that are in eval_dict
            sym_args = [s for s in free_symbols if s in eval_dict]
            # Create a lambda function
            f_lamb = sp.lambdify([sp.Symbol(s) for s in sym_args], expr, modules=["numpy"])
            
            # Call lambda
            arg_vals = [eval_dict[s] for s in sym_args]
            pred_correction = f_lamb(*arg_vals)
            
            # Broadcast scalar predictions if needed
            if np.isscalar(pred_correction):
                pred_correction = np.full_like(x_sorted, pred_correction)
        except Exception:
            # Fallback to direct eval if lambdify fails
            try:
                local_env = {**eval_dict, "np": np, "sp": sp, "exp": np.exp, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "log": np.log, "pi": np.pi}
                pred_correction = eval(self.best_expr, {"__builtins__": None}, local_env)
                if np.isscalar(pred_correction):
                    pred_correction = np.full_like(x_sorted, pred_correction)
            except Exception:
                pred_correction = np.zeros_like(x_sorted)

        # Build figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100)
        
        # Plot 1: Residual & fit
        axes[0].scatter(x_vals, self.residual, color='#1f77b4', alpha=0.6, edgecolors='none', label='Observed Anomaly (Residual)')
        axes[0].plot(x_sorted, pred_correction, color='#d62728', linewidth=2.5, label=f'ADCD Correction: {self.best_expr}')
        axes[0].set_xlabel(f"Independent Variable ({primary_var})", fontsize=11)
        axes[0].set_ylabel("Anomaly Residual $\\delta$", fontsize=11)
        axes[0].set_title("Residual Fitting", fontsize=12, fontweight='bold', pad=12)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
        
        # Plot 2: Observed vs Classical vs Corrected
        y_obs_sorted = self.y_obs[sort_idx]  # noqa: used for plotting alignment
        y_classical_sorted = self.y_classical[sort_idx]
        
        # Compute corrected model prediction
        if self.scenario.correction_type == "multiplicative":
            y_corrected_sorted = y_classical_sorted * (1.0 + pred_correction)
        else:
            y_corrected_sorted = y_classical_sorted + pred_correction
            
        axes[1].scatter(x_vals, self.y_obs, color='#2ca02c', alpha=0.5, edgecolors='none', label='Observed Data')
        axes[1].plot(x_sorted, y_classical_sorted, color='#7f7f7f', linestyle='--', linewidth=1.5, label='Classical Theory')
        axes[1].plot(x_sorted, y_corrected_sorted, color='#d62728', linewidth=2.0, label='ADCD Corrected Theory')
        axes[1].set_xlabel(f"Independent Variable ({primary_var})", fontsize=11)
        axes[1].set_ylabel("Observable $y$", fontsize=11)
        axes[1].set_title("Theory Comparison", fontsize=12, fontweight='bold', pad=12)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def _repr_html_(self) -> str:
        """IPython/Jupyter notebook rich representation."""
        status_color = "#28a745" if self.converged else "#dc3545"
        status_text = "CONVERGED" if self.converged else "NOT CONVERGED"
        
        html = [
            "<div style='border: 1px solid #ddd; border-radius: 8px; padding: 16px; font-family: sans-serif; max-width: 800px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>",
            "  <h3 style='margin-top: 0; color: #333;'>ADCD Correction Discovery Results</h3>",
            "  <div style='display: flex; gap: 20px; margin-bottom: 16px;'>",
            f"    <div><strong>Scenario:</strong> {self.scenario.name}</div>",
            f"    <div><strong>Domain:</strong> {self.scenario.domain}</div>",
            f"    <div><strong>Status:</strong> <span style='color: {status_color}; font-weight: bold;'>{status_text}</span></div>",
            "  </div>",
            "  <div style='background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 12px; margin-bottom: 16px; border-radius: 0 4px 4px 0;'>",
            "    <div style='font-size: 0.9em; color: #666;'>Discovered Correction (\\(\\Delta\\)):</div>",
            f"    <div style='font-size: 1.25em; font-family: monospace; font-weight: bold; margin-top: 4px;'>{self.best_expr}</div>",
            "  </div>"
        ]
        
        if self.best_theta:
            html.append("  <div style='margin-bottom: 16px;'>")
            html.append("    <strong>Parameters:</strong>")
            html.append("    <ul style='margin: 4px 0; padding-left: 20px;'>")
            for k, v in self.best_theta.items():
                html.append(f"      <li><code>{k}</code> = {v:.6f}</li>")
            html.append("    </ul>")
            html.append("  </div>")
            
        html.extend([
            "  <table style='width: 100%; border-collapse: collapse; margin-bottom: 8px;'>",
            "    <tr style='border-bottom: 1px solid #eee;'>",
            "      <td style='padding: 6px 0; color: #555;'>Residual NMSE:</td>",
            f"      <td style='padding: 6px 0; text-align: right; font-family: monospace;'>{self.best_nmse_residual:.6e}</td>",
            "    </tr>",
            "    <tr style='border-bottom: 1px solid #eee;'>",
            "      <td style='padding: 6px 0; color: #555;'>Full Model NMSE:</td>",
            f"      <td style='padding: 6px 0; text-align: right; font-family: monospace;'>{self.best_nmse_full:.6e}</td>",
            "    </tr>",
            "    <tr>",
            "      <td style='padding: 6px 0; color: #555;'>Execution Time:</td>",
            f"      <td style='padding: 6px 0; text-align: right; font-family: monospace;'>{self.search_result.total_time_seconds:.2f} s</td>",
            "    </tr>",
            "  </table>",
            "</div>"
        ])
        return "\n".join(html)
