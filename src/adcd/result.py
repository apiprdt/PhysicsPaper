import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, Any, Optional

from adcd.correction_orchestrator import CorrectionSearchResult
from adcd.display import (
    r_squared, fit_quality, r2_bar, pretty_expr, latex_expr,
    format_gate_funnel, html_r2_bar, html_badge,
    _top, _bottom, _thick_hline, _hline, _row, _empty, BOX_WIDTH,
)


class ADCDResult:
    """
    Result wrapper for an ADCD correction discovery run.

    Provides a unified summary with two integrated layers:
      • Physicist layer  — plain-language quality, formula, parameters
      • Developer layer  — gate stats, NMSE, BIC, timing (audit-transparent)

    Usage
    -----
    result = adcd.fit(X, y_obs, y_classical, ...)
    print(result.summary())          # full output (both layers)
    print(result.summary(brief=True))  # physicist layer only
    result.plot_residuals()          # residual + theory comparison plot
    result.show_candidates()         # ranked candidate list
    """

    def __init__(
        self,
        search_result: CorrectionSearchResult,
        scenario: Any,
        X: Dict[str, np.ndarray],
        y_obs: np.ndarray,
        y_classical: np.ndarray,
    ):
        self.search_result = search_result
        self.scenario = scenario
        self.X = X
        self.y_obs = y_obs
        self.y_classical = y_classical

        if self.scenario.correction_type == "multiplicative":
            safe_classical = np.where(self.y_classical == 0, 1e-15, self.y_classical)
            self.residual = self.y_obs / safe_classical - 1.0
        else:
            self.residual = self.y_obs - self.y_classical

    # ─────────────────────────────────────────────────────────────────────────
    # Core properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def best_expr(self) -> str:
        """Symbolic expression of the best discovered correction term."""
        return self.search_result.best_expr

    @property
    def best_theta(self) -> Dict[str, float]:
        """Optimized parameter values (theta_0, theta_1, ...)."""
        return self.search_result.best_theta

    @property
    def best_nmse_residual(self) -> float:
        """Residual NMSE — primary fit quality metric (lower is better)."""
        return self.search_result.best_nmse_residual

    @property
    def best_nmse_full(self) -> float:
        """Full-model NMSE — diagnostic only, not used in success criterion."""
        return self.search_result.best_nmse_full

    @property
    def converged(self) -> bool:
        """True if the search converged below the NMSE threshold."""
        return self.search_result.converged

    @property
    def r_squared(self) -> float:
        """Variance explained by the discovered correction (0–1)."""
        return r_squared(self.best_nmse_residual)

    @property
    def fit_quality_label(self) -> str:
        """Plain-language fit quality: Excellent / Good / Acceptable / Poor."""
        return fit_quality(self.best_nmse_residual)[0]

    @property
    def latex(self) -> str:
        """LaTeX string for the discovered correction term."""
        return latex_expr(self.best_expr)

    # ─────────────────────────────────────────────────────────────────────────
    # summary() — unified dual-layer terminal output
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self, brief: bool = False) -> str:
        """
        Formatted summary combining physicist and developer layers.

        Parameters
        ----------
        brief : bool
            If True, show only the physicist layer (formula + quality).
            Default False shows the complete output.
        """
        W = BOX_WIDTH
        ev = self.search_result.evaluation
        ql, ql_sym = fit_quality(self.best_nmse_residual)
        r2_pct = self.r_squared * 100
        bar = r2_bar(self.best_nmse_residual)
        n_iter = len(self.search_result.history)
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        status_sym = "✓" if self.converged else "✗"

        lines = []
        lines.append(_top(W))
        lines.append(_row(f"ADCD  ·  Correction Discovery Results", W))
        lines.append(_thick_hline(W))

        # ── Scenario info ────────────────────────────────────────────────────
        lines.append(_row(f"Scenario   : {self.scenario.name}", W))
        lines.append(_row(f"Domain     : {self.scenario.domain}", W))
        lines.append(_row(
            f"Type       : {self.scenario.correction_type}  "
            f"│  Limit: {self.scenario.classical_limit_variable} → "
            f"{self.scenario.classical_limit_direction}", W
        ))
        conv_label = f"{status_sym} {status}  (iteration {n_iter} of "
        if self.search_result.history:
            # max_iterations is not stored; infer from search result
            conv_label += f"{n_iter})"
        else:
            conv_label += "?)"
        lines.append(_row(f"Status     : {conv_label}", W))

        # ── Discovered correction ────────────────────────────────────────────
        lines.append(_thick_hline(W))
        lines.append(_row("DISCOVERED CORRECTION", W))
        lines.append(_hline(width=W))
        lines.append(_row(f"  Δ = {self.best_expr}", W))
        if self.latex:
            lines.append(_row(f"  LaTeX: \\Delta = {self.latex}", W))
        if self.best_theta:
            lines.append(_row(f"  Substituted: {pretty_expr(self.best_expr, self.best_theta)}", W))

        # ── Parameters ───────────────────────────────────────────────────────
        if self.best_theta:
            lines.append(_hline(width=W))
            lines.append(_row("FITTED PARAMETERS", W))
            for k, v in self.best_theta.items():
                lines.append(_row(f"  {k} = {v:.8g}", W))

        # ── Fit quality (physicist layer) ────────────────────────────────────
        lines.append(_thick_hline(W))
        lines.append(_row("FIT QUALITY", W))
        lines.append(_hline(width=W))
        lines.append(_row(f"  Variance explained  : {bar}", W))
        lines.append(_row(f"  Quality             : {ql}  {ql_sym}", W))
        lines.append(_row(f"  Success criterion   : NMSE_res < 0.20  ->  {'PASS v' if self.best_nmse_residual < 0.20 else 'FAIL x'}", W))

        if not self.converged:
            lines.append(_hline(width=W))
            lines.append(_row("  ! Search did not fully converge.", W))
            lines.append(_row("    The correction above is the best candidate found so far.", W))
            lines.append(_row("    -> Try: increase max_iterations or n_data_points", W))

        if brief:
            lines.append(_bottom(W))
            return "\n".join(lines)

        # ── Developer / audit layer ──────────────────────────────────────────
        lines.append(_thick_hline(W))
        lines.append(_row("PIPELINE STATISTICS  [developer / audit]", W))
        lines.append(_hline(width=W))

        # Gate funnel
        proposed = self.search_result.total_candidates_proposed
        survived = self.search_result.total_candidates_survived_stage1
        optimized = getattr(self.search_result, "total_candidates_optimized", "—")
        lines.extend(format_gate_funnel(proposed, survived, optimized, W))

        lines.append(_hline(width=W))
        lines.append(_row(f"  NMSE_res  : {self.best_nmse_residual:.6e}  (success if < 0.20)", W))
        lines.append(_row(f"  NMSE_full : {self.best_nmse_full:.6e}  (diagnostic, not used in pass/fail)", W))
        if ev:
            lines.append(_row(f"  BIC score : {ev.bic:.4f}  (lower = fewer params for same fit)", W))
            disc_cls = ev.discovered_class or "—"
            true_cls = ev.true_class or "unknown"
            match_sym = "✓" if ev.class_match else "✗"
            lines.append(_row(f"  Struct. match : {match_sym}  discovered={disc_cls}  │  ground truth={true_cls}", W))
        lines.append(_row(f"  Runtime   : {self.search_result.total_time_seconds:.3f} s", W))

        lines.append(_bottom(W))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        ql = self.fit_quality_label
        return (
            f"ADCDResult(expr='{self.best_expr}', "
            f"R²={self.r_squared:.4f}, quality='{ql}', "
            f"converged={self.converged})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # show_candidates() — ranked candidate table
    # ─────────────────────────────────────────────────────────────────────────

    def show_candidates(self, top_k: int = 5) -> Any:
        """
        Display the top-k candidate expressions ranked by residual NMSE.
        Returns HTML table in Jupyter; prints plain text in terminal.
        """
        all_candidates = []
        seen = set()
        for step in self.search_result.history:
            if hasattr(step, "top_5") and step.top_5:
                for expr, nmse in step.top_5:
                    if expr not in seen:
                        seen.add(expr)
                        all_candidates.append((expr, nmse))

        all_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]

        try:
            from IPython.display import HTML
            rows = ""
            for idx, (expr, nmse) in enumerate(all_candidates, 1):
                ql, _ = fit_quality(nmse)
                r2 = r_squared(nmse)
                bg = "#ffffff" if idx % 2 == 0 else "#f9f9f9"
                star = "⭐ " if idx == 1 else ""
                rows += (
                    f"<tr style='background:{bg};'>"
                    f"<td style='padding:8px;border:1px solid #ddd;text-align:center;'>{star}{idx}</td>"
                    f"<td style='padding:8px;border:1px solid #ddd;font-family:monospace;'>{expr}</td>"
                    f"<td style='padding:8px;border:1px solid #ddd;text-align:right;font-family:monospace;'>{nmse:.4e}</td>"
                    f"<td style='padding:8px;border:1px solid #ddd;text-align:right;'>{r2*100:.1f}%</td>"
                    f"<td style='padding:8px;border:1px solid #ddd;'>{ql}</td>"
                    f"</tr>"
                )
            html = (
                "<table style='border-collapse:collapse;width:100%;font-family:sans-serif;font-size:0.9em;'>"
                "<tr style='background:#f0f0f0;font-weight:bold;'>"
                "<th style='padding:8px;border:1px solid #ddd;'>Rank</th>"
                "<th style='padding:8px;border:1px solid #ddd;'>Expression</th>"
                "<th style='padding:8px;border:1px solid #ddd;'>NMSE_res</th>"
                "<th style='padding:8px;border:1px solid #ddd;'>R²</th>"
                "<th style='padding:8px;border:1px solid #ddd;'>Quality</th>"
                "</tr>"
                + rows
                + "</table>"
            )
            return HTML(html)
        except ImportError:
            header = f"{'Rank':<5}  {'Expression':<50}  {'NMSE_res':<14}  {'R²':>7}  Quality"
            sep = "─" * len(header)
            print(header)
            print(sep)
            for idx, (expr, nmse) in enumerate(all_candidates, 1):
                ql, _ = fit_quality(nmse)
                r2 = r_squared(nmse)
                star = "★" if idx == 1 else " "
                print(f"{star}{idx:<4}  {expr:<50}  {nmse:<14.4e}  {r2*100:>6.1f}%  {ql}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # export_latex()
    # ─────────────────────────────────────────────────────────────────────────

    def export_latex(self) -> str:
        """LaTeX string for the discovered correction term."""
        if not self.best_expr:
            return "No correction discovered."
        latex = self.latex
        return f"\\Delta = {latex}" if latex else self.best_expr

    # ─────────────────────────────────────────────────────────────────────────
    # plot_residuals()
    # ─────────────────────────────────────────────────────────────────────────

    def plot_residuals(self, save_path: Optional[str] = None):
        """
        Two-panel visualization:
          Left  — Observed residual vs. independent variable, overlaid with ADCD fit
          Right — Classical theory vs. ADCD-corrected theory vs. observed data
        """
        primary_var = self.scenario.classical_limit_variable
        if primary_var not in self.X:
            primary_var = list(self.X.keys())[0]

        x_vals = self.X[primary_var]
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]

        eval_dict = {k: v[sort_idx] for k, v in self.X.items()}
        for k, v in self.scenario.classical_constants.items():
            eval_dict[k] = v
        if self.best_theta:
            for k, v in self.best_theta.items():
                eval_dict[k] = v

        try:
            expr_sym = sp.sympify(self.best_expr)
            free_syms = [str(s) for s in expr_sym.free_symbols]
            sym_args = [s for s in free_syms if s in eval_dict]
            f_lamb = sp.lambdify([sp.Symbol(s) for s in sym_args], expr_sym, modules=["numpy"])
            arg_vals = [eval_dict[s] for s in sym_args]
            pred_correction = f_lamb(*arg_vals)
            if np.isscalar(pred_correction):
                pred_correction = np.full_like(x_sorted, float(pred_correction))
        except Exception:
            pred_correction = np.zeros_like(x_sorted)

        ql, _ = fit_quality(self.best_nmse_residual)
        r2_pct = self.r_squared * 100

        # ── Figure ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=110)
        fig.suptitle(
            f"ADCD — {self.scenario.name}  │  {ql}  (R² = {r2_pct:.1f}%)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        # Left: residual fit
        axes[0].scatter(
            x_vals, self.residual,
            color="#1f77b4", alpha=0.55, s=20, edgecolors="none",
            label="Observed residual δ",
        )
        axes[0].plot(
            x_sorted, pred_correction,
            color="#d62728", linewidth=2.2,
            label=f"ADCD: {self.best_expr}",
        )
        axes[0].set_xlabel(primary_var, fontsize=11)
        axes[0].set_ylabel("Anomaly Δ", fontsize=11)
        axes[0].set_title("Residual Fit", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend(fontsize=9, framealpha=0.9)

        # Annotation: NMSE
        axes[0].annotate(
            f"NMSE_res = {self.best_nmse_residual:.2e}",
            xy=(0.03, 0.96), xycoords="axes fraction",
            fontsize=8, va="top", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        # Right: theory comparison
        y_obs_sorted = self.y_obs[sort_idx]
        y_classical_sorted = self.y_classical[sort_idx]
        if self.scenario.correction_type == "multiplicative":
            y_corrected = y_classical_sorted * (1.0 + pred_correction)
        else:
            y_corrected = y_classical_sorted + pred_correction

        axes[1].scatter(
            x_vals, self.y_obs,
            color="#2ca02c", alpha=0.45, s=20, edgecolors="none",
            label="Observed data",
        )
        axes[1].plot(
            x_sorted, y_classical_sorted,
            color="#7f7f7f", linestyle="--", linewidth=1.5,
            label="Classical theory",
        )
        axes[1].plot(
            x_sorted, y_corrected,
            color="#d62728", linewidth=2.0,
            label="ADCD corrected",
        )
        axes[1].set_xlabel(primary_var, fontsize=11)
        axes[1].set_ylabel("Observable y", fontsize=11)
        axes[1].set_title("Theory Comparison", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(fontsize=9, framealpha=0.9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif plt.isinteractive():
            plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Jupyter rich repr
    # ─────────────────────────────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        """Rich HTML for Jupyter Notebook display."""
        ev = self.search_result.evaluation
        status_color = "#28a745" if self.converged else "#dc3545"
        status_text = "CONVERGED" if self.converged else "NOT CONVERGED"
        ql, _ = fit_quality(self.best_nmse_residual)
        r2_html = html_r2_bar(self.best_nmse_residual)
        disc_cls = (ev.discovered_class or "—") if ev else "—"
        true_cls = (ev.true_class or "—") if ev else "—"
        match_badge = (
            html_badge("MATCH ✓", "#28a745") if (ev and ev.class_match)
            else html_badge("NO MATCH ✗", "#dc3545")
        ) if ev else ""

        # Parameters table
        param_rows = ""
        if self.best_theta:
            for k, v in self.best_theta.items():
                param_rows += f"<tr><td style='padding:4px 8px;color:#555;'><code>{k}</code></td><td style='padding:4px 8px;font-family:monospace;'>{v:.8g}</td></tr>"
            param_section = (
                "<div style='margin:12px 0;'><strong>Parameters</strong>"
                "<table style='margin-top:6px;border-collapse:collapse;font-size:0.9em;'>"
                + param_rows + "</table></div>"
            )
        else:
            param_section = ""

        # Developer details (collapsible)
        proposed = self.search_result.total_candidates_proposed
        survived = self.search_result.total_candidates_survived_stage1
        optimized = getattr(self.search_result, "total_candidates_optimized", "—")
        bic_str = f"{ev.bic:.4f}" if ev else "N/A"
        dev_details = (
            "<details style='margin-top:12px;font-size:0.85em;color:#555;'>"
            "<summary style='cursor:pointer;color:#007bff;font-weight:600;'>Developer / Audit Details ▸</summary>"
            "<div style='padding:10px;background:#f8f9fa;border-radius:4px;margin-top:6px;'>"
            "<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='padding:3px 8px;'>Candidates proposed</td><td style='padding:3px 8px;font-family:monospace;'>{proposed}</td></tr>"
            f"<tr><td style='padding:3px 8px;'>Passed Stage 1 gates</td><td style='padding:3px 8px;font-family:monospace;'>{survived}</td></tr>"
            f"<tr><td style='padding:3px 8px;'>Sent to optimizer</td><td style='padding:3px 8px;font-family:monospace;'>{optimized}</td></tr>"
            f"<tr><td style='padding:3px 8px;'>NMSE_res</td><td style='padding:3px 8px;font-family:monospace;'>{self.best_nmse_residual:.6e}  <span style='color:#888;font-size:0.85em;'>(success: &lt; 0.20)</span></td></tr>"
            f"<tr><td style='padding:3px 8px;'>NMSE_full</td><td style='padding:3px 8px;font-family:monospace;'>{self.best_nmse_full:.6e}  <span style='color:#888;font-size:0.85em;'>(diagnostic only)</span></td></tr>"
            f"<tr><td style='padding:3px 8px;'>BIC score</td><td style='padding:3px 8px;font-family:monospace;'>{bic_str}</td></tr>"
            f"<tr><td style='padding:3px 8px;'>Structural class</td><td style='padding:3px 8px;'>{disc_cls}  vs  {true_cls}  {match_badge}</td></tr>"
            f"<tr><td style='padding:3px 8px;'>Runtime</td><td style='padding:3px 8px;font-family:monospace;'>{self.search_result.total_time_seconds:.3f} s</td></tr>"
            "</table></div></details>"
        )

        latex_str = self.latex
        latex_display = (
            f"<div style='font-size:0.85em;color:#888;margin-top:4px;'>LaTeX: <code>\\Delta = {latex_str}</code></div>"
            if latex_str else ""
        )

        return (
            "<div style='border:1px solid #ddd;border-radius:8px;padding:18px;"
            "font-family:sans-serif;max-width:720px;box-shadow:0 2px 8px rgba(0,0,0,0.07);'>"

            # Header
            "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;'>"
            "<h3 style='margin:0;font-size:1.1em;color:#222;'>⚗ ADCD — Correction Discovery</h3>"
            f"<span style='color:{status_color};font-weight:700;font-size:0.9em;'>{status_text}</span>"
            "</div>"

            # Scenario info
            f"<div style='font-size:0.88em;color:#555;margin-bottom:14px;'>"
            f"<strong>Scenario:</strong> {self.scenario.name} &nbsp;│&nbsp; "
            f"<strong>Domain:</strong> {self.scenario.domain} &nbsp;│&nbsp; "
            f"<strong>Type:</strong> {self.scenario.correction_type}"
            f"</div>"

            # Discovered correction — highlighted
            "<div style='background:#f0f4ff;border-left:4px solid #4361ee;padding:12px 16px;"
            "border-radius:0 6px 6px 0;margin-bottom:14px;'>"
            "<div style='font-size:0.8em;color:#666;margin-bottom:4px;'>Discovered correction Δ</div>"
            f"<div style='font-size:1.3em;font-family:monospace;font-weight:700;color:#1a1a2e;'>{self.best_expr}</div>"
            + latex_display +
            "</div>"

            # Parameters
            + param_section +

            # Fit quality — physicist layer
            "<div style='margin-bottom:12px;'>"
            f"<div style='margin-bottom:6px;'><strong>Fit Quality:</strong> {ql}</div>"
            f"<div>Variance explained: {r2_html}</div>"
            "</div>"

            # Developer details (collapsible)
            + dev_details +

            "</div>"
        )
