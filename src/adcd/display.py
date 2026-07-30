"""
display.py
==========
Formatting utilities for ADCD terminal and HTML output.

Design principles:
  - Physicist layer:  clean, plain-language, no jargon
  - Developer layer:  complete, honest, audit-transparent
  - One unified output — smart ordering, not two separate calls
"""

from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BOX_WIDTH = 76  # terminal column width for box drawing


# ─────────────────────────────────────────────────────────────────────────────
# Fit quality helpers
# ─────────────────────────────────────────────────────────────────────────────

def r_squared(nmse_residual: float) -> float:
    """
    Convert residual NMSE to R² (variance explained fraction).
    R² = 1 - NMSE_res, clamped to [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - nmse_residual))


def fit_quality(nmse_residual: float) -> Tuple[str, str]:
    """
    Return (label, symbol) based on residual NMSE.

    Thresholds (based on ADCD success criterion NMSE < 0.20):
      Excellent : NMSE < 0.01   → R² > 99%
      Good      : NMSE < 0.05   → R² > 95%
      Acceptable: NMSE < 0.20   → R² > 80%  (success threshold)
      Poor      : NMSE ≥ 0.20   → R² ≤ 80%  (below threshold)
    """
    if nmse_residual < 0.01:
        return "Excellent", "✓✓"
    elif nmse_residual < 0.05:
        return "Good", "✓"
    elif nmse_residual < 0.20:
        return "Acceptable", "~"
    else:
        return "Poor (below threshold)", "X"


def r2_bar(nmse_residual: float, width: int = 20) -> str:
    """ASCII bar representing R^2 value."""
    r2 = r_squared(nmse_residual)
    filled = round(r2 * width)
    bar = "#" * filled + "." * (width - filled)
    pct = r2 * 100
    return f"[{bar}] {pct:.1f}%"


# -----------------------------------------------------------------------------
# Box drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hline(char: str = "-", width: int = BOX_WIDTH) -> str:
    return "|" + char * (width - 2) + "|"


def _top(width: int = BOX_WIDTH) -> str:
    return "+" + "=" * (width - 2) + "+"


def _bottom(width: int = BOX_WIDTH) -> str:
    return "+" + "=" * (width - 2) + "+"


def _thick_hline(width: int = BOX_WIDTH) -> str:
    return "|" + "=" * (width - 2) + "|"


def _row(text: str, width: int = BOX_WIDTH) -> str:
    """Pad text inside box borders."""
    inner = width - 4  # 2 for borders + 2 for spaces
    text = text[:inner]  # truncate if needed
    return "| " + text.ljust(inner) + " |"


def _empty(width: int = BOX_WIDTH) -> str:
    return "|" + " " * (width - 2) + "|"


def _section(title: str, width: int = BOX_WIDTH) -> str:
    """Section header row inside box."""
    inner = width - 4
    label = f"  {title.upper()}  "
    padded = label.ljust(inner)
    return "| " + padded + " |"


# ─────────────────────────────────────────────────────────────────────────────
# Gate funnel formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_gate_funnel(
    proposed: int,
    passed_stage1: int,
    optimized: int,
    width: int = BOX_WIDTH,
) -> list:
    """Returns list of formatted rows for gate funnel display."""
    def pct(n, d):
        return f"({n/d*100:.0f}%)" if d > 0 else "(n/a)"

    lines = []
    lines.append(_row(f"  Candidates proposed   : {proposed:>6}", width))
    lines.append(_row(f"  +-- Passed Stage 1    : {passed_stage1:>6}  {pct(passed_stage1, proposed)}", width))
    lines.append(_row(f"  +-- Sent to optimizer : {optimized:>6}  {pct(optimized, proposed) if isinstance(optimized, int) else ''}", width))
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Expression formatter
# ─────────────────────────────────────────────────────────────────────────────

def pretty_expr(expr_str: str, theta: Dict[str, float]) -> str:
    """
    Return expression with theta symbols replaced by formatted values.
    e.g. 'theta_0 * (v/c)**2' with {theta_0: -0.75} → '-0.75 × (v/c)²'
    Used for the physicist display layer only.
    """
    result = expr_str
    # Sort by length desc so theta_10 is replaced before theta_1
    for k in sorted(theta.keys(), key=len, reverse=True):
        v = theta[k]
        formatted = f"{v:.4g}"
        result = result.replace(k, formatted)
    return result


def latex_expr(expr_str: str) -> str:
    """Convert expression string to LaTeX via SymPy. Returns empty string on failure."""
    try:
        import sympy as sp
        sym = sp.sympify(expr_str)
        return sp.latex(sym)
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers (Jupyter)
# ─────────────────────────────────────────────────────────────────────────────

def html_r2_bar(nmse_residual: float, width_px: int = 200) -> str:
    """Return HTML progress bar representing R²."""
    r2 = r_squared(nmse_residual)
    pct = r2 * 100
    if r2 >= 0.95:
        color = "#28a745"   # green
    elif r2 >= 0.80:
        color = "#ffc107"   # amber
    else:
        color = "#dc3545"   # red

    bar = (
        f"<div style='display:inline-flex;align-items:center;gap:8px;'>"
        f"<div style='width:{width_px}px;background:#e9ecef;border-radius:4px;height:12px;overflow:hidden;'>"
        f"<div style='width:{pct:.1f}%;background:{color};height:100%;border-radius:4px;transition:width 0.3s;'></div>"
        f"</div>"
        f"<span style='font-weight:600;color:{color};'>{pct:.1f}%</span>"
        f"</div>"
    )
    return bar


def html_badge(text: str, color: str = "#007bff") -> str:
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:12px;font-size:0.75em;font-weight:600;'>{text}</span>"
    )
