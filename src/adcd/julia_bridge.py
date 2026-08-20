"""
julia_bridge.py
===============
Python interface to the ADCD Julia engine (ADCDEngine.jl).

Usage:
    from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig

    engine = ADCDJuliaEngine()
    result = engine.run(config, data)

Requires: juliacall  (pip install juliacall)
"""
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Julia import — only load when engine is actually used
# ---------------------------------------------------------------------------

_ADCD_ENGINE_PATH = Path(os.environ.get("ADCD_ENGINE_PATH", Path(__file__).parent.parent.parent / "ADCDEngine"))

# ---------------------------------------------------------------------------
# Config / Result dataclasses (mirror Julia structs)
# ---------------------------------------------------------------------------

@dataclass
class JuliaEngineConfig:
    """Configuration for the ADCD Julia engine."""
    domain: str
    target_dim: str                       # "dimensionless", "acceleration", etc.
    input_vars: list[str]                 # physical variable names in data
    known_constants: dict[str, float] = field(default_factory=dict)
    bic_threshold: float = 6.0
    nmse_coarse: float = 1.0
    nmse_fine: float = 0.1
    n_restarts: int = 15
    max_proposals: int = 500
    groups: Optional[list[int]] = None   # for hierarchical BIC (e.g. SPARC galaxies)
    excluded_primitives: list[str] = field(default_factory=list)  # for positive/ablation control
    # Bug #2 fix: correction_type propagates mode_detection output to Julia engine.
    # "multiplicative": y_pred = y_cl*(1+Δ).  "additive": y_pred = y_cl+Δ.
    # Previously missing → 4/5 real-physics scenarios with y_classical≡0 were
    # structurally unfittable (gradient=0 everywhere, optimizer blind).
    correction_type: str = "multiplicative"
    classical_limit_direction: str = "0"  # "0" or "oo"

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "target_dim": self.target_dim,
            "input_vars": self.input_vars,
            "known_constants": self.known_constants,
            "bic_threshold": self.bic_threshold,
            "nmse_coarse": self.nmse_coarse,
            "nmse_fine": self.nmse_fine,
            "n_restarts": self.n_restarts,
            "max_proposals": self.max_proposals,
            "groups": self.groups,
            "excluded_primitives": self.excluded_primitives,
            "correction_type": self.correction_type,
            "classical_limit_direction": self.classical_limit_direction,
        }


@dataclass
class JuliaEngineData:
    """Data payload for the ADCD Julia engine."""
    y_classical: np.ndarray               # classical model predictions
    y_obs: np.ndarray                     # observed target values
    vars: dict[str, np.ndarray]           # physical variable arrays
    sigma_y: Optional[np.ndarray] = None  # per-point uncertainty (for weighted loss)

    def to_dict(self) -> dict:
        return {
            "y_classical": self.y_classical.tolist(),
            "y_obs": self.y_obs.tolist(),
            "vars": {k: v.tolist() for k, v in self.vars.items()},
            "sigma_y": self.sigma_y.tolist() if self.sigma_y is not None else None,
        }


@dataclass
class CandidateResult:
    """A single proposal evaluated by the Julia engine."""
    description: str       # human readable structure string (e.g. "D_lor(theta*u)")
    expr_str: str          # sympy parsable structure string
    pattern: str           # e.g. "singleton", "additive"
    primitives: list[str]  # e.g. ["D_lor"]
    n_params: int
    theta: list[float]
    nmse: float
    likelihood: float
    converged: bool
    verdict: str           # "IDENTIFIABLE", "WITHHELD", "POSITIVE_CONTROL_FAILED"
    delta_bic: float

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateResult":
        return cls(**d)

    @property
    def is_identifiable(self) -> bool:
        return self.verdict == "IDENTIFIABLE"

    @property
    def expr(self) -> str:
        """Return the sympy parsable string for exact symbolic matching."""
        return self.expr_str


@dataclass
class ADCDEngineResult:
    """Full result from a Julia engine run."""
    n_proposals_generated: int      # total proposals from CorrectionProposer (before exclusion)
    gate_stats: dict[str, int]
    results: list[CandidateResult]
    n_proposals_evaluated: int = 0  # after excluded_primitives filter
    primitives_active: list[str] = field(default_factory=list)  # primitives actually searched

    @property
    def identifiable(self) -> list[CandidateResult]:
        return [r for r in self.results if r.is_identifiable]

    @property
    def best(self) -> Optional[CandidateResult]:
        return self.identifiable[0] if self.identifiable else None

    def summary(self) -> str:
        lines = [
            "ADCD Engine v2 Result",
            f"  Proposals generated : {self.n_proposals_generated}",
            f"  Gate A (dim)        : {self.gate_stats.get('n_pass_gate_a',0)}",
            f"  Gate B (asymp)      : {self.gate_stats.get('n_pass_gate_b',0)}",
            f"  Gate C (coarse)     : {self.gate_stats.get('n_pass_gate_c',0)}",
            f"  Gate D (fine)       : {self.gate_stats.get('n_pass_gate_d',0)}",
            f"  IDENTIFIABLE        : {self.gate_stats.get('n_identifiable',0)}",
            f"  WITHHELD            : {self.gate_stats.get('n_withheld',0)}",
        ]
        if self.best:
            b = self.best
            lines += [
                "\nBest result:",
                f"  Description : {b.description}",
                f"  Pattern     : {b.pattern}",
                f"  Primitives  : {', '.join(b.primitives)}",
                f"  n_params    : {b.n_params}",
                f"  theta       : {b.theta}",
                f"  NMSE        : {b.nmse:.6f}",
                f"  delta_BIC   : {b.delta_bic:.2f}",
                f"  Verdict     : {b.verdict}",
            ]
        else:
            lines.append("\nNo IDENTIFIABLE results found.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class ADCDJuliaEngine:
    """
    Python interface to the ADCD Julia engine.

    This class is the boundary between Python (user-facing API) and
    Julia (computational engine). All heavy computation runs in Julia.

    Example
    -------
    >>> engine = ADCDJuliaEngine()
    >>> config = JuliaEngineConfig(
    ...     domain="lorentz_special_relativity",
    ...     target_dim="dimensionless",
    ...     input_vars=["v", "c"],
    ...     known_constants={"c": 3e8},
    ... )
    >>> data = JuliaEngineData(
    ...     y_classical=y_cl,
    ...     y_obs=y_obs,
    ...     vars={"v": v_arr, "c": c_arr},
    ... )
    >>> result = engine.run(config, data)
    >>> print(result.summary())
    """

    def __init__(self, prefer_subprocess: bool = True):
        self._jl = None  # lazy init
        self._use_subprocess = False
        self._prefer_subprocess = prefer_subprocess

    def _load(self):
        if self._jl is not None:
            return

        engine_path = str(_ADCD_ENGINE_PATH)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"ADCDEngine.jl project not found at: {engine_path}\n"
                "Make sure e:\\ADCD\\ADCDEngine\\ exists."
            )

        backend = os.environ.get("ADCD_JULIA_BACKEND", "subprocess" if self._prefer_subprocess else "juliacall")
        if backend == "subprocess":
            self._jl = "subprocess"
            self._use_subprocess = True
            logger.info("[julia_bridge] Using system Julia subprocess runner.")
            return

        try:
            from juliacall import Main as jl
            logger.info("[julia_bridge] Loading Julia ADCDEngine.jl via juliacall...")
            jl.seval(f'using Pkg; Pkg.activate(raw"{engine_path}")')
            jl.seval("using ADCDEngine")
            self._jl = jl
            self._use_subprocess = False
            logger.info("[julia_bridge] ADCDEngine.jl loaded successfully via juliacall.")
        except Exception as e:
            logger.info(f"[julia_bridge] juliacall not available ({e}), falling back to Julia subprocess runner.")
            self._jl = "subprocess"
            self._use_subprocess = True

    def run(
        self,
        config: JuliaEngineConfig,
        data: JuliaEngineData,
    ) -> ADCDEngineResult:
        """
        Run the ADCD Julia engine on the provided config and data.

        Parameters
        ----------
        config : JuliaEngineConfig
        data   : JuliaEngineData

        Returns
        -------
        ADCDEngineResult with gate statistics and ranked candidate list.
        """
        self._load()

        config_json = json.dumps(config.to_dict())
        data_json   = json.dumps(data.to_dict())

        logger.info(f"[ADCDJuliaEngine] Running domain={config.domain}, "
                    f"vars={config.input_vars}, n_data={len(data.y_obs)}")

        if self._use_subprocess:
            import subprocess
            import tempfile
            cli_script = str(_ADCD_ENGINE_PATH / "scripts" / "adcd_cli.jl")
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f_cfg, \
                 tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f_dat, \
                 tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f_out:
                f_cfg.write(config_json)
                f_cfg.flush()
                f_dat.write(data_json)
                f_dat.flush()
                cfg_path, dat_path, out_path = f_cfg.name, f_dat.name, f_out.name

            try:
                cmd = ["julia", f"--project={_ADCD_ENGINE_PATH}", cli_script, cfg_path, dat_path, out_path]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("Julia Engine Crash Stderr:", e.stderr)
                raise
            try:
                with open(out_path, "r", encoding="utf-8") as f_res:
                    raw = json.load(f_res)
            finally:
                for p in [cfg_path, dat_path, out_path]:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
        else:
            # Call Julia in-process via juliacall
            raw_json = self._jl.ADCDEngine.run_adcd(config_json, data_json)
            raw = json.loads(str(raw_json))

        return ADCDEngineResult(
            n_proposals_generated=raw["n_proposals_generated"],
            n_proposals_evaluated=raw.get("n_proposals_evaluated", raw["n_proposals_generated"]),
            primitives_active=list(raw.get("primitives_active", [])),
            gate_stats=dict(raw["gate_stats"]),
            results=[CandidateResult.from_dict(dict(r)) for r in raw["results"]],
        )

    def verify_dimension(self, expr_dict: dict, target_dim: str) -> bool:
        """
        Expose Julia's hard dimensional gate directly to Python.

        Parameters
        ----------
        expr_dict  : ADCD canonical expression dict
        target_dim : target dimension name (e.g. "dimensionless")

        Returns True iff expression matches target dimension.
        """
        self._load()
        import json
        expr_json = json.dumps(expr_dict)
        if self._use_subprocess:
            import subprocess
            import tempfile
            cli_script = str(_ADCD_ENGINE_PATH / "scripts" / "adcd_cli.jl")
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f_expr:
                f_expr.write(expr_json)
                f_expr.flush()
                expr_path = f_expr.name
            try:
                cmd = ["julia", f"--project={_ADCD_ENGINE_PATH}", cli_script, "--verify-dim", expr_path, target_dim]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
                return bool(lines and lines[-1].lower() == "true")
            finally:
                if os.path.exists(expr_path):
                    try:
                        os.remove(expr_path)
                    except Exception:
                        pass
        else:
            return bool(self._jl.ADCDEngine.ADCDDimensions.verify_dimension(
                expr_json, target_dim))

    def list_primitives(self):
        """Print all available primitives and their descriptions."""
        self._load()
        if self._use_subprocess:
            import subprocess
            cli_script = str(_ADCD_ENGINE_PATH / "scripts" / "adcd_cli.jl")
            cmd = ["julia", f"--project={_ADCD_ENGINE_PATH}", cli_script, "--list-primitives"]
            subprocess.run(cmd, check=True)
        else:
            self._jl.ADCDEngine.list_primitives()


# ---------------------------------------------------------------------------
# Convenience factory: build config from auto_scenario ScenarioContext
# ----------------------------------------------------------------------
def config_from_scenario(scenario: Any, threshold_cfg=None, X: dict = None) -> JuliaEngineConfig:
    """
    Build a JuliaEngineConfig from an AnomalyScenario (anomaly_scenarios.py).
    Enables direct use of ADCD scenarios with the Julia engine.

    Parameters
    ----------
    scenario : adcd.anomaly_scenarios.AnomalyScenario
    threshold_cfg : ScenarioThresholdConfig or None
        If None, auto-detected from scenario domain/tier via for_scenario().
    X : dict or None
        Data dictionary from generate_data(), used to extract grouping (e.g., galaxy_id).

    Returns
    -------
    JuliaEngineConfig
    """
    # Auto-calibrate thresholds if not provided
    if threshold_cfg is None:
        # Import here to avoid circular dependency
        try:
            from adcd.run_adcd_v3_validation_blind import ScenarioThresholdConfig
            threshold_cfg = ScenarioThresholdConfig.for_scenario(scenario)
        except ImportError:
            pass  # Fall back to defaults below

    bic_threshold = getattr(threshold_cfg, "bic_threshold", 10.0) if threshold_cfg else 10.0
    nmse_fine = getattr(threshold_cfg, "nmse_fine", 0.1) if threshold_cfg else 0.1
    nmse_coarse = getattr(threshold_cfg, "nmse_coarse", 1.0) if threshold_cfg else 1.0
    
    # Check for galaxy_id in X to build groups
    groups = None
    if X is not None and "galaxy_id" in X:
        galaxy_ids = np.asarray(X["galaxy_id"])
        # Use np.unique but preserve order of first appearance if possible, or just np.unique
        unique_ids = np.unique(galaxy_ids)
        # Add 1 because Julia uses 1-based indexing!
        groups = [(np.where(galaxy_ids == uid)[0] + 1).tolist() for uid in unique_ids]
    elif threshold_cfg:
        groups = getattr(threshold_cfg, "groups", None)

    # AUDIT FIX: use classical_variables (AnomalyScenario), not variable_names
    input_vars = list(scenario.classical_variables)

    return JuliaEngineConfig(
        domain          = scenario.domain,
        target_dim      = getattr(scenario, "target_dim", "dimensionless"),
        input_vars      = input_vars,
        known_constants = dict(scenario.classical_constants),
        bic_threshold   = bic_threshold,
        nmse_coarse     = nmse_coarse,
        nmse_fine       = nmse_fine,
        n_restarts      = int(getattr(scenario, "n_restarts", 15)),
        max_proposals   = 500,
        groups          = groups,
        correction_type = getattr(scenario, "correction_type", "multiplicative"),
        classical_limit_direction = getattr(scenario, "classical_limit_direction", "0"),
    )


def data_from_scenario(
    scenario: Any,
    y_classical: np.ndarray,
    y_obs: np.ndarray,
    X: dict,
) -> JuliaEngineData:
    """
    Build JuliaEngineData from an AnomalyScenario + pre-generated arrays.

    Parameters
    ----------
    scenario : adcd.anomaly_scenarios.AnomalyScenario
    y_classical : np.ndarray
        Classical model predictions (from scenario.generate_data()).
    y_obs : np.ndarray
        Observed (noisy) values (from scenario.generate_data()).
    X : dict[str, np.ndarray]
        Variable arrays (first return value of scenario.generate_data()).
        NOTE: AnomalyScenario has no .variable_data attribute; data must
        be generated by calling scenario.generate_data() explicitly.

    Returns
    -------
    JuliaEngineData
    """
    return JuliaEngineData(
        y_classical=np.asarray(y_classical, dtype=np.float64),
        y_obs=np.asarray(y_obs, dtype=np.float64),
        vars={k: np.asarray(v, dtype=np.float64) for k, v in X.items()
              if isinstance(v, (np.ndarray, list))},
        sigma_y=None,
    )
