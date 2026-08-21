from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ADCD_ENGINE_PATH = Path(os.environ.get("ADCD_ENGINE_PATH", Path(__file__).parent.parent.parent / "ADCDEngine"))

@dataclass
class JuliaEngineConfig:
    domain: str
    target_dim: str
    input_vars: list[str]
    known_constants: dict[str, float] = field(default_factory=dict)
    bic_threshold: float = 6.0
    nmse_coarse: float = 1.0
    nmse_fine: float = 0.1
    n_restarts: int = 15
    max_proposals: int = 500
    groups: Optional[list[list[int]]] = None  # Perbaikan tipe list of lists
    excluded_primitives: list[str] = field(default_factory=list)
    correction_type: str = "multiplicative"
    classical_limit_direction: str = "0"
    classical_limit_variable: str = ""

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
            "classical_limit_variable": self.classical_limit_variable,
        }

@dataclass
class JuliaEngineData:
    y_classical: np.ndarray
    y_obs: np.ndarray
    vars: dict[str, np.ndarray]
    sigma_y: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "y_classical": np.asarray(self.y_classical, dtype=float).tolist(),
            "y_obs": np.asarray(self.y_obs, dtype=float).tolist(),
            "vars": {k: np.asarray(v, dtype=float).tolist() for k, v in self.vars.items()},
            "sigma_y": np.asarray(self.sigma_y, dtype=float).tolist() if self.sigma_y is not None else None,
        }

@dataclass
class CandidateResult:
    description: str
    expr_str: str
    pattern: str
    primitives: list[str]
    n_params: int
    theta: list[float]
    nmse: float
    likelihood: float
    converged: bool
    verdict: str
    delta_bic: float
    theta_fit: dict = field(default_factory=dict)  # Default factory mencegah crash

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateResult":
        data = dict(d)
        if "theta_fit" not in data and "theta" in data:
            data["theta_fit"] = {f"theta_{i}": v for i, v in enumerate(data["theta"])}
        return cls(**data)

    @property
    def is_identifiable(self) -> bool:
        return self.verdict == "IDENTIFIABLE"

    @property
    def expr(self) -> str:
        return self.expr_str

@dataclass
class ADCDEngineResult:
    n_proposals_generated: int
    gate_stats: dict[str, int]
    results: list[CandidateResult]
    n_proposals_evaluated: int = 0
    primitives_active: list[str] = field(default_factory=list)

    @property
    def identifiable(self) -> list[CandidateResult]:
        return [r for r in self.results if r.is_identifiable]

    @property
    def best(self) -> Optional[CandidateResult]:
        return self.identifiable[0] if self.identifiable else None

class ADCDJuliaEngine:
    def __init__(self, prefer_subprocess: bool = True):
        self._jl = None
        self._use_subprocess = False
        self._prefer_subprocess = prefer_subprocess

    def _load(self):
        if self._jl is not None:
            return
        engine_path = str(_ADCD_ENGINE_PATH)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"ADCDEngine.jl project not found at: {engine_path}")

        backend = os.environ.get("ADCD_JULIA_BACKEND", "subprocess" if self._prefer_subprocess else "juliacall")
        if backend == "subprocess":
            self._jl = "subprocess"
            self._use_subprocess = True
            return

        try:
            from juliacall import Main as jl
            jl.seval(f'using Pkg; Pkg.activate(raw"{engine_path}")')
            jl.seval("using ADCDEngine")
            self._jl = jl
            self._use_subprocess = False
        except Exception:
            self._jl = "subprocess"
            self._use_subprocess = True

    def run(self, config: JuliaEngineConfig, data: JuliaEngineData) -> ADCDEngineResult:
        self._load()
        config_json = json.dumps(config.to_dict())
        data_json = json.dumps(data.to_dict())

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
                with open(out_path, "r", encoding="utf-8") as f_res:
                    raw = json.load(f_res)
            finally:
                for p in [cfg_path, dat_path, out_path]:
                    if os.path.exists(p):
                        os.remove(p)
        else:
            raw_json = self._jl.ADCDEngine.run_adcd(config_json, data_json)
            raw = json.loads(str(raw_json))

        return ADCDEngineResult(
            n_proposals_generated=raw["n_proposals_generated"],
            n_proposals_evaluated=raw.get("n_proposals_evaluated", raw["n_proposals_generated"]),
            primitives_active=list(raw.get("primitives_active", [])),
            gate_stats=dict(raw["gate_stats"]),
            results=[CandidateResult.from_dict(dict(r)) for r in raw["results"]],
        )

def data_from_scenario(scenario: Any, y_classical: np.ndarray, y_obs: np.ndarray, X: dict) -> JuliaEngineData:
    sigma_y = None
    if "sigma_y" in X:
        sigma_y = np.asarray(X["sigma_y"], dtype=np.float64)
    elif hasattr(scenario, "sigma_y") and scenario.sigma_y is not None:
        sigma_y = np.asarray(scenario.sigma_y, dtype=np.float64)

    return JuliaEngineData(
        y_classical=np.asarray(y_classical, dtype=np.float64),
        y_obs=np.asarray(y_obs, dtype=np.float64),
        vars={k: np.asarray(v, dtype=np.float64) for k, v in X.items() if isinstance(v, (np.ndarray, list)) and k != "sigma_y"},
        sigma_y=sigma_y,
    )
