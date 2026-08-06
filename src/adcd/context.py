from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from abc import ABC, abstractmethod

class BaseProposer(ABC):
    @abstractmethod
    def propose(self, context: 'ProposalContext') -> List[str]:
        """Generate candidate equation strings"""
        pass

import numpy as np

@dataclass
class ProposalContext:
    # --- Standard Statistics & Parameters ---
    variable_names: List[str]
    target_name: str
    data_statistics: Dict[str, Dict[str, float]]
    n_candidates: int = 50
    iteration: int = 0
    stuck_count: int = 0  # Number of iterations without NMSE improvement

    # --- Rich Physical Metadata for Guided Discovery (Paper) ---
    domain: str = "classical physics"
    classical_expr: str = ""
    variables_with_units: Dict[str, str] = field(default_factory=dict)
    anomaly_description: str = "None"
    known_limits: List[dict] = field(default_factory=list)
    classical_limit_condition: str = ""
    max_nodes: int = 15
    structural_hints: List[str] = field(default_factory=list)

    # --- Search History (Feedback Loop) ---
    previous_best: Optional[List[Tuple[str, float]]] = None  # List of (expr, nmse)
    physical_hints: Optional[List[str]] = None  # Kept for compatibility
    constants: Dict[str, float] = field(default_factory=dict)
    residual_features: Optional[Any] = None
    X_data: Optional[Dict[str, np.ndarray]] = None
    residual_data: Optional[np.ndarray] = None
