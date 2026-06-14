import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class AnomalyScenario:
    name: str
    tier: str                         # "textbook" | "cross_domain" | "synthetic"
    domain: str                       # e.g., "gravity", "thermodynamics"
    
    # The known classical law
    classical_expr: str               # e.g., "0.5 * m * v**2"
    classical_variables: List[str]    # e.g., ["m", "v"]
    classical_constants: Dict[str, float]  # e.g., {"c": 3e8}
    
    # The hidden correction (ground truth, hidden from the pipeline)
    correction_type: str              # "multiplicative" or "additive"
    correction_expr: str              # e.g., "theta_0 * (v / c)**2"
    correction_constants: Dict[str, float]  # e.g., {"theta_0": 0.75}
    
    # Physical metadata for the LLM
    anomaly_regime: str               # e.g., "high speeds v approaching c"
    variables_with_units: Dict[str, str]
    classical_limit_variable: str     # e.g., "v"
    classical_limit_direction: str    # e.g., "0" (Δ -> 0 as v -> 0)
    
    # Structural classification (for evaluation)
    correction_class: str             # "exponential" | "power_law" | "rational" | "trigonometric" | "polynomial" | "logarithmic"

    def generate_data(self, n_points: int = 200, noise_level: float = 0.0, seed: int = 42) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates variables X, classical prediction y_classical, 
        noisy observation y_obs, and the corresponding residual.
        """
        rng = np.random.RandomState(seed)
        X = {}
        
        # 1. Generate domain-specific variables in the anomaly-sensitive regime
        if self.name == "Relativistic KE" or self.name.startswith("Subtle Misspecification"):
            c = self.classical_constants["c"]
            X["m"] = rng.uniform(0.5, 10.0, size=n_points)
            X["v"] = rng.uniform(0.1 * c, 0.85 * c, size=n_points)
            
        elif self.name == "Yukawa Gravity":
            X["m"] = rng.uniform(1.0, 10.0, size=n_points)
            X["M"] = rng.uniform(10.0, 100.0, size=n_points)
            X["r"] = rng.uniform(0.5, 5.0, size=n_points)
            
        elif self.name == "Anharmonic Spring":
            X["k"] = rng.uniform(5.0, 50.0, size=n_points)
            X["x"] = rng.uniform(0.1, 3.0, size=n_points)
            
        elif self.name == "Screened Coulomb":
            X["q1"] = rng.uniform(1e-6, 1e-5, size=n_points)
            X["q2"] = rng.uniform(1e-6, 1e-5, size=n_points)
            X["r"] = rng.uniform(0.2, 4.0, size=n_points)
            
        elif self.name == "Net Radiation":
            X["A"] = rng.uniform(0.1, 2.0, size=n_points)
            X["T"] = rng.uniform(250.0, 800.0, size=n_points)
            
        elif "Nonlinear Drag" in self.name:
            X["b"] = rng.uniform(0.5, 5.0, size=n_points)
            X["v"] = rng.uniform(0.1, 5.0, size=n_points)
            
        elif self.name == "Mystery-A":
            X["m"] = rng.uniform(1.0, 10.0, size=n_points)
            X["M"] = rng.uniform(10.0, 100.0, size=n_points)
            X["r"] = rng.uniform(0.5, 5.0, size=n_points)
            
        elif self.name == "Mystery-B":
            X["m"] = rng.uniform(0.5, 5.0, size=n_points)
            X["v"] = rng.uniform(0.1, 10.0, size=n_points)
            
        elif self.name == "Mystery-C":
            X["k"] = rng.uniform(10.0, 100.0, size=n_points)
            X["x"] = rng.uniform(0.1, 5.0, size=n_points)
            
        elif self.name == "Blind-1: Van der Waals":
            X["n"] = rng.uniform(1.0, 5.0, size=n_points)
            X["T"] = rng.uniform(250.0, 450.0, size=n_points)
            X["V"] = rng.uniform(1.0, 5.0, size=n_points)
            
        elif self.name == "Blind-2: Stokes-Einstein":
            X["T"] = rng.uniform(270.0, 350.0, size=n_points)
            X["r"] = rng.uniform(1.0, 5.0, size=n_points)
            
        elif self.name == "Blind-3: Wien Displacement" or self.name == "Blind-8: Composite Blackbody":
            X["T"] = rng.uniform(1000.0, 6000.0, size=n_points)
            X["f"] = rng.uniform(1e12, 1e14, size=n_points)

        elif self.name == "Blind-4: Relativistic Pendulum":
            c = self.classical_constants["c"]
            X["m"] = rng.uniform(0.5, 10.0, size=n_points)
            X["v"] = rng.uniform(0.1 * c, 0.85 * c, size=n_points)

        elif self.name == "Blind-5: Clausius-Mossotti Field" or self.name == "Blind-7: Casimir Vacuum":
            X["m"] = rng.uniform(1.0, 10.0, size=n_points)
            X["M"] = rng.uniform(10.0, 100.0, size=n_points)
            X["r"] = rng.uniform(0.5, 5.0, size=n_points)

        elif self.name == "Blind-6: Magnus Wind-Tunnel" or self.name == "Blind-9: Composite Relativistic Drag":
            X["b"] = rng.uniform(0.5, 5.0, size=n_points)
            X["v"] = rng.uniform(0.1, 5.0, size=n_points)
            
        elif self.name == "Misspecification 1: Wrong Baseline Form":
            X["m"] = rng.uniform(1.0, 10.0, size=n_points)
            X["v"] = rng.uniform(0.1, 5.0, size=n_points)
            
        elif self.name == "Misspecification 2: Missing Variable":
            X["m"] = rng.uniform(1.0, 10.0, size=n_points)
            X["g"] = np.full(n_points, 9.81)
            # We generate 'v' here internally to create the ground truth, 
            # even though the user (classical_variables) didn't specify it.
            # We must explicitly add it to local_corr_dict later.
            self._hidden_v = rng.uniform(0.1, 5.0, size=n_points)
            
        elif self.name == "Misspecification 3: Spurious Variable":
            X["k"] = rng.uniform(5.0, 50.0, size=n_points)
            X["x"] = rng.uniform(0.1, 3.0, size=n_points)
            X["T"] = rng.uniform(250.0, 400.0, size=n_points) # Irrelevant variable
            
        else:
            # Fallback random generator for arbitrary names
            for var in self.classical_variables:
                X[var] = rng.uniform(1.0, 10.0, size=n_points)

        # 2. Evaluate classical law
        local_dict = {**X, **self.classical_constants}
        y_classical = eval(self.classical_expr, {"np": np, "sp": sp}, local_dict)
        
        # 3. Evaluate ground-truth correction
        local_corr_dict = {**X, **self.classical_constants, **self.correction_constants}
        
        # Inject hidden variables for missing variable case
        if self.name == "Misspecification 2: Missing Variable":
            local_corr_dict["v"] = self._hidden_v
            
        # Safely evaluate ground truth correction
        # Replace theta_X names with their actual values in the expression
        expr_str = self.correction_expr
        for k, v in self.correction_constants.items():
            expr_str = expr_str.replace(k, str(v))
        
        # Map exp, sin, cos, tanh to numpy counterparts
        eval_env = {
            "np": np, 
            "sp": sp,
            "exp": np.exp,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "log": np.log,
            "sqrt": np.sqrt
        }
        delta_true = eval(expr_str, eval_env, local_corr_dict)

        # 4. Compute y_true
        if self.correction_type == "multiplicative":
            y_true = y_classical * (1.0 + delta_true)
        else:  # additive
            y_true = y_classical + delta_true
            
        # 5. Add observational Gaussian noise
        if noise_level > 0.0:
            # Multiplicative noise relative to y_true
            noise = rng.normal(0, noise_level, size=n_points)
            y_obs = y_true * (1.0 + noise)
        else:
            y_obs = y_true.copy()
            
        # 6. Compute residual
        if self.correction_type == "multiplicative":
            residual = y_obs / y_classical - 1.0
        else:
            residual = y_obs - y_classical
            
        return X, y_obs, y_classical, residual

def get_all_scenarios() -> List[AnomalyScenario]:
    """Returns the 9 standard scenarios across the 3 tiers."""
    return [
        # =========================================================================
        # TIER 1: Textbook Corrections (LLM seen these)
        # =========================================================================
        AnomalyScenario(
            name="Relativistic KE",
            tier="textbook",
            domain="relativistic mechanics",
            classical_expr="0.5 * m * v**2",
            classical_variables=["m", "v"],
            classical_constants={"c": 3.0e8},
            correction_type="multiplicative",
            correction_expr="theta_0 * (v / c)**2",
            correction_constants={"theta_0": 0.75},
            anomaly_regime="high speeds v approaching c",
            variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        AnomalyScenario(
            name="Yukawa Gravity",
            tier="textbook",
            domain="gravitation",
            classical_expr="G * m * M / r**2",
            classical_variables=["m", "M", "r"],
            classical_constants={"G": 6.6743e-11},
            correction_type="multiplicative",
            correction_expr="theta_0 * exp(-r / theta_1)",
            correction_constants={"theta_0": 0.15, "theta_1": 2.5},
            anomaly_regime="short distances r < 5.0",
            variables_with_units={"m": "kg", "M": "kg", "r": "m", "G": "N*m^2/kg^2"},
            classical_limit_variable="r",
            classical_limit_direction="oo",
            correction_class="exponential"
        ),
        AnomalyScenario(
            name="Anharmonic Spring",
            tier="textbook",
            domain="mechanics",
            classical_expr="0.5 * k * x**2",
            classical_variables=["k", "x"],
            classical_constants={},
            correction_type="additive",
            correction_expr="theta_0 * x**4",
            correction_constants={"theta_0": 0.15},
            anomaly_regime="large amplitude displacements x > 1.5",
            variables_with_units={"k": "N/m", "x": "m"},
            classical_limit_variable="x",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # =========================================================================
        # TIER 2: Cross-Domain (known physics, unusual pairing)
        # =========================================================================
        AnomalyScenario(
            name="Screened Coulomb",
            tier="cross_domain",
            domain="electrostatics",
            classical_expr="k_e * q1 * q2 / r**2",
            classical_variables=["q1", "q2", "r"],
            classical_constants={"k_e": 8.9876e9},
            correction_type="multiplicative",
            correction_expr="exp(-r / theta_0) - 1.0",
            correction_constants={"theta_0": 1.5},
            anomaly_regime="shielded plasma environments, large distances r > 1.0",
            variables_with_units={"q1": "C", "q2": "C", "r": "m", "k_e": "N*m^2/C^2"},
            classical_limit_variable="r",
            classical_limit_direction="0",
            correction_class="exponential"
        ),
        AnomalyScenario(
            name="Net Radiation",
            tier="cross_domain",
            domain="thermodynamics",
            classical_expr="sigma * A * T**4",
            classical_variables=["A", "T"],
            classical_constants={"sigma": 5.6704e-8},
            correction_type="multiplicative",
            correction_expr="- (theta_0 / T)**4",
            # We treat T_env = 293.15 K as theta_0 parameter
            correction_constants={"theta_0": 293.15},
            anomaly_regime="cool temperatures close to ambient temperature T < 500 K",
            variables_with_units={"A": "m^2", "T": "K", "sigma": "W/(m^2*K^4)"},
            classical_limit_variable="T",
            classical_limit_direction="oo",
            correction_class="power_law"
        ),
        AnomalyScenario(
            name="Nonlinear Drag",
            tier="cross_domain",
            domain="fluid dynamics",
            classical_expr="b * v",
            classical_variables=["b", "v"],
            classical_constants={},
            correction_type="additive",
            # F_drag = b*v + theta_0 * v**2
            # residual = F_drag - b*v = theta_0 * v**2
            # Enforces addition of quadratic drag at higher Reynolds numbers
            correction_expr="theta_0 * v**2",
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed turbulent flows v > 2.0",
            variables_with_units={"b": "kg/s", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # =========================================================================
        # TIER 3: Synthetic / Novel (LLM has never seen these)
        # =========================================================================
        AnomalyScenario(
            name="Mystery-A",
            tier="synthetic",
            domain="gravitation",
            classical_expr="G * m * M / r**2",
            classical_variables=["m", "M", "r"],
            classical_constants={"G": 6.6743e-11},
            correction_type="multiplicative",
            correction_expr="-tanh(theta_0 / r)**2",
            correction_constants={"theta_0": 1.2},
            anomaly_regime="sub-wavelength strong gravitational fields, small r < 3.0",
            variables_with_units={"m": "kg", "M": "kg", "r": "m", "G": "N*m^2/kg^2"},
            classical_limit_variable="r",
            classical_limit_direction="oo",
            correction_class="trigonometric"
        ),
        AnomalyScenario(
            name="Mystery-B",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.5 * m * v**2",
            classical_variables=["m", "v"],
            classical_constants={},
            correction_type="multiplicative",
            # sinc correction function: sinc(v/v_0) - 1
            correction_expr="sin(v / theta_0) / (v / theta_0) - 1.0",
            correction_constants={"theta_0": 4.5},
            anomaly_regime="velocity fluctuations under quantum boundary, v > 1.0",
            variables_with_units={"m": "kg", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="trigonometric"
        ),
        AnomalyScenario(
            name="Mystery-C",
            tier="synthetic",
            domain="mechanics",
            classical_expr="k * x",
            classical_variables=["k", "x"],
            classical_constants={},
            correction_type="multiplicative",
            correction_expr="log(1.0 + x / theta_0) / (x / theta_0) - 1.0",
            correction_constants={"theta_0": 2.0},
            anomaly_regime="nonlinear polymer stretching, x > 0.5",
            variables_with_units={"k": "N/m", "x": "m"},
            classical_limit_variable="x",
            classical_limit_direction="0",
            correction_class="logarithmic"
        ),
        
        # ── BLIND TEST SCENARIOS ──────────────────────────────────────────────
        # Ground truth DISEMBUNYIKAN dari pipeline. Kita hanya tahu correction_class.
        # Ini untuk membuktikan generalisasi di luar benchmark yang dibuat sendiri.
        
        AnomalyScenario(
            name="Blind-1: Van der Waals",
            tier="blind",
            domain="thermodynamics",
            # Classical: Ideal gas law: P = nRT/V
            # Anomaly: Van der Waals correction factor (a/V^2 pressure term)
            classical_expr="n * R * T / V",
            classical_variables=["n", "T", "V"],
            classical_constants={"R": 8.314},
            correction_type="multiplicative",
            # Correction: (1 - a*n^2/V^2) factor, simplified as additive delta
            correction_expr="theta_0 * n**2 / V**2",
            correction_constants={"theta_0": 0.364},  # 'a' for CO2 in Pa·m^6/mol^2
            anomaly_regime="high pressure / low volume gas, V < 5L",
            variables_with_units={"n": "mol", "T": "K", "V": "m^3"},
            classical_limit_variable="V",
            classical_limit_direction="oo",  # correction -> 0 as V -> infinity (ideal gas limit)
            correction_class="rational"
        ),
        AnomalyScenario(
            name="Blind-2: Stokes-Einstein",
            tier="blind",
            domain="biophysics",
            # Classical: Einstein diffusion D = kT/(6*pi*eta*r)
            # Anomaly: Shape correction factor for non-spherical particles
            classical_expr="k_B * T / (6 * pi * eta * r)",
            classical_variables=["T", "r"],
            classical_constants={"k_B": 1.380649e-23, "pi": 3.14159, "eta": 1e-3},
            correction_type="multiplicative",
            # Oblate spheroid correction: (3/8)*sqrt(pi)*r^0.5 - 1  (simplified)
            correction_expr="theta_0 * (r / theta_1)**0.5",
            correction_constants={"theta_0": 0.15, "theta_1": 1.0},
            anomaly_regime="non-spherical macromolecules, r > 5nm",
            variables_with_units={"T": "K", "r": "m"},
            classical_limit_variable="r",
            classical_limit_direction="0",
            correction_class="power_law"
        ),
        AnomalyScenario(
            name="Blind-3: Wien Displacement",
            tier="blind",
            domain="quantum_optics",
            # Classical: Rayleigh-Jeans law: I = 2*k*T*f^2/c^2 (low freq)
            # Anomaly: Planck quantum correction
            classical_expr="2 * k_B * T * f**2 / c**2",
            classical_variables=["T", "f"],
            classical_constants={"k_B": 1.380649e-23, "c": 3e8},
            correction_type="multiplicative",
            # Quantum correction: (hf/kT)/(exp(hf/kT) - 1) relative to kT/hf limit
            # Simplified as: exp(-theta_0 * f / T) correction
            correction_expr="exp(-theta_0 * f / T) / (1 - exp(-theta_0 * f / T)) * (theta_0 * f / T)",
            correction_constants={"theta_0": 4.799e-11},  # h/k_B
            anomaly_regime="high frequency UV/visible regime, f > 1e13 Hz",
            variables_with_units={"T": "K", "f": "Hz"},
            classical_limit_variable="f",
            classical_limit_direction="0",
            correction_class="exponential"
        ),
        AnomalyScenario(
            name="Blind-4: Relativistic Pendulum",
            tier="blind",
            domain="relativistic mechanics",
            classical_expr="0.5 * m * v**2",
            classical_variables=["m", "v"],
            classical_constants={"c": 3.0e8},
            correction_type="multiplicative",
            correction_expr="theta_0 * (v / c)**2 / (1.0 - (v / c)**2)",
            correction_constants={"theta_0": 0.5},
            anomaly_regime="high speeds approaching c",
            variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="rational"
        ),
        AnomalyScenario(
            name="Blind-5: Clausius-Mossotti Field",
            tier="blind",
            domain="gravitation",
            classical_expr="G * m * M / r**2",
            classical_variables=["m", "M", "r"],
            classical_constants={"G": 6.6743e-11},
            correction_type="multiplicative",
            correction_expr="theta_0 * (r / theta_1) / (1.0 + r / theta_1)",
            correction_constants={"theta_0": 0.25, "theta_1": 2.0},
            anomaly_regime="short distance gravitational field scaling",
            variables_with_units={"m": "kg", "M": "kg", "r": "m", "G": "N*m^2/kg^2"},
            classical_limit_variable="r",
            classical_limit_direction="0",
            correction_class="rational"
        ),
        AnomalyScenario(
            name="Blind-6: Magnus Wind-Tunnel",
            tier="blind",
            domain="fluid dynamics",
            classical_expr="b * v",
            classical_variables=["b", "v"],
            classical_constants={},
            correction_type="multiplicative",
            correction_expr="theta_0 * (v / theta_1)**1.5",
            correction_constants={"theta_0": 0.4, "theta_1": 1.0},
            anomaly_regime="turbulent high speed airflow scaling",
            variables_with_units={"b": "kg/s", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="power_law"
        ),
        AnomalyScenario(
            name="Blind-7: Casimir Vacuum",
            tier="blind",
            domain="gravitation",
            classical_expr="G * m * M / r**2",
            classical_variables=["m", "M", "r"],
            classical_constants={"G": 6.6743e-11},
            correction_type="additive",
            correction_expr="-theta_0 / r**4",
            correction_constants={"theta_0": 0.05},
            anomaly_regime="sub-micron distance vacuum force correction",
            variables_with_units={"m": "kg", "M": "kg", "r": "m", "G": "N*m^2/kg^2"},
            classical_limit_variable="r",
            classical_limit_direction="oo",
            correction_class="power_law"
        ),
        AnomalyScenario(
            name="Blind-8: Composite Blackbody",
            tier="blind",
            domain="quantum_optics",
            classical_expr="2 * k_B * T * f**2 / c**2",
            classical_variables=["T", "f"],
            classical_constants={"k_B": 1.380649e-23, "c": 3e8},
            correction_type="multiplicative",
            # Composite product of polynomial and exponential
            correction_expr="(theta_0 * f / T) * exp(-theta_0 * f / T)",
            correction_constants={"theta_0": 4.799e-11},
            anomaly_regime="high frequency radiation limit",
            variables_with_units={"T": "K", "f": "Hz"},
            classical_limit_variable="f",
            classical_limit_direction="0",
            correction_class="exponential"
        ),
        AnomalyScenario(
            name="Blind-9: Composite Relativistic Drag",
            tier="blind",
            domain="fluid dynamics",
            classical_expr="b * v",
            classical_variables=["b", "v"],
            classical_constants={"c": 3e8},
            correction_type="multiplicative",
            # Composite product of polynomial and exponential
            correction_expr="(v / c)**2 * exp(-v / c)",
            correction_constants={},
            anomaly_regime="relativistic drag limits",
            variables_with_units={"b": "kg/s", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="exponential"
        )
    ]
