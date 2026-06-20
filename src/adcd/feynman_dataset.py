import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from adcd.constants import G as G_CODATA, K_B as K_B_CODATA

@dataclass
class FeynmanProblem:
    name: str
    equation: str
    variables: List[str]
    constants: Dict[str, float]
    target_dimension: Optional[str]
    regimes: List[dict]  # keys: 'variable', 'limit', 'expected'
    
    # --- Advanced Physical Metadata for Guided Discovery (Q1 Paper) ---
    domain: str = "classical physics"
    classical_expr: str = ""
    variables_with_units: Dict[str, str] = field(default_factory=dict)
    anomaly_description: str = "None"
    known_limits: List[dict] = field(default_factory=list)
    classical_limit_condition: str = ""
    structural_hints: List[str] = field(default_factory=list)

    def generate_data(self, n_points: int = 100, seed: int = 42) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        rng = np.random.RandomState(seed)
        X = {}
        
        if self.name == "Kinetic Energy":
            X["m"] = rng.uniform(0.5, 10.0, size=n_points)
            X["v"] = rng.uniform(0.5, 10.0, size=n_points)
            y = 0.5 * X["m"] * X["v"]**2
            
        elif self.name == "Gravitational Force":
            X["m"] = rng.uniform(1.0, 100.0, size=n_points)
            X["M"] = rng.uniform(1.0, 100.0, size=n_points)
            X["r"] = rng.uniform(0.5, 10.0, size=n_points)
            y = self.constants["G"] * X["m"] * X["M"] / X["r"]**2
            
        elif self.name == "Coulomb's Law":
            X["q1"] = rng.uniform(1e-6, 1e-3, size=n_points)
            X["q2"] = rng.uniform(1e-6, 1e-3, size=n_points)
            X["r"] = rng.uniform(0.1, 5.0, size=n_points)
            theta_0 = self.constants.get("theta_0", 8.99e9)
            y = theta_0 * X["q1"] * X["q2"] / X["r"]**2
            
        elif self.name == "Spring Potential":
            X["k"] = rng.uniform(1.0, 100.0, size=n_points)
            X["r"] = rng.uniform(0.1, 5.0, size=n_points)
            y = 0.5 * X["k"] * X["r"]**2
            
        elif self.name == "Simple Pendulum Period":
            X["r"] = rng.uniform(0.1, 5.0, size=n_points)
            y = 2.0 * np.pi * np.sqrt(X["r"] / 9.8)
            
        elif self.name == "Linear Motion":
            X["v"] = rng.uniform(1.0, 20.0, size=n_points)
            X["t"] = rng.uniform(0.1, 5.0, size=n_points)
            y = X["v"] * X["t"]
            
        elif self.name == "Gravitational Potential Energy":
            X["m"] = rng.uniform(1.0, 100.0, size=n_points)
            X["M"] = rng.uniform(1.0, 100.0, size=n_points)
            X["r"] = rng.uniform(0.5, 10.0, size=n_points)
            y = -self.constants["G"] * X["m"] * X["M"] / X["r"]
            
        elif self.name == "Relativistic Kinetic Energy":
            X["m"] = rng.uniform(1.0, 10.0, size=n_points)
            X["v"] = rng.uniform(1e3, 1e5, size=n_points)
            c = self.constants["c"]
            y = 0.5 * X["m"] * X["v"]**2 + 0.375 * X["m"] * X["v"]**4 / c**2
            
        elif self.name == "Ideal Gas Law":
            X["N"] = rng.uniform(1e20, 1e23, size=n_points)
            X["t"] = rng.uniform(200.0, 500.0, size=n_points)
            k_B = self.constants.get("theta_0", K_B_CODATA)
            y = k_B * X["N"] * X["t"]
            
        elif self.name == "Projectile Range":
            X["v"] = rng.uniform(5.0, 30.0, size=n_points)
            X["t"] = rng.uniform(0.1, 1.5, size=n_points)
            y = X["v"]**2 * np.sin(X["t"]) / 9.8

        elif self.name == "Ohm's Law":
            X["I"] = rng.uniform(0.1, 10.0, size=n_points)
            X["R"] = rng.uniform(1.0, 100.0, size=n_points)
            y = X["I"] * X["R"]

        elif self.name == "Power Dissipation":
            X["I"] = rng.uniform(0.1, 10.0, size=n_points)
            X["R"] = rng.uniform(1.0, 100.0, size=n_points)
            y = X["I"]**2 * X["R"]

        elif self.name == "Capacitor Energy":
            X["C"] = rng.uniform(1e-6, 1e-3, size=n_points)
            X["V"] = rng.uniform(1.0, 100.0, size=n_points)
            y = 0.5 * X["C"] * X["V"]**2

        elif self.name == "Wave Speed":
            X["f"] = rng.uniform(100.0, 10000.0, size=n_points)
            X["lam"] = rng.uniform(0.01, 10.0, size=n_points)
            y = X["f"] * X["lam"]

        elif self.name == "Doppler Effect":
            X["f"] = rng.uniform(100.0, 5000.0, size=n_points)
            X["v"] = rng.uniform(0.1, 50.0, size=n_points)
            c_s = self.constants["c_s"]
            y = X["f"] * (1.0 + X["v"] / c_s)

        elif self.name == "Stefan-Boltzmann":
            X["A"] = rng.uniform(0.1, 10.0, size=n_points)
            X["T"] = rng.uniform(300.0, 3000.0, size=n_points)
            sigma = self.constants["sigma"]
            y = sigma * X["A"] * X["T"]**4

        elif self.name == "Wien's Law":
            X["T"] = rng.uniform(1000.0, 10000.0, size=n_points)
            b = self.constants["b"]
            y = b / X["T"]

        elif self.name == "Lorentz Force":
            X["q"] = rng.uniform(1e-9, 1e-6, size=n_points)
            X["v"] = rng.uniform(1e3, 1e6, size=n_points)
            X["B"] = rng.uniform(0.01, 10.0, size=n_points)
            y = X["q"] * X["v"] * X["B"]

        elif self.name == "Buoyancy Force":
            X["rho"] = rng.uniform(500.0, 2000.0, size=n_points)
            X["V_f"] = rng.uniform(0.001, 1.0, size=n_points)
            g = self.constants["g"]
            y = X["rho"] * g * X["V_f"]

        elif self.name == "Lens Equation":
            X["d_o"] = rng.uniform(0.1, 5.0, size=n_points)
            X["d_i"] = rng.uniform(0.1, 5.0, size=n_points)
            y = X["d_o"] * X["d_i"] / (X["d_o"] + X["d_i"])

        else:
            raise ValueError(f"Unknown problem name: {self.name}")

        # Inject constants into X dict
        for const_name, const_val in self.constants.items():
            X[const_name] = np.full(n_points, const_val)

        return X, y

PROBLEMS = [
    FeynmanProblem(
        name="Kinetic Energy",
        equation="0.5 * m * v**2",
        variables=["m", "v"],
        constants={},
        target_dimension="E",
        regimes=[
            {"variable": "v", "limit": 0, "expected": "0"},
            {"variable": "v", "limit": "oo", "expected": "oo"}
        ],
        domain="classical mechanics",
        classical_expr="0.5 * m * v**2",
        variables_with_units={"m": "kg (mass)", "v": "m/s (velocity)"},
        anomaly_description="Deviations observed when velocity scales approach the relativistic domain.",
        known_limits=[{"variable": "v", "limit": "0", "expected": "0"}],
        classical_limit_condition="v -> 0",
        structural_hints=["relativistic factors", "rational functions", "terms like 1 / sqrt(1 - v**2/c**2)"]
    ),
    FeynmanProblem(
        name="Gravitational Force",
        equation="G * m * M / r**2",
        variables=["m", "M", "r"],
        constants={"G": G_CODATA},
        target_dimension="F",
        regimes=[
            {"variable": "r", "limit": "oo", "expected": "0"},
            {"variable": "r", "limit": 0, "expected": "oo"}
        ],
        domain="astrophysics",
        classical_expr="G * m * M / r**2",
        variables_with_units={"m": "kg (mass 1)", "M": "kg (mass 2)", "r": "m (distance)"},
        anomaly_description="Anomalies detected at short range galactic boundaries.",
        known_limits=[{"variable": "r", "limit": "oo", "expected": "0"}],
        classical_limit_condition="r -> oo",
        structural_hints=["corrections in 1/r**3", "exponential decay terms", "Yukawa potentials"]
    ),
    FeynmanProblem(
        name="Coulomb's Law",
        equation="theta_0 * q1 * q2 / r**2",
        variables=["q1", "q2", "r"],
        constants={"theta_0": 8.99e9},
        target_dimension=None,
        regimes=[
            {"variable": "r", "limit": "oo", "expected": "0"}
        ],
        domain="electromagnetism",
        classical_expr="theta_0 * q1 * q2 / r**2",
        variables_with_units={"q1": "C (charge 1)", "q2": "C (charge 2)", "r": "m (distance)"},
        anomaly_description="Electrodynamic deviations at subatomic separation distances.",
        known_limits=[{"variable": "r", "limit": "oo", "expected": "0"}],
        classical_limit_condition="r -> oo",
        structural_hints=["exponential screening", "shielding terms", "rational approximations"]
    ),
    FeynmanProblem(
        name="Spring Potential",
        equation="0.5 * k * r**2",
        variables=["k", "r"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "r", "limit": 0, "expected": "0"}
        ],
        domain="classical mechanics",
        classical_expr="0.5 * k * r**2",
        variables_with_units={"k": "N/m (spring constant)", "r": "m (displacement)"},
        anomaly_description="Nonlinear spring behavior at high deformation limits.",
        known_limits=[{"variable": "r", "limit": "0", "expected": "0"}],
        classical_limit_condition="r -> 0",
        structural_hints=["higher-order polynomials", "anharmonic corrections", "r**4 terms"]
    ),
    FeynmanProblem(
        name="Simple Pendulum Period",
        equation="theta_0 * sqrt(r / 9.8)",
        variables=["r"],
        constants={"theta_0": 2.0 * np.pi},
        target_dimension=None,
        regimes=[
            {"variable": "r", "limit": 0, "expected": "0"}
        ],
        domain="oscillatory systems",
        classical_expr="2 * pi * sqrt(r / 9.8)",
        variables_with_units={"r": "m (length)"},
        anomaly_description="Large-angle oscillations deviate from simple harmonic motion.",
        known_limits=[{"variable": "r", "limit": "0", "expected": "0"}],
        classical_limit_condition="r -> 0",
        structural_hints=["series expansions", "sin(theta) corrections", "elliptic integral approximations"]
    ),
    FeynmanProblem(
        name="Linear Motion",
        equation="v * t",
        variables=["v", "t"],
        constants={},
        target_dimension="r",
        regimes=[
            {"variable": "t", "limit": 0, "expected": "0"},
            {"variable": "v", "limit": 0, "expected": "0"}
        ],
        domain="kinematics",
        classical_expr="v * t",
        variables_with_units={"v": "m/s (velocity)", "t": "s (time)"},
        anomaly_description="Drag and deceleration force anomalies at extreme velocities.",
        known_limits=[{"variable": "t", "limit": "0", "expected": "0"}],
        classical_limit_condition="t -> 0",
        structural_hints=["exponential drag offsets", "velocity-dependent coefficients", "damping terms"]
    ),
    FeynmanProblem(
        name="Gravitational Potential Energy",
        equation="-G * m * M / r",
        variables=["m", "M", "r"],
        constants={"G": G_CODATA},
        target_dimension="E",
        regimes=[
            {"variable": "r", "limit": "oo", "expected": "0"}
        ],
        domain="astrophysics",
        classical_expr="-G * m * M / r",
        variables_with_units={"m": "kg (mass 1)", "M": "kg (mass 2)", "r": "m (distance)"},
        anomaly_description="Relativistic distortions in gravitational wells near mass boundaries.",
        known_limits=[{"variable": "r", "limit": "oo", "expected": "0"}],
        classical_limit_condition="r -> oo",
        structural_hints=["Schwarzschild radius offsets", "1/r**2 corrections", "general relativity factors"]
    ),
    FeynmanProblem(
        name="Relativistic Kinetic Energy",
        equation="0.5 * m * v**2 + theta_0 * m * v**4 / c**2",
        variables=["m", "v"],
        constants={"c": 3.0e8, "theta_0": 0.375},
        target_dimension="E",
        regimes=[
            {"variable": "v", "limit": 0, "expected": "0"}
        ],
        domain="relativistic mechanics",
        classical_expr="0.5 * m * v**2",
        variables_with_units={"m": "kg (mass)", "v": "m/s (velocity)"},
        anomaly_description="Nonlinear expansion corrections observed as velocity approaches light speed c.",
        known_limits=[{"variable": "v", "limit": "0", "expected": "0"}],
        classical_limit_condition="v -> 0",
        structural_hints=["higher-order series terms", "v**4 / c**2 factor", "Taylor expansions of gamma"]
    ),
    FeynmanProblem(
        name="Ideal Gas Law",
        equation="theta_0 * N * t",
        variables=["N", "t"],
        constants={"theta_0": K_B_CODATA},
        target_dimension=None,
        regimes=[
            {"variable": "t", "limit": 0, "expected": "0"}
        ],
        domain="thermodynamics",
        classical_expr="theta_0 * N * t",
        variables_with_units={"N": "count (particles)", "t": "K (temperature)"},
        anomaly_description="High-density condensation and attractive molecular force anomalies.",
        known_limits=[{"variable": "t", "limit": "0", "expected": "0"}],
        classical_limit_condition="t -> 0",
        structural_hints=["van der Waals corrections", "N/V density factors", "attractive terms a*N**2/V"]
    ),
    FeynmanProblem(
        name="Projectile Range",
        equation="v**2 * sin(theta_0 * t) / 9.8",
        variables=["v", "t"],
        constants={"theta_0": 1.0},
        target_dimension=None,
        regimes=[
            {"variable": "v", "limit": 0, "expected": "0"}
        ],
        domain="kinematics",
        classical_expr="v**2 * sin(t) / 9.8",
        variables_with_units={"v": "m/s (velocity)", "t": "rad (launch angle)"},
        anomaly_description="Aerodynamic drag introduces severe structural anomalies.",
        known_limits=[{"variable": "v", "limit": "0", "expected": "0"}],
        classical_limit_condition="v -> 0",
        structural_hints=["air resistance factors", "velocity exponential dampening", "polynomial offsets"]
    ),
    FeynmanProblem(
        name="Ohm's Law",
        equation="I * R",
        variables=["I", "R"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "I", "limit": 0, "expected": "0"},
            {"variable": "R", "limit": 0, "expected": "0"}
        ],
        domain="electromagnetism",
        classical_expr="I * R",
        variables_with_units={"I": "A (current)", "R": "Ohm (resistance)"},
        anomaly_description="Nonlinear resistance effects at high current densities.",
        known_limits=[{"variable": "I", "limit": "0", "expected": "0"}],
        classical_limit_condition="I -> 0",
        structural_hints=["nonlinear resistance", "I^2 corrections", "temperature-dependent R"]
    ),
    FeynmanProblem(
        name="Power Dissipation",
        equation="I**2 * R",
        variables=["I", "R"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "I", "limit": 0, "expected": "0"}
        ],
        domain="electromagnetism",
        classical_expr="I**2 * R",
        variables_with_units={"I": "A (current)", "R": "Ohm (resistance)"},
        anomaly_description="Resistive heating anomalies at high frequencies.",
        known_limits=[{"variable": "I", "limit": "0", "expected": "0"}],
        classical_limit_condition="I -> 0",
        structural_hints=["quadratic dissipation", "skin effect corrections", "harmonic distortion"]
    ),
    FeynmanProblem(
        name="Capacitor Energy",
        equation="0.5 * C * V**2",
        variables=["C", "V"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "V", "limit": 0, "expected": "0"}
        ],
        domain="electromagnetism",
        classical_expr="0.5 * C * V**2",
        variables_with_units={"C": "F (capacitance)", "V": "V (voltage)"},
        anomaly_description="Dielectric saturation at high electric fields.",
        known_limits=[{"variable": "V", "limit": "0", "expected": "0"}],
        classical_limit_condition="V -> 0",
        structural_hints=["nonlinear dielectric", "V**4 corrections", "electrostrictive effects"]
    ),
    FeynmanProblem(
        name="Wave Speed",
        equation="f * lam",
        variables=["f", "lam"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "f", "limit": 0, "expected": "0"},
            {"variable": "lam", "limit": 0, "expected": "0"}
        ],
        domain="wave mechanics",
        classical_expr="f * lam",
        variables_with_units={"f": "Hz (frequency)", "lam": "m (wavelength)"},
        anomaly_description="Dispersion effects in non-ideal media.",
        known_limits=[{"variable": "f", "limit": "0", "expected": "0"}],
        classical_limit_condition="f -> 0",
        structural_hints=["dispersive corrections", "nonlinear wave propagation", "medium-dependent velocity"]
    ),
    FeynmanProblem(
        name="Doppler Effect",
        equation="f * (1 + v / c_s)",
        variables=["f", "v"],
        constants={"c_s": 343.0},
        target_dimension=None,
        regimes=[
            {"variable": "v", "limit": 0, "expected": "f"}
        ],
        domain="acoustics",
        classical_expr="f * (1 + v / c_s)",
        variables_with_units={"f": "Hz (source frequency)", "v": "m/s (relative velocity)"},
        anomaly_description="Relativistic Doppler effects at high velocities.",
        known_limits=[{"variable": "v", "limit": "0", "expected": "f"}],
        classical_limit_condition="v -> 0",
        structural_hints=["additive velocity ratio", "relativistic correction sqrt(1-v/c)", "medium motion terms"]
    ),
    FeynmanProblem(
        name="Stefan-Boltzmann",
        equation="sigma * A * T**4",
        variables=["A", "T"],
        constants={"sigma": 5.67e-8},
        target_dimension=None,
        regimes=[
            {"variable": "T", "limit": 0, "expected": "0"}
        ],
        domain="thermodynamics",
        classical_expr="sigma * A * T**4",
        variables_with_units={"A": "m^2 (surface area)", "T": "K (temperature)"},
        anomaly_description="Emissivity corrections and quantum cavity effects.",
        known_limits=[{"variable": "T", "limit": "0", "expected": "0"}],
        classical_limit_condition="T -> 0",
        structural_hints=["T**4 power law", "emissivity epsilon corrections", "cavity radiation"]
    ),
    FeynmanProblem(
        name="Wien's Law",
        equation="b / T",
        variables=["T"],
        constants={"b": 2.898e-3},
        target_dimension=None,
        regimes=[
            {"variable": "T", "limit": "oo", "expected": "0"}
        ],
        domain="thermodynamics",
        classical_expr="b / T",
        variables_with_units={"T": "K (temperature)"},
        anomaly_description="Quantum corrections to blackbody peak wavelength.",
        known_limits=[{"variable": "T", "limit": "oo", "expected": "0"}],
        classical_limit_condition="T -> oo",
        structural_hints=["inverse temperature", "logarithmic corrections", "Planck distribution peak"]
    ),
    FeynmanProblem(
        name="Lorentz Force",
        equation="q * v * B",
        variables=["q", "v", "B"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "v", "limit": 0, "expected": "0"},
            {"variable": "q", "limit": 0, "expected": "0"}
        ],
        domain="electromagnetism",
        classical_expr="q * v * B",
        variables_with_units={"q": "C (charge)", "v": "m/s (velocity)", "B": "T (magnetic field)"},
        anomaly_description="Relativistic corrections to magnetic force at ultra-relativistic speeds.",
        known_limits=[{"variable": "v", "limit": "0", "expected": "0"}],
        classical_limit_condition="v -> 0",
        structural_hints=["cross-product velocity components", "relativistic gamma factor", "radiation reaction"]
    ),
    FeynmanProblem(
        name="Buoyancy Force",
        equation="rho * g * V_f",
        variables=["rho", "V_f"],
        constants={"g": 9.8},
        target_dimension=None,
        regimes=[
            {"variable": "rho", "limit": 0, "expected": "0"},
            {"variable": "V_f", "limit": 0, "expected": "0"}
        ],
        domain="fluid dynamics",
        classical_expr="rho * g * V_f",
        variables_with_units={"rho": "kg/m^3 (fluid density)", "V_f": "m^3 (displaced volume)"},
        anomaly_description="Non-Newtonian fluid corrections at high viscosity.",
        known_limits=[{"variable": "rho", "limit": "0", "expected": "0"}],
        classical_limit_condition="rho -> 0",
        structural_hints=["viscosity corrections", "surface tension effects", "compressibility"]
    ),
    FeynmanProblem(
        name="Lens Equation",
        equation="d_o * d_i / (d_o + d_i)",
        variables=["d_o", "d_i"],
        constants={},
        target_dimension=None,
        regimes=[
            {"variable": "d_o", "limit": "oo", "expected": "d_i"}
        ],
        domain="optics",
        classical_expr="d_o * d_i / (d_o + d_i)",
        variables_with_units={"d_o": "m (object distance)", "d_i": "m (image distance)"},
        anomaly_description="Aberration corrections in thick lens systems.",
        known_limits=[{"variable": "d_o", "limit": "oo", "expected": "d_i"}],
        classical_limit_condition="d_o -> oo",
        structural_hints=["harmonic mean form", "rational d_o*d_i/(d_o+d_i)", "aberration polynomial"]
    )
]

def get_all_problems() -> List[FeynmanProblem]:
    return PROBLEMS

def get_problem(name: str) -> FeynmanProblem:
    for problem in PROBLEMS:
        if problem.name == name:
            return problem
    raise KeyError(f"Problem '{name}' not found.")
