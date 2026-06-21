import logging
from dataclasses import dataclass
from typing import List, Union, Any, Dict, Sequence
import numpy as np
import sympy as sp

# Konfigurasi Logging Terstruktur
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARCScorer")


@dataclass
class AsymptoticRegime:
    """
    Representasi formal dari kondisi batas fisik asimtotik (Regime Bounds).
    R_k = (variable, limit_target, ground_truth_expression, importance_weight)
    """
    variable: Union[str, sp.Symbol]
    limit_target: Any  # Bisa angka numerik (0, 1) or sp.oo / -sp.oo
    ground_truth_expr: Union[str, sp.Expr]
    weight: float = 1.0

    def __post_init__(self):
        # Konversi string ke objek simbolik SymPy secara otomatis jika diperlukan
        if isinstance(self.variable, str):
            self.variable = sp.Symbol(self.variable)
        if isinstance(self.ground_truth_expr, str):
            self.ground_truth_expr = sp.sympify(self.ground_truth_expr)


def calculate_similarity(expr1: sp.Expr, expr2: sp.Expr) -> float:
    """
    Mengevaluasi kesamaan matematis struktural antara dua ekspresi aljabar
    menggunakan arsitektur verifikasi Three-Tier (Symbolic -> Divergence -> Numerical).
    """
    # --- TIER 1: VERIFIKASI SIMBOLIK EKSAK ---
    try:
        diff = sp.simplify(expr1 - expr2)
        if diff == 0:
            return 1.0
    except Exception as e:
        logger.debug(f"Tier 1 simplification split failed: {e}")

    # --- TIER 3: DETEKSI DIVERGENSI (HARD FAILURE GATE) ---
    # Jika salah satu menuju tak hingga/tak terdefinisi sedangkan yang lain bernilai konstan
    inf_tokens = [sp.oo, -sp.oo, sp.zoo]
    is_inf1 = expr1 in inf_tokens or getattr(expr1, "is_infinite", False)
    is_inf2 = expr2 in inf_tokens or getattr(expr2, "is_infinite", False)

    if is_inf1 != is_inf2:
        return 0.0
    if is_inf1 and is_inf2:
        return 1.0 if expr1 == expr2 else 0.0

    # --- TIER 2: EVALUASI KEDEKATAN NUMERIK (FALLBACK STRATEGY) ---
    # Jika penyederhanaan aljabar buntu akibat fungsi transendental non-elementer,
    # lakukan sampling 100 titik acak pada variabel konstanta fisis tersisa (e.g., m, c, G, M).
    free_symbols = expr1.free_symbols.union(expr2.free_symbols)
    
    if not free_symbols:
        try:
            val1 = float(expr1.evalf())
            val2 = float(expr2.evalf())
            if np.isnan(val1) or np.isnan(val2):
                return 0.0
            rel_error = abs(val1 - val2) / (abs(val2) + 1e-9)
            return float(np.exp(-rel_error))
        except Exception:
            return 0.0

    # Generator angka acak yang konsisten (seeded untuk stabilitas testing)
    rng = np.random.default_rng(42)
    symbols_list = list(free_symbols)
    errors = []

    for _ in range(100):
        # Berikan nilai fisis positif acak yang masuk akal [0.5, 2.0] untuk parameter tersisa
        sample_vals = rng.uniform(0.5, 2.0, size=len(symbols_list))
        subs_dict = dict(zip(symbols_list, sample_vals))
        
        try:
            val1 = float(expr1.subs(subs_dict).evalf())
            val2 = float(expr2.subs(subs_dict).evalf())
            
            if np.isinf(val1) or np.isinf(val2) or np.isnan(val1) or np.isnan(val2):
                return 0.0
                
            rel_error = abs(val1 - val2) / (abs(val2) + 1e-9)
            errors.append(rel_error)
        except Exception:
            return 0.0

    if not errors:
        return 0.0

    mean_relative_error = np.mean(errors)
    return float(np.exp(-mean_relative_error))


def _parse_limit_tokens(
    limit_variables: Union[str, Sequence[str]],
    limit_directions: Union[str, Sequence[str]],
) -> tuple[List[str], List[str]]:
    """Parse comma-separated or sequence limit specs into aligned variable/direction lists."""
    if isinstance(limit_variables, str):
        vars_list = [v.strip() for v in limit_variables.split(",") if v.strip()]
    else:
        vars_list = [str(v).strip() for v in limit_variables]

    if isinstance(limit_directions, str):
        dirs_list = [d.strip() for d in limit_directions.split(",") if d.strip()]
    else:
        dirs_list = [str(d).strip() for d in limit_directions]

    if not vars_list:
        raise ValueError("At least one limit variable is required.")

    if not dirs_list:
        dirs_list = ["0"]
    if len(dirs_list) < len(vars_list):
        dirs_list.extend([dirs_list[-1]] * (len(vars_list) - len(dirs_list)))
    elif len(dirs_list) > len(vars_list):
        dirs_list = dirs_list[: len(vars_list)]

    return vars_list, dirs_list


def build_arc_regimes(
    limit_variables: Union[str, Sequence[str]],
    limit_directions: Union[str, Sequence[str]] = "0",
    ground_truth_expr: Union[str, sp.Expr] = "0",
    weight: float = 1.0,
) -> List[AsymptoticRegime]:
    """
    Build ARC asymptotic regimes for one or more limit variables.

    Supports multi-variable corrections Δ(x₁, x₂, …) by specifying comma-separated
    limits, e.g. limit_variables="x,y" and limit_directions="0,oo".
    """
    vars_list, dirs_list = _parse_limit_tokens(limit_variables, limit_directions)
    regimes: List[AsymptoticRegime] = []
    for var, direction in zip(vars_list, dirs_list):
        limit_target = sp.oo if direction == "oo" else 0
        regimes.append(
            AsymptoticRegime(
                variable=sp.Symbol(var),
                limit_target=limit_target,
                ground_truth_expr=ground_truth_expr,
                weight=weight,
            )
        )
    return regimes


def _resolve_limit(candidate: sp.Expr, variable: sp.Symbol, limit_target: Any):
    """Compute lim_{variable -> limit_target}(candidate), robust to undetermined-sign parameters.

    SymPy's ``sp.limit`` raises ``NotImplementedError`` for limits whose result
    depends on the sign of a free parameter, e.g. ``lim_{r->oo} exp(-r/theta_1)``
    is unresolved when ``theta_1`` is an unsigned symbol. This is the dominant
    failure mode for exponential-family corrections at ``-> oo`` regimes (Yukawa,
    screened Coulomb) and previously caused the ARC gate to reject ~70% of
    candidates including the literal ground-truth form.

    All ADCD fit parameters (``theta_i``) are positive scale parameters — the
    JAX optimizer log-parameterizes them to enforce positivity — so we declare
    every ``theta_*`` symbol positive and retry. If the limit still fails, the
    caller treats the candidate as rejected (returns ``None``).
    """
    try:
        return sp.limit(candidate, variable, limit_target)
    except Exception:
        pass
    # Retry assuming every theta_* free symbol is strictly positive.
    theta_syms = [s for s in candidate.free_symbols if str(s).startswith("theta_")]
    if not theta_syms:
        return None
    try:
        positive_map = {s: sp.Symbol(str(s), positive=True) for s in theta_syms}
        pos_candidate = candidate.subs(positive_map)
        return sp.limit(pos_candidate, variable, limit_target)
    except Exception:
        return None


class ARCScorer:
    """
    Mesin utama Stage 1 Gatekeeper untuk menghitung bobot kelayakan
    struktur asimtotik formula kandidat dari LLM sebelum diteruskan ke graf JAX.
    """
    def __init__(self, regimes: List[AsymptoticRegime]):
        if not regimes:
            raise ValueError("Daftar kondisi batas (regimes) tidak boleh kosong.")
        self.regimes = regimes
        self.total_weight = sum(r.weight for r in regimes)

    def score(self, candidate_expr: Union[str, sp.Expr], constants: Dict[str, float] = None) -> float:
        """
        Menghitung nilai akhir ARC Score untuk satu kandidat fungsi.
        Menggunakan evaluasi limit matematis murni tanpa pencocokan string biasa.
        """
        try:
            candidate = sp.sympify(candidate_expr)
            if constants:
                subs_dict = {sp.Symbol(k): v for k, v in constants.items() if sp.Symbol(k) in candidate.free_symbols}
                if subs_dict:
                    candidate = candidate.subs(subs_dict)
        except Exception as e:
            logger.error(f"Gagal memproses sintaks ekspresi kandidat: {e}")
            return 0.0

        weighted_similarity_sum = 0.0

        for r in self.regimes:
            evaluated_limit = _resolve_limit(candidate, r.variable, r.limit_target)
            if evaluated_limit is None:
                # Kasus kegagalan matematis kritis (seperti PoleError) langsung diberi penalti 0
                logger.warning(f"Kegagalan komputasi limit pada variabel {r.variable}")
                continue
            # Hitung skor kedekatan fisis limit kandidat vs ground truth boundary
            similarity = calculate_similarity(evaluated_limit, r.ground_truth_expr)
            weighted_similarity_sum += r.weight * similarity
            logger.debug(f"Regime {r.variable}->{r.limit_target} | Limit: {evaluated_limit} | Sim: {similarity}")

        return weighted_similarity_sum / self.total_weight
