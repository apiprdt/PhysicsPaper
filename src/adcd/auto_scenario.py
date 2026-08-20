"""
auto_scenario.py -- Zero-Leakage Deterministic Scenario Builder for ADCD (v2, hardened)

This module implements a strictly deterministic, rule-based ingestion layer for
converting raw CSV observational data into valid ADCD AnomalyScenario instances,
with NO learned neural weights and NO generative LLM involvement anywhere in the
pipeline.

Changes from v1 (see accompanying audit report for full rationale):
  Bug #1 (critical): verify_name_invariance previously scrambled headers using
      the CANONICAL unit string instead of the raw matched token, causing the
      audit itself to report false negatives for suffix-style headers
      (e.g. "radius_km") -- exactly the format used in production. Fixed by
      threading the raw matched token through parse_header() and scrambling
      with THAT, not with the normalized output.
  Bug #2 (critical): unrecognized units silently fell back to "dimensionless"
      with scale=1.0, and the variable name was left uncleaned. This let
      out-of-vocabulary units pass Gate 2 (dimensional consistency) as if they
      were validated dimensionless quantities. Fixed: parse_header() now
      raises UnitParseError by default (fail loud). A strict=False escape
      hatch exists for callers that want to surface ambiguity to a human
      instead of crashing.
  Bug #3: AutoCSVScenario.generate_data() computed the multiplicative residual
      as y_obs/y_classical instead of y_obs/y_classical - 1, which does not
      match the Delta definition in ADCD's own Eq. 1. Fixed.
  Bug #4: short ambiguous suffix tokens (min, m, s, g, k, n, c, e, hr, day)
      could silently collide with ordinary English word endings in variable
      names (e.g. "sample_min" parsed as minutes). These are now only
      resolved via explicit bracket notation, never via bare suffix matching.
  Bug #5: only the FIRST bracket group in a header was checked; if it wasn't
      a recognized unit, later valid bracket groups were ignored entirely.
      Fixed: all bracket groups are tried in order.
  New: unit resolution is now backed by `pint` (a mature, deterministic,
      non-ML unit-algebra library) instead of a hand-maintained dictionary,
      to get close to universal coverage of unit strings/prefixes without
      manually enumerating every combination. ADCD's own 5D (M,L,T,Theta,Q)
      dimensional representation is preserved exactly -- pint is used only
      to resolve arbitrary unit strings down to SI base units; the mapping
      from SI base units to ADCD's charge-based 5D vector is done by ADCD's
      own deterministic linear transform (A = Q/T), not by pint.
  New: domain string is validated against DOMAIN_TAXONOMY at construction
      time (fail loud) instead of silently disabling the taxonomy prior,
      closing the same class of silent-fallback bug found in the hand-written
      scenario definitions.
"""

from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
import sympy as sp
from typing import Dict, List, Optional, Tuple, Set

from adcd.anomaly_scenarios import AnomalyScenario
from adcd.quickfit import DOMAIN_TAXONOMY

# --- Fix (dependency hygiene): fail with an actionable message at import
# time instead of a bare ModuleNotFoundError deep inside a call stack.
try:
    import pint
except ImportError as e:
    raise ImportError(
        "auto_scenario.py requires the 'pint' package for near-universal "
        "unit parsing. Install it with:\n"
        "    pip install pint --break-system-packages\n"
        "and add 'pint' to requirements.txt / pyproject.toml before "
        "distributing this module."
    ) from e


# ============================================================================
# 0. ERRORS
# ============================================================================

class UnitParseError(ValueError):
    """Raised when a header's unit cannot be resolved with confidence.
    Deliberately NOT caught anywhere in this module -- the caller (a human,
    or an explicit strict=False call site) must decide what to do next.
    """
    pass


class UnsupportedDimensionError(UnitParseError):
    """Raised when a unit resolves fine (pint recognizes it) but involves a
    dimension ADCD's 5D (M,L,T,Theta,Q) engine cannot represent (mol,
    candela). Distinct from UnitParseError (genuinely unrecognized string)
    because the correct handling differs: an unsupported-dimension column
    can be gracefully excluded from the dimensional pipeline rather than
    aborting ingestion of the whole CSV (see AutoCSVScenario.__init__).
    """
    pass


class DomainTaxonomyError(ValueError):
    """Raised when `domain` does not match any DOMAIN_TAXONOMY key, which
    would otherwise silently disable the taxonomy prior (see Section 5,
    Discussion, "Taxonomy as prior, not hint" -- silent bypass of this
    mechanism defeats the entire point of a domain-guided search).
    """
    pass


# ============================================================================
# 1. UNIT RESOLUTION (pint-backed, near-universal, still zero learned weights)
# ============================================================================

_UREG = pint.UnitRegistry()

# Tokens that are only resolved via EXPLICIT bracket/paren notation, never
# via bare underscore-suffix matching, because they collide with common
# English word endings in variable names (Bug #4).
AMBIGUOUS_BARE_SUFFIXES = {
    "m", "s", "g", "k", "n", "c", "e", "min", "hr", "day", "a", "j", "t",
}

# Underscore-style compound notation ("km_s", "m_s2") that pint does not
# parse natively (pint expects "km/s", "m/s**2"). This table normalizes the
# small, closed set of compound patterns ADCD scenarios actually use before
# handing the string to pint. Anything not covered here that still contains
# an underscore falls through to bare single-token suffix matching.
_COMPOUND_SUFFIX_TO_PINT = {
    "m_s": "m/s", "km_s": "km/s", "km_h": "km/h", "kms": "km/s", "kmh": "km/h",
    "m_s2": "m/s**2", "m_s_2": "m/s**2", "km_s2": "km/s**2",
    "ms2": "m/s**2", "ms_2": "m/s**2",
}


def _pint_dimensionality_to_adcd_5d(quantity) -> Tuple[int, int, int, int, int]:
    """
    Deterministic linear transform from pint's 7-base-SI dimensionality to
    ADCD's charge-based 5D (M, L, T, Theta, Q) vector.

    pint treats electrical dimension via [current] (Ampere-based, SI's actual
    7th base unit). ADCD's dimensional_checker.py instead treats charge [Q]
    as the fundamental axis (see paper Sec. 3.3 -- this is what makes the
    q1/q2 pairing and the theta_4 subscript story for Screened Coulomb work).

    The two are related exactly by A = Q / T (current = charge / time), so
    for any dimensionality with exponents (m, l, t, i, theta, n, lum):
        M     = m
        L     = l
        T_adcd = t - i      (substituting A^i = Q^i * T^-i)
        Theta = theta
        Q     = i
    This is exact dimensional algebra, not an approximation -- verified
    against Newton, Joule, Tesla, Volt, Ohm, Farad in the accompanying tests.

    Raises UnitParseError if the unit involves [substance] (mol) or
    [luminous_intensity] (candela), which ADCD's 5D engine does not model.
    """
    d = quantity.dimensionality
    m_dim = d.get("[mass]", 0)
    l_dim = d.get("[length]", 0)
    t_dim = d.get("[time]", 0)
    i_dim = d.get("[current]", 0)
    theta_dim = d.get("[temperature]", 0)
    n_dim = d.get("[substance]", 0)
    lum_dim = d.get("[luminous_intensity]", 0)

    if n_dim != 0 or lum_dim != 0:
        raise UnsupportedDimensionError(
            f"Unit has mol/candela dimensionality ({d}), which ADCD's 5D "
            f"(M,L,T,Theta,Q) engine does not model. This column will be "
            f"excluded from the dimensional pipeline rather than blocking "
            f"the whole CSV -- see AutoCSVScenario for graceful handling."
        )

    M, L, T, Theta, Q = m_dim, l_dim, (t_dim - i_dim), theta_dim, i_dim
    return (M, L, T, Theta, Q)


class UnitExtractor:
    """
    Deterministic unit extraction from column header strings.

    Resolution order (each step tried in order, first match wins):
      1. ALL bracket/paren/brace groups in the header, tried left to right
         (fixes Bug #5 -- previously only the first group was tried).
      2. Underscore-suffix compound notation via a small closed lookup
         (km_s, m_s2, ...) normalized to pint syntax.
      3. Underscore-suffix single-token notation via bare pint lookup,
         EXCLUDING ambiguous short tokens (fixes Bug #4).
      4. Whole-string match.
      5. UnitParseError (fail loud) -- or, if strict=False, an explicit
         "AMBIGUOUS" sentinel for the caller to route to a human.

    No numeric data is ever read by this class -- only header strings.
    """

    @staticmethod
    def _find_all_brackets(header: str) -> List[str]:
        return re.findall(r"[\(\[\{]([^\)\]\}]+)[\)\]\}]", header)

    @staticmethod
    def _strip_all_brackets(header: str) -> str:
        return re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", "", header).strip().rstrip("_").strip()

    @staticmethod
    def _try_pint(token: str):
        """Returns (5d_vector, scale_to_si, canonical_si_str) or None."""
        try:
            q = _UREG(token)
        except Exception:
            return None
        try:
            vec = _pint_dimensionality_to_adcd_5d(q)
        except UnitParseError:
            raise
        base = q.to_base_units()
        return vec, float(base.magnitude), str(base.units)

    @staticmethod
    def parse_header(
        header: str,
        strict: bool = True,
        trusted_bare_suffixes: Optional[Set[str]] = None,
    ):
        """
        Returns (clean_var_name, canonical_si_unit, dim_vector_5d, scale_to_si,
                 raw_matched_token).

        raw_matched_token is the literal substring that was resolved (e.g.
        "km/s", not "m/s") -- this is threaded through specifically so
        verify_name_invariance() can scramble using the SAME raw token the
        real parser would see, instead of a post-normalization artifact
        (this was the root cause of Bug #1).

        trusted_bare_suffixes: explicit, per-call opt-in for short tokens
        that are normally refused (Bug #4 mitigation). Bracket notation
        ("radius [m]") is always unambiguous and never needs this. This
        parameter exists because blanket-refusing ALL bare short suffixes
        makes perfectly ordinary headers like "radius_m" or "time_s" fail,
        which is a real UX cost -- see accompanying trade-off discussion.
        Passing e.g. {"m", "s"} tells the parser "in THIS dataset's naming
        convention, I know these bare suffixes are units, not word endings."
        Every use emits a warning so it is visible in logs, not silent.
        """
        header_clean = header.strip()
        trusted = trusted_bare_suffixes or set()

        # 1. All bracket groups, in order (fix Bug #5) -- always unambiguous,
        # never subject to the short-token restriction.
        for unit_str in UnitExtractor._find_all_brackets(header_clean):
            token = unit_str.strip()
            result = UnitExtractor._try_pint(token)
            if result is not None:
                vec, scale, canon = result
                var_name = UnitExtractor._strip_all_brackets(header_clean)
                return var_name, canon, vec, scale, token

        # 2 & 3. Underscore suffix (compound lookup, then bare single token)
        if "_" in header_clean:
            parts = header_clean.split("_")
            for k in range(len(parts) - 1, 0, -1):
                suffix = "_".join(parts[-k:]).strip().lower()

                if suffix in _COMPOUND_SUFFIX_TO_PINT:
                    pint_token = _COMPOUND_SUFFIX_TO_PINT[suffix]
                    result = UnitExtractor._try_pint(pint_token)
                    if result is not None:
                        vec, scale, canon = result
                        var_name = "_".join(parts[:-k]).strip().rstrip("_")
                        return var_name, canon, vec, scale, suffix

                if k == 1 and suffix in AMBIGUOUS_BARE_SUFFIXES:
                    if suffix in trusted:
                        result = UnitExtractor._try_pint(suffix)
                        if result is not None:
                            vec, scale, canon = result
                            var_name = "_".join(parts[:-k]).strip().rstrip("_")
                            warnings.warn(
                                f"Header '{header}': resolved ambiguous bare "
                                f"suffix '{suffix}' as a unit because it was "
                                f"explicitly trusted via trusted_bare_suffixes. "
                                f"Verify this is correct for every column using "
                                f"this suffix in this dataset.",
                                stacklevel=2,
                            )
                            return var_name, canon, vec, scale, suffix
                    continue  # refuse to guess on collision-prone bare tokens (fix Bug #4)

                if k == 1:
                    result = UnitExtractor._try_pint(suffix)
                    if result is not None:
                        vec, scale, canon = result
                        var_name = "_".join(parts[:-k]).strip().rstrip("_")
                        return var_name, canon, vec, scale, suffix

        # 4. Whole-string match
        result = UnitExtractor._try_pint(header_clean)
        if result is not None:
            vec, scale, canon = result
            return header_clean, canon, vec, scale, header_clean

        # 5. Fail loud (fix Bug #2)
        var_name_cleaned = UnitExtractor._strip_all_brackets(header_clean) or header_clean
        if strict:
            raise UnitParseError(
                f"Could not resolve a unit for header '{header}'. Refusing to "
                f"silently default to dimensionless. Options:\n"
                f"  1. Rename the column to use bracket notation, e.g. "
                f"'{var_name_cleaned} [unit]' (always unambiguous).\n"
                f"  2. If this header ends in a short suffix ({sorted(AMBIGUOUS_BARE_SUFFIXES)}) "
                f"that you are certain denotes a unit in this dataset, pass "
                f"trusted_bare_suffixes={{'<suffix>'}} explicitly."
            )
        return var_name_cleaned, "AMBIGUOUS", None, None, None


def verify_name_invariance(headers: List[str]) -> bool:
    """
    Feynman-style falsification audit (fixed -- see Bug #1).

    Replaces every variable name with an arbitrary gibberish token while
    preserving the RAW matched unit token (not its post-normalization form),
    then verifies UnitExtractor produces an identical (dim_vector, scale)
    regardless of whether the name is a physics word ('velocity') or noise
    ('xJ9'). This is a genuine test of "does the parser look at the name at
    all", unlike the v1 version, which accidentally could not pass for the
    project's own primary header format (suffix-style, no brackets).
    """
    originals = [UnitExtractor.parse_header(h, strict=False) for h in headers]

    scrambled_headers = []
    for i, (_, _, _, _, raw_token) in enumerate(originals):
        if raw_token is None:
            scrambled_headers.append(f"token_{i}")
        else:
            scrambled_headers.append(f"token_{i}_{raw_token}")

    scrambled = [UnitExtractor.parse_header(h, strict=False) for h in scrambled_headers]

    for (_, canon_o, vec_o, scale_o, _), (_, canon_s, vec_s, scale_s, _) in zip(originals, scrambled):
        if canon_o != canon_s or vec_o != vec_s or scale_o != scale_s:
            return False
    return True


# ============================================================================
# 2. DETERMINISTIC AUTO-SCENARIO FROM CSV
# ============================================================================

class AutoCSVScenario(AnomalyScenario):
    """A 100% deterministic AnomalyScenario backed directly by a real CSV dataset."""

    def __init__(
        self,
        name: str,
        csv_path: str,
        target_col: str,
        classical_expr: str,
        domain: str,
        classical_constants: Optional[Dict[str, float]] = None,
        classical_limit_variable: str = "x",
        classical_limit_direction: str = "0",
        correction_type: str = "additive",
        correction_class: str = "polynomial",
        anomaly_regime: str = "observational_data",
        strict_units: bool = True,
        trusted_bare_suffixes: Optional[Set[str]] = None,
        engine: str = "python",
    ):
        # `domain` is now REQUIRED (no silent default) and validated (fix:
        # the same silent-taxonomy-bypass class of bug found in the
        # hand-written scenarios -- see audit report).
        if domain not in DOMAIN_TAXONOMY:
            raise DomainTaxonomyError(
                f"domain='{domain}' does not match any DOMAIN_TAXONOMY key "
                f"({sorted(DOMAIN_TAXONOMY.keys())}). Using an unrecognized "
                f"domain string would silently disable the taxonomy prior "
                f"instead of raising -- refusing to proceed. Pick a "
                f"registered key or add a new taxonomy entry with a "
                f"peer-reviewed justification first."
            )

        self.csv_path = csv_path
        self.target_col_raw = target_col

        df = pd.read_csv(csv_path)
        self._df_raw = df

        parsed_cols = {}
        variables_with_units = {}
        self._scale_factors = {}
        self._canonical_cols = {}
        self.excluded_columns: Dict[str, str] = {}  # col -> reason, for graceful degradation

        for col in df.columns:
            try:
                clean_name, canon_unit, dim_vec, scale, _raw = UnitExtractor.parse_header(
                    col, strict=strict_units, trusted_bare_suffixes=trusted_bare_suffixes
                )
            except UnsupportedDimensionError as e:
                # Graceful degradation (mitigates trade-off #2): a column
                # with mol/candela dimensionality doesn't abort ingestion of
                # the whole CSV. It's excluded from the dimensional pipeline
                # and cannot be used in classical_expr; everything else
                # proceeds. If the excluded column IS actually needed by
                # classical_expr, that fails later, explicitly, at
                # expression-evaluation time -- not silently here.
                if col == target_col:
                    raise UnitParseError(
                        f"Target column '{col}' has mol/candela dimensionality, "
                        f"which ADCD's 5D engine cannot represent as a target. "
                        f"This cannot be gracefully degraded -- choose a "
                        f"different target or extend dimensional_checker.py."
                    ) from e
                warnings.warn(
                    f"Column '{col}' excluded from dimensional analysis: {e}",
                    stacklevel=2,
                )
                self.excluded_columns[col] = str(e)
                continue

            valid_name = re.sub(r"\W|^\d", "_", clean_name)
            parsed_cols[col] = valid_name
            variables_with_units[valid_name] = canon_unit
            self._scale_factors[valid_name] = scale if scale is not None else 1.0
            self._canonical_cols[col] = valid_name

        feature_cols = [v for k, v in parsed_cols.items() if k != target_col]

        super().__init__(
            name=name,
            tier="observational",
            domain=domain,
            classical_expr=classical_expr,
            classical_variables=feature_cols,
            classical_constants=classical_constants or {},
            correction_type=correction_type,
            correction_expr="0.0",  # ground truth not assumed for real data
            correction_constants={},
            anomaly_regime=anomaly_regime,
            variables_with_units=variables_with_units,
            classical_limit_variable=classical_limit_variable,
            classical_limit_direction=classical_limit_direction,
            correction_class=correction_class,
            engine=engine,
        )

    def generate_data(
        self, n_points: int = 200, noise_level: float = 0.0, seed: int = 42, domain_max: float = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        df = self._df_raw.copy()
        X = {}

        for raw_col, clean_name in self._canonical_cols.items():
            if raw_col == self.target_col_raw:
                continue
            values = df[raw_col].to_numpy(dtype=float)
            scale = self._scale_factors.get(clean_name, 1.0)
            X[clean_name] = values * scale

        all_vars = {**X, **self.classical_constants}
        expr = sp.sympify(self.classical_expr)
        f_lamb = sp.lambdify(list(all_vars.keys()), expr, modules="numpy")
        y_classical = np.array(f_lamb(**all_vars), dtype=float)
        if y_classical.shape == ():
            y_classical = np.full(len(df), y_classical)

        target_clean = self._canonical_cols.get(self.target_col_raw, self.target_col_raw)
        target_scale = self._scale_factors.get(target_clean, 1.0)
        y_obs = df[self.target_col_raw].to_numpy(dtype=float) * target_scale

        if self.correction_type == "multiplicative":
            # Delta = y_obs / y_classical - 1  (matches paper Eq. 1 -- fix Bug #3,
            # v1 was missing the "- 1" and returned a raw ratio instead of a
            # correction that vanishes at the classical limit)
            residual = np.where(y_classical != 0, y_obs / y_classical - 1.0, y_obs)
        else:
            residual = y_obs - y_classical

        return X, y_obs, y_classical, residual


def build_scenario_from_csv(
    csv_path: str,
    scenario_name: str,
    target_col: str,
    classical_expr: str,
    domain: str,
    classical_constants: Optional[Dict[str, float]] = None,
    classical_limit_variable: str = "x",
    classical_limit_direction: str = "0",
    strict_units: bool = True,
    trusted_bare_suffixes: Optional[Set[str]] = None,
    engine: str = "python",
) -> AutoCSVScenario:
    """Factory function to construct an AutoCSVScenario from a raw CSV file.

    `domain` is required and must be a valid DOMAIN_TAXONOMY key -- this is
    the one input that stays firmly in human hands (see prior discussion:
    domain/taxonomy selection is the highest-leverage point for accidental
    semantic leakage, so it is never inferred automatically).

    `trusted_bare_suffixes`: optional explicit opt-in for short ambiguous
    unit suffixes (e.g. {"m", "s"}) for datasets whose naming convention you
    already know is unambiguous. Leave unset to require bracket notation
    for anything short/ambiguous -- the safer default.

    `engine`: "python" (default, legacy JAX + SymPy pipeline) or "julia"
    (new ADCDEngine.jl with expanded 6-pattern grammar, no JAX recompilation).
    """
    return AutoCSVScenario(
        name=scenario_name,
        csv_path=csv_path,
        target_col=target_col,
        classical_expr=classical_expr,
        domain=domain,
        classical_constants=classical_constants,
        classical_limit_variable=classical_limit_variable,
        classical_limit_direction=classical_limit_direction,
        strict_units=strict_units,
        trusted_bare_suffixes=trusted_bare_suffixes,
        engine=engine,
    )
