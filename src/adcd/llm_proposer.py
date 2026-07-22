import numpy as np
import logging
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional, Any
import sympy as sp

logger = logging.getLogger(__name__)

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


class BaseProposer(ABC):
    @abstractmethod
    def propose(self, context: ProposalContext) -> List[str]:
        """Generate candidate equation strings"""
        pass

class MockProposer(BaseProposer):
    """
    Stochastic and template-driven equation generator for testing and benchmarks.
    Generates candidates from a curated template bank, stochastic mutations,
    and refinements of previous best equations.
    """
    def __init__(self, seed: int = 42, extended: bool = False):
        self.seed = seed
        self.extended = extended
        self._templates = [
            "theta_0 * {v1} * {v2}**2",
            "theta_0 * {v2} * {v1}**2",
            "theta_0 * {v1} * {v2}**theta_1",
            "theta_0 * {v1} / {v2}**theta_1",
            "theta_0 * {v1}**theta_1",
            "theta_0 * {v1} * {v2}",
            "theta_0 * {v1} / {v2}**2",
            "theta_0 * {v1} * {v2} / {v3}**theta_1",
            "theta_0 * {v1} * {v2} / {v3}**2",
            "theta_0 * {v1} * sin(theta_1 * {v2})",
            "theta_0 * {v1}**2 * sin(theta_1 * {v2})",
            "theta_0 * {v1}**2 * sin(theta_1 * {v2}) / theta_2",
            "theta_0 * sqrt({v1})",
            "theta_0 * {v1} * {v2}**2 + theta_3 * {v1} * {v2}**4 / {v4}**2",
            "theta_0 * {v1} * {v2} * {v3}**theta_1",
            "theta_0 * {v1} * (1 + theta_1 * {v2} / {v3})",
            "theta_0 * {v1} * {v2} / ({v1} + {v2})",
            "theta_0 * {v1} * {v2} / (theta_1 * {v1} + theta_2 * {v2})"
        ]
        if self.extended:
            self._templates.extend([
                "theta_0 * ({v1} / {v2}) * exp(-{v3} / theta_1)",
                "theta_0 * {v1} * exp(-{v2} / theta_1)",
                "theta_0 * ({v1} / {v2}) * exp(-{v3} / theta_1) - 1.0"
            ])

    def propose(self, context: ProposalContext) -> List[str]:
        rng = np.random.RandomState(self.seed + context.iteration)
        vars_available = list(context.variable_names)
        if context.constants:
            for const_name in context.constants:
                if const_name not in vars_available and not const_name.startswith("theta_"):
                    vars_available.append(const_name)
        # Build locals dictionary to avoid clashes with SymPy builtins (like N)
        sym_locals = {}
        for var in vars_available:
            sym_locals[var] = sp.Symbol(var)
        for i in range(100):
            sym_locals[f"theta_{i}"] = sp.Symbol(f"theta_{i}")

        if context.classical_expr:
            try:
                expr = sp.sympify(context.classical_expr, locals=sym_locals)
                for sym in expr.free_symbols:
                    sym_str = str(sym)
                    if sym_str not in vars_available and sym_str not in ("pi", "E", "I") and not sym_str.startswith("theta_"):
                        vars_available.append(sym_str)
                        sym_locals[sym_str] = sp.Symbol(sym_str)
            except Exception:
                pass

        # ── 0. Deterministic physics-constant injection ──────────────────────
        # When non-theta constants are present (e.g. G, k_B, k_e), exhaustively
        # build forms that multiply the constant with every ordered 1- and 2-var
        # combination from the problem variables and divide by each var**1 and
        # var**2.  These are always included before the random templates so they
        # reach Stage 2 regardless of the random seed.
        injected: list = []
        non_theta_constants = [c for c in (context.constants or {}) if not c.startswith("theta_")]
        phys_vars = list(context.variable_names)  # only problem variables, not constants
        if non_theta_constants and phys_vars:
            for const in non_theta_constants:
                # const alone (scaled by theta_0)
                injected.append(f"theta_0 * {const}")
                # const * v1
                for v1 in phys_vars:
                    injected.append(f"theta_0 * {const} * {v1}")
                    injected.append(f"theta_0 * {const} / {v1}**2")
                    injected.append(f"theta_0 * {const} / {v1}")
                    injected.append(f"theta_0 * {v1} / {const}")
                    # const * v1 * v2 / v3**n
                    for v2 in phys_vars:
                        if v2 == v1:
                            continue
                        injected.append(f"theta_0 * {const} * {v1} * {v2}")
                        for v3 in phys_vars:
                            if v3 == v1 or v3 == v2:
                                continue
                            injected.append(f"theta_0 * {const} * {v1} * {v2} / {v3}**2")
                            injected.append(f"theta_0 * {const} * {v1} * {v2} / {v3}")
                        
                        # Inject special constant-bound patterns (Doppler and Stefan-Boltzmann)
                        injected.append(f"theta_0 * {const} * {v1} * {v2}**4")
                        injected.append(f"theta_0 * {const} * {v2} * {v1}**4")
                        injected.append(f"theta_0 * {v1} * (1 + theta_1 * {v2} / {const})")
                        injected.append(f"theta_0 * {v2} * (1 + theta_1 * {v1} / {const})")

        # Also inject all full-variable products (catches Ideal Gas N*k_B*t pattern
        # where all problem variables must appear).
        if len(phys_vars) >= 2:
            # 2-variable product
            for i, v1 in enumerate(phys_vars):
                for v2 in phys_vars[i+1:]:
                    injected.append(f"theta_0 * {v1} * {v2}")
                    injected.append(f"theta_0 * {v1} * {v2}**2")
                    injected.append(f"theta_0 * {v2} * {v1}**2")
                    injected.append(f"theta_0 * {v1} * {v2}**theta_1")
                    injected.append(f"theta_0 * {v2} * {v1}**theta_1")
                    injected.append(f"theta_0 * {v1}**2 * sin(theta_1 * {v2})")
                    injected.append(f"theta_0 * {v2}**2 * sin(theta_1 * {v1})")
            # 3-variable product
            if len(phys_vars) >= 3:
                for i, v1 in enumerate(phys_vars):
                    for j, v2 in enumerate(phys_vars[i+1:], i+1):
                        for v3 in phys_vars[j+1:]:
                            injected.append(f"theta_0 * {v1} * {v2} * {v3}")
                            
        # Task P1-4: Extended Bivariate Templates
        if self.extended and len(phys_vars) >= 2:
            v1, v2 = phys_vars[0], phys_vars[1]
            bivariate_templates = [
                # Bivariate polynomial (ARC-safe: vanish at v1→0 AND v2→0)
                f"theta_0 * ({v1} / theta_1) * ({v2} / theta_2)",
                f"theta_0 * ({v1} / theta_1)**2 * ({v2} / theta_2)",
                f"theta_0 * ({v1} / theta_1)**theta_3 * ({v2} / theta_2)**theta_4",
                
                # Bivariate exponential-linear (ARC-safe)
                f"theta_0 * ({v1} / theta_1) * exp(-{v2} / theta_2)",
                f"theta_0 * ({v2} / theta_1) * exp(-{v1} / theta_2)",
                
                # Bivariate with (1+R) factor (ARC-safe: factor → 1 at limit, multiplied by 0)
                f"theta_0 * ({v1} / theta_1) * (1.0 + {v2} / theta_2)",
                f"exp(-{v1} / theta_0) * ({v1} / theta_1) * (1.0 + {v2} / theta_2)",
                
                # Bivariate rational (ARC-safe with proper limit)
                f"theta_0 * ({v1} / theta_1) / (1.0 + {v1} / theta_1) * ({v2} / theta_2)",
            ]
            for bt in bivariate_templates:
                injected.append(bt)

        # 1. Generate from Template Bank (~30%)
        n_templates = int(context.n_candidates * 0.3)
        template_candidates = []
        for _ in range(n_templates * 2):  # Oversample to filter valid ones
            template = rng.choice(self._templates)
            needed_vars = 1
            if "{v2}" in template: needed_vars = 2
            if "{v3}" in template: needed_vars = 3
            if "{v4}" in template: needed_vars = 4

            if len(vars_available) >= needed_vars:
                chosen_vars = rng.choice(vars_available, size=needed_vars, replace=False)
                fmt_dict = {f"v{i+1}": chosen_vars[i] for i in range(needed_vars)}
                expr_str = template.format(**fmt_dict)
                template_candidates.append(expr_str)
            else:
                chosen_var = rng.choice(vars_available)
                template_candidates.append(f"theta_0 * {chosen_var}")

        # 2. Refinements of Previous Best (~20%)
        n_refine = int(context.n_candidates * 0.2)
        refine_candidates = []
        if context.previous_best:
            sorted_prev = sorted(context.previous_best, key=lambda x: x[1])
            for _ in range(n_refine):
                base_expr, _ = sorted_prev[rng.choice(len(sorted_prev))]
                mutated = self._mutate_exponent(base_expr, rng)
                refine_candidates.append(mutated)
        else:
            n_templates += n_refine

        # 3. Stochastic Mutations (~40%)
        n_mutations = int(context.n_candidates * 0.4)
        mutation_candidates = []
        bases = template_candidates[:n_mutations]
        if not bases:
            bases = [f"theta_0 * {rng.choice(vars_available)}" for _ in range(n_mutations)]
        while len(bases) < n_mutations:
            bases.append(rng.choice(bases))

        for base in bases:
            mut_type = rng.choice(["operator", "exponent", "add_term", "simplify_term"])
            if mut_type == "operator":
                mutated = self._mutate_operator(base, rng)
            elif mut_type == "exponent":
                mutated = self._mutate_exponent(base, rng)
            elif mut_type == "add_term":
                mutated = self._add_term(base, vars_available, rng)
            else:
                mutated = self._simplify_term(base, rng)
            mutation_candidates.append(mutated)

        # 4. Deliberately Wrong / Adversarial (~10%)
        n_wrong = context.n_candidates - len(template_candidates[:n_templates]) - len(refine_candidates[:n_refine]) - len(mutation_candidates[:n_mutations])
        n_wrong = max(n_wrong, int(context.n_candidates * 0.1))
        wrong_candidates = []
        for _ in range(n_wrong):
            v1 = rng.choice(vars_available)
            if len(vars_available) > 1:
                v2 = rng.choice([v for v in vars_available if v != v1])
                wrong_candidates.append(f"{v1} + {v2}")
            else:
                wrong_candidates.append(f"{v1} + 1/0")

        # Filter injected to be unique and cap it to prevent starving the templates/probabilistic pools
        seen_injected = set()
        unique_injected = []
        for cand in injected:
            try:
                sp.sympify(cand, locals=sym_locals)
                if cand not in seen_injected:
                    seen_injected.add(cand)
                    unique_injected.append(cand)
            except Exception:
                continue
        max_injected = int(context.n_candidates * 0.4)
        unique_injected = unique_injected[:max_injected]

        # Combine: injected first (highest priority), then probabilistic pools
        all_pool = (
            unique_injected +
            template_candidates[:n_templates] +
            refine_candidates[:n_refine] +
            mutation_candidates[:n_mutations] +
            wrong_candidates[:n_wrong]
        )

        seen = set()
        unique_candidates = []
        for cand in all_pool:
            try:
                sp.sympify(cand, locals=sym_locals)
                if cand not in seen:
                    seen.add(cand)
                    unique_candidates.append(cand)
            except Exception:
                continue

        # Pad with diverse unique templates
        attempts = 0
        while len(unique_candidates) < context.n_candidates and attempts < 300:
            attempts += 1
            v1 = rng.choice(vars_available)
            v2 = rng.choice(vars_available)
            idx = rng.randint(0, 100)
            exp_val = rng.choice([1, 2, 3])
            
            if len(vars_available) > 1 and rng.choice([True, False]):
                fallback = f"theta_{idx} * {v1} * {v2}**{exp_val}"
            else:
                fallback = f"theta_{idx} * {v1}**{exp_val}"
                
            try:
                sp.sympify(fallback, locals=sym_locals)
                if fallback not in seen:
                    seen.add(fallback)
                    unique_candidates.append(fallback)
            except Exception:
                continue

        logger.info(f"[MockProposer] Proposed {len(unique_candidates)} unique candidates "
                    f"({len(injected)} injected + probabilistic). Vars: {vars_available}")
        return unique_candidates[:context.n_candidates]


    def _mutate_operator(self, expr_str: str, rng: np.random.RandomState) -> str:
        if "*" in expr_str and rng.choice([True, False]):
            return expr_str.replace("*", "+", 1)
        elif "+" in expr_str:
            return expr_str.replace("+", "*", 1)
        elif "/" in expr_str:
            return expr_str.replace("/", "*", 1)
        return expr_str

    def _mutate_exponent(self, expr_str: str, rng: np.random.RandomState) -> str:
        if "**theta_1" in expr_str:
            return expr_str.replace("**theta_1", "**theta_2")
        elif "**2" in expr_str:
            return expr_str.replace("**2", "**theta_1")
        elif "theta_1" in expr_str:
            return expr_str.replace("theta_1", "theta_2")
        return expr_str + "**2"

    def _add_term(self, expr_str: str, variables: List[str], rng: np.random.RandomState) -> str:
        var = rng.choice(variables)
        n = rng.randint(2, 5)
        return f"{expr_str} + theta_{n} * {var}"

    def _simplify_term(self, expr_str: str, rng: np.random.RandomState) -> str:
        if "+" in expr_str:
            parts = expr_str.rsplit("+", 1)
            return parts[0].strip()
        return expr_str


class AnthropicProposer(BaseProposer):
    """
    Production stub for generating candidates via Anthropic Claude API.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def propose(self, context: ProposalContext) -> List[str]:
        if not self.api_key:
            raise NotImplementedError(
                "AnthropicProposer requires a valid Anthropic API key to function."
            )
        
        prompt = self.get_prompt_template(context)
        raise NotImplementedError(
            "API Call not implemented. The prompt that would be sent is:\n\n"
            f"{prompt}"
        )

def _build_rich_context_strings(context: ProposalContext) -> Tuple[str, str]:
    """Compute residual percentile statistics and asymptotic scaling hint for LLM prompt injection.

    Returns:
        percentile_str: Human-readable string of absolute residual percentiles (5th–95th).
                        Always includes the target variable's implicit units context.
        exponent_hint_str: A standalone bullet string with the leading scaling exponent hint,
                           or an empty string if no exponent data is available.
                           Format is consistent regardless of presence: either empty or a
                           fully-formed "- Key: value" bullet, NOT a newline-prefixed fragment.
    """
    percentile_str = "None"
    if context.residual_data is not None and len(context.residual_data) > 0:
        res_abs = np.abs(context.residual_data)
        p5  = np.percentile(res_abs, 5)
        p25 = np.percentile(res_abs, 25)
        p50 = np.percentile(res_abs, 50)
        p75 = np.percentile(res_abs, 75)
        p95 = np.percentile(res_abs, 95)
        # Include target name for unit context so LLM can interpret magnitude correctly.
        unit_ctx = f" [{context.target_name} units]" if context.target_name else ""
        percentile_str = (
            f"5th: {p5:.2e}, 25th: {p25:.2e}, 50th: {p50:.2e}, "
            f"75th: {p75:.2e}, 95th: {p95:.2e}{unit_ctx}"
        )

    # Exponent hint is formatted as a self-contained bullet so callers can
    # safely insert it anywhere in the prompt without structural side-effects.
    exponent_hint_str = ""
    if context.residual_features and hasattr(context.residual_features, 'leading_exponent'):
        le = context.residual_features.leading_exponent
        if le is not None:
            exponent_hint_str = (
                f"- Leading scaling exponent hint: ~{le:.2f} "
                f"(NAAE-estimated; propose corrections following this asymptotic power law)."
            )
    return percentile_str, exponent_hint_str


class GeminiProposer(BaseProposer):
    """
    Generates candidate equations using Google Gemini API (free tier available on Google AI Studio).
    Uses standard library urllib for zero-dependency execution.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    def _select_mode(self, iteration: int, stuck_count: int) -> Tuple[str, str]:
        MODE_INSTRUCTIONS = {
            "explore": "Try a STRUCTURALLY RADICAL equation. Change the mathematical family entirely (e.g. from polynomial to rational, exponential, or transcendental).",
            "exploit": "Perform MINIMAL modifications to the top-performing candidate's structure to improve numerical fit or correct dimensional errors.",
            "escape":  "The search is stuck in a local minimum. Propose a completely different mathematical structure (using fractional powers, rational, or implicit terms)."
        }
        if stuck_count > 4:
            return "escape", MODE_INSTRUCTIONS["escape"]
        if iteration % 3 == 0 and iteration > 0:
            return "explore", MODE_INSTRUCTIONS["explore"]
        return "exploit", MODE_INSTRUCTIONS["exploit"]

    def propose(self, context: ProposalContext) -> List[str]:
        if not self.api_key:
            raise ValueError("Gemini API key is required.")
        
        prompt = self.get_prompt_template(context)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_response(text, context.n_candidates)
            
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            logger.error(f"Gemini API Error: {e.code} - {err_msg}")
            raise RuntimeError(f"Gemini API Call failed with code {e.code}: {err_msg}")
        except Exception as e:
            logger.error(f"Failed to connect to Gemini API: {e}")
            raise e

    def get_prompt_template(self, context: ProposalContext) -> str:
        mode, mode_desc = self._select_mode(context.iteration, context.stuck_count)
        percentile_str, exponent_hint_str = _build_rich_context_strings(context)
        
        hints_str = "\n".join([f"- {h}" for h in context.structural_hints]) if context.structural_hints else "None"
        prev_best_str = ""
        if context.previous_best:
            prev_best_str = "\n".join([f"- {expr} (NMSE: {nmse:.4e})" for expr, nmse in context.previous_best[:5]])
        else:
            prev_best_str = "None"
            
        limits_str = ""
        if context.known_limits:
            limits_str = "\n".join([f"- Limit of {lim.get('variable', lim.get('var'))} approaching {lim.get('limit')} must yield {lim.get('expected')}" for lim in context.known_limits])
        else:
            limits_str = "None"
            
        units_str = ", ".join([f"'{k}' ({v})" for k, v in context.variables_with_units.items()]) if context.variables_with_units else "None"

        return f"""You are a physics equation discovery system operating in [{mode.upper()}] mode.
Your task: Propose exactly {context.n_candidates} symbolic candidate equations that model the target physical quantity '{context.target_name}' to explain observed anomalies.

SEARCH MODE INSTRUCTION ({mode.upper()}):
{mode_desc}

PHYSICAL CONTEXT:
- Domain: {context.domain}
- Classical law: {context.classical_expr}
- Observed anomaly: {context.anomaly_description}
- Variables with units: {units_str}
- Absolute residual magnitude percentiles: {percentile_str}
{('- ' + exponent_hint_str) if exponent_hint_str else ''}- Known limits that MUST be satisfied:
{limits_str}

HARD CONSTRAINTS (violation = immediate rejection):
1. Expression must be dimensionally homogeneous.
2. Must reduce to the classical law in the asymptotic limit: {context.classical_limit_condition}
3. Use 'theta_0', 'theta_1', 'theta_2', ... for free numerical parameters (dimensionless coefficients only).
4. Maximum complexity: {context.max_nodes} AST nodes.

PREVIOUS CANDIDATES AND THEIR SCORES (FEEDBACK LOOP):
{prev_best_str}
(Lower NMSE = better fit. Learn from what failed.)

GENERATE EXACTLY {context.n_candidates} CANDIDATE EQUATIONS.
- Do NOT just modify coefficients of previous candidates.
- Propose structurally diverse mathematical terms.
- Consider structural hints: {hints_str}
- Output ONLY the raw equation strings in SymPy-parseable format, one per line.
- Do NOT include any markdown formatting, backticks, or explanation.

Candidates:"""

    def _parse_response(self, text: str, n_candidates: int) -> List[str]:
        candidates = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove bullets
            if line.startswith(("-", "*", "•")):
                line = line[1:].strip()
            elif "." in line[:4]:
                parts = line.split(".", 1)
                if parts[0].strip().isdigit():
                    line = parts[1].strip()
            line = line.replace("`", "").strip()
            if not line:
                continue
                
            # If the LLM outputted "y = expression" or similar, keep only the right-hand side
            if "=" in line:
                line = line.split("=", 1)[1].strip()
                
            # Standardize exponent symbol
            line = line.replace("^", "**")
            
            try:
                sp.sympify(line)
                candidates.append(line)
            except Exception:
                continue
        return candidates[:n_candidates]


class OpenAICompatibleProposer(BaseProposer):
    """
    Generates candidate equations using any OpenAI-compatible API (DeepSeek, Groq, OpenRouter, etc.).
    Uses standard library urllib for zero-dependency execution.
    """
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def _select_mode(self, iteration: int, stuck_count: int) -> Tuple[str, str]:
        # Same mode selection as Gemini
        g = GeminiProposer("")
        return g._select_mode(iteration, stuck_count)

    def propose(self, context: ProposalContext) -> List[str]:
        if not self.api_key:
            raise ValueError("API key is required.")
        
        prompt = self.get_prompt_template(context)
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a physics symbolic regression assistant that outputs raw equations one per line without markdown formatting or explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
            text = res_data["choices"][0]["message"]["content"]
            return self._parse_response(text, context.n_candidates)
            
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            logger.error(f"OpenAI-Compatible API Error: {e.code} - {err_msg}")
            raise RuntimeError(f"API Call failed with code {e.code}: {err_msg}")
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            raise e

    def get_prompt_template(self, context: ProposalContext) -> str:
        # Reuses the exact same prompt structure as Gemini
        g = GeminiProposer("")
        return g.get_prompt_template(context)

    def _parse_response(self, text: str, n_candidates: int) -> List[str]:
        g = GeminiProposer("")
        return g._parse_response(text, n_candidates)


class CorrectionMockProposer(BaseProposer):
    """Proposes dimensionless correction terms Δ(x; θ) for physical anomalies."""
    def __init__(self, seed: int = 42, extended: bool = False):
        self.seed = seed
        self.extended = extended
        self._templates = [
            # Power-law family
            "theta_0 * ({v1} / theta_1)**2",
            "theta_0 * ({v1} / theta_1)**theta_2",
            "theta_0 * ({v1} / theta_1)**2 + theta_2 * ({v1} / theta_1)**4",
            "theta_0 * ({v1} / {c1})**2",
            "theta_0 * ({v1} / {c1})**theta_1",
            "theta_0 * ({v1} / {c1})**2 + theta_1 * ({v1} / {c1})**4",
            "theta_0 * ({v1} / theta_1)**(-theta_2)",
            "theta_0 * ({v1} / {c1})**(-theta_1)",
            "theta_0 * ({v1} / theta_1)**(-4)",
            "theta_0 * ({v1} / {c1})**(-4)",
            "-(theta_0 / {v1})**4",
            "-(theta_0 / {v1})**theta_1",
            "theta_0 * (theta_1 / {v1})**4 - 1.0",
            "-(theta_0 / {v1})**theta_1 + theta_2",
            
            # Exponential family
            "theta_0 * exp(-{v1} / theta_1)",
            "theta_0 * exp(-theta_1 / {v1})",
            "theta_0 * (1.0 - exp(-{v1} / theta_1))",
            "exp(-{v1} / theta_1) - 1.0",
            "exp(-{v1} / {c1}) - 1.0",
            "theta_0 * exp(-{v1} / {c1})",
            
            # Rational family
            "theta_0 * {v1} / ({v1} + theta_1)",
            "theta_0 / (1.0 + theta_1 * {v1}**2)",
            "theta_0 * {v1}**2 / ({v1}**2 + theta_1**2)",
            
            # Trigonometric family
            "theta_0 * sin({v1} / theta_1)",
            "-theta_0 * tanh(theta_1 / {v1})**2",
            "theta_0 * tanh(theta_1 / {v1})**2",
            "theta_0 * sin({v1} / theta_1) / ({v1} / theta_1)",
            "sin({v1} / theta_1) / ({v1} / theta_1) - 1.0",
            
            # Logarithmic family
            "theta_0 * log(1.0 + {v1} / theta_1)",
            "log(1.0 + {v1} / theta_1) / ({v1} / theta_1) - 1.0",
            "theta_0 * log(1.0 + {v1} / theta_1) / ({v1} / theta_1)",
            
            # Additive/dimensioned templates
            "theta_0 * {v1}**4",
            "theta_0 * {v1}**2",
            "theta_0 * {v1}**3"
        ]
        
        self.families = {
            "power_law": [
                "theta_0 * ({v1} / theta_1)**theta_2",
                "theta_0 * ({v1} / {c1})**theta_1",
                "theta_0 * ({v1} / theta_1)**(-theta_2)",
                "theta_0 * ({v1} / {c1})**(-theta_1)",
                "theta_0 * ({v1} / theta_1)**(-4)",
                "theta_0 * ({v1} / {c1})**(-4)",
                "-(theta_0 / {v1})**4",
                "-(theta_0 / {v1})**theta_1",
                "theta_0 * (theta_1 / {v1})**4 - 1.0",
                "-(theta_0 / {v1})**theta_1 + theta_2"
            ],
            "polynomial": [
                "theta_0 * ({v1} / theta_1)**2",
                "theta_0 * ({v1} / theta_1)**2 + theta_2 * ({v1} / theta_1)**4",
                "theta_0 * ({v1} / {c1})**2",
                "theta_0 * ({v1} / {c1})**2 + theta_1 * ({v1} / {c1})**4",
                "theta_0 * {v1}**4",
                "theta_0 * {v1}**2",
                "theta_0 * {v1}**3"
            ],
            "exponential": [
                "theta_0 * exp(-{v1} / theta_1)",
                "theta_0 * exp(-theta_1 / {v1})",
                "theta_0 * (1.0 - exp(-{v1} / theta_1))",
                "exp(-{v1} / theta_1) - 1.0",
                "exp(-{v1} / {c1}) - 1.0",
                "theta_0 * exp(-{v1} / {c1})"
            ],
            "rational": [
                "theta_0 * {v1} / ({v1} + theta_1)",
                "theta_0 / (1.0 + theta_1 * {v1}**2)",
                "theta_0 * {v1}**2 / ({v1}**2 + theta_1**2)"
            ],
            "trigonometric": [
                "theta_0 * sin({v1} / theta_1)",
                "-theta_0 * tanh(theta_1 / {v1})**2",
                "theta_0 * tanh(theta_1 / {v1})**2",
                "theta_0 * sin({v1} / theta_1) / ({v1} / theta_1)",
                "sin({v1} / theta_1) / ({v1} / theta_1) - 1.0"
            ],
            "logarithmic": [
                "theta_0 * log(1.0 + {v1} / theta_1)",
                "log(1.0 + {v1} / theta_1) / ({v1} / theta_1) - 1.0",
                "theta_0 * log(1.0 + {v1} / theta_1) / ({v1} / theta_1)"
            ]
        }
        
        if extended:
            self._extend_templates()

    def _extend_templates(self):
        # Extend standard families with multi-variable physics templates
        self.families["power_law"].extend([
            "theta_0 * {v1}**theta_1 * {v2}**theta_2",
            "theta_0 * ({v1} / {v2})**theta_1",
            "theta_0 * ({v1} * {v2})**theta_1",
            "theta_0 * ({v1} / ({v2} * theta_1))**theta_2",
            "theta_0 * ({v1} * {v2} / theta_1)**theta_2",
            "theta_0 * ({v1} / (theta_1 + {v2}))**theta_2",
            "theta_0 * sqrt({v1} * {v2}) / theta_1",
            "theta_0 * ({v1}**theta_1) * ({v2}**(-theta_2))",
        ])
        
        self.families["polynomial"].extend([
            "theta_0 * {v1} * {v2}",
            "theta_0 * {v1}**2 * {v2}",
            "theta_0 * {v1} * {v2}**2",
            "theta_0 * {v1}**2 * {v2}**2",
            "theta_0 * ({v1} / theta_1)**2 + theta_2 * ({v2} / theta_3)**2",
            "theta_0 * ({v1} / theta_1) * ({v2} / theta_2)",
            "theta_0 * {v1}**3 * {v2}**(-1)",
            "theta_0 * {v1}**2 + theta_1 * {v2}**2 + theta_2 * {v1} * {v2}",
        ])
        
        self.families["exponential"].extend([
            "theta_0 * exp(-{v1} * {v2} / theta_1)",
            "theta_0 * exp(-{v1} / ({v2} * theta_1))",
            "theta_0 * exp(-({v1} / theta_1)**2 - ({v2} / theta_2)**2)",
            "theta_0 * exp(-sqrt({v1}**2 + {v2}**2) / theta_1)",
            "theta_0 * (1.0 - exp(-{v1} * {v2} / theta_1))",
            "theta_0 * (1.0 - exp(-{v1} / ({v2} * theta_1)))",
            "exp(-{v1} * {v2} / theta_1) - 1.0",
            "exp(-{v1} / ({v2} * theta_1)) - 1.0",
        ])
        
        self.families["rational"].extend([
            "theta_0 * {v1} / ({v1} + {v2})",
            "theta_0 * {v1} * {v2} / ({v1} + {v2} + theta_1)",
            "theta_0 * {v1}**2 / ({v1}**2 + {v2}**2 + theta_1**2)",
            "theta_0 * {v1} * {v2} / ({v1}**2 + {v2}**2 + theta_1**2)",
            "theta_0 / (1.0 + theta_1 * {v1}**2 + theta_2 * {v2}**2)",
            "theta_0 * {v1} / ({v2} + theta_1)",
            "theta_0 * ({v1} - {v2}) / ({v1} + {v2} + theta_1)",
            "theta_0 * {v1}**2 / ({v2} + theta_1)**2",
        ])
        
        self.families["trigonometric"].extend([
            "theta_0 * sin({v1} / {v2})",
            "theta_0 * sin({v1} * {v2} / theta_1)",
            "theta_0 * cos({v1} / {v2})",
            "theta_0 * cos({v1} * {v2} / theta_1)",
            "theta_0 * tanh({v1} / {v2})**2",
            "theta_0 * tanh({v1} * {v2} / theta_1)**2",
            "theta_0 * sin({v1} / theta_1) * cos({v2} / theta_2)",
            "theta_0 * sin({v1} / theta_1) * sin({v2} / theta_2)",
            "theta_0 * cos({v1} / theta_1) * cos({v2} / theta_2)",
        ])
        
        self.families["logarithmic"].extend([
            "theta_0 * log(1.0 + {v1} / {v2})",
            "theta_0 * log(1.0 + {v1} * {v2} / theta_1)",
            "theta_0 * log(theta_1 + {v1} / {v2})",
            "theta_0 * log(1.0 + ({v1} / theta_1)**2 + ({v2} / theta_2)**2)",
            "log(1.0 + {v1} / {v2}) / ({v1} / {v2}) - 1.0",
            "theta_0 * log(1.0 + {v1} / {v2}) / ({v1} / {v2})",
        ])
        
        # Synchronize list of all templates
        all_templates = []
        for templates in self.families.values():
            all_templates.extend(templates)
        self._templates = list(set(all_templates))


    def _inject_physics_constants(self, phys_vars: List[str], non_theta_constants: List[str]) -> List[str]:
        """Deterministic templates from physical constants (always prepended, never replaces sampling)."""
        injected: List[str] = []
        if not non_theta_constants or not phys_vars:
            return injected
        for const in non_theta_constants:
            injected.append(f"theta_0 * {const}")
            for v1 in phys_vars:
                injected.append(f"theta_0 * {const} * {v1}")
                injected.append(f"theta_0 * {const} / {v1}**2")
                injected.append(f"theta_0 * {const} / {v1}")
                injected.append(f"theta_0 * {v1} / {const}")
                if v1 in ("vc2", "beta_rel"):
                    injected.append(f"theta_0 * {v1}")
                for v2 in phys_vars:
                    if v2 == v1:
                        continue
                    injected.append(f"theta_0 * {const} * {v1} * {v2}")
                    for v3 in phys_vars:
                        if v3 == v1 or v3 == v2:
                            continue
                        injected.append(f"theta_0 * {const} * {v1} * {v2} / {v3}**2")
                        injected.append(f"theta_0 * {const} * {v1} * {v2} / {v3}")
                    injected.append(f"theta_0 * {const} * {v1} * {v2}**4")
                    injected.append(f"theta_0 * {const} * {v2} * {v1}**4")
                    injected.append(f"theta_0 * {v1} * (1 + theta_1 * {v2} / {const})")
        if len(phys_vars) >= 2:
            for i, v1 in enumerate(phys_vars):
                for v2 in phys_vars[i + 1:]:
                    injected.append(f"theta_0 * {v1} * {v2}")
                    injected.append(f"theta_0 * {v1}**2 * sin(theta_1 * {v2})")
            if len(phys_vars) >= 3:
                for i, v1 in enumerate(phys_vars):
                    for j, v2 in enumerate(phys_vars[i + 1:], i + 1):
                        for v3 in phys_vars[j + 1:]:
                            injected.append(f"theta_0 * {v1} * {v2} * {v3}")
        return injected

    def propose(self, context: ProposalContext) -> List[str]:
        rng = np.random.RandomState(self.seed + context.iteration)
        candidates = []
        vars_available = list(context.variable_names)
        constants_available = [c for c in (context.constants or {}) if not c.startswith("theta_")]

        sym_locals = {}
        for var in vars_available:
            sym_locals[var] = sp.Symbol(var)
        for c in constants_available:
            sym_locals[c] = sp.Symbol(c)
        for i in range(100):
            sym_locals[f"theta_{i}"] = sp.Symbol(f"theta_{i}")

        # Extended mode: targeted templates for real-data parametrizations only (not standard benchmark)
        if self.extended:
            if "vc2" in vars_available:
                candidates.append("theta_0 * vc2")
            if "P" in vars_available:
                # Target generic power law only
                candidates.append("theta_0 * P**(-theta_1)")

        # Phase 1: Determine prior weights for template families based on residual features
        weights = {
            "power_law": 1.0,
            "polynomial": 1.0,
            "exponential": 1.0,
            "rational": 1.0,
            "trigonometric": 1.0,
            "logarithmic": 1.0
        }
        
        if context.residual_features:
            rf = context.residual_features
            # 1. Monotonic decay points strongly to exponential or rational decay
            if rf.decay_rate > 0.4:
                weights["exponential"] += 8.0
                weights["rational"] += 4.0
            
            # 2. Curvature and high monotonicity without exponential decay point to logarithmic/power law
            if abs(rf.monotonicity) > 0.7:
                if rf.decay_rate < 0.2:
                    weights["logarithmic"] += 6.0
                    weights["power_law"] += 4.0
                    weights["rational"] += 2.0
            
            # 3. High oscillation score points to trigonometric (or tanh squared)
            if rf.oscillation_score > 0.15:
                weights["trigonometric"] += 10.0
            
            # 4. Low oscillation and strong curvature sign/symmetry points to polynomial
            if rf.oscillation_score < 0.05:
                if hasattr(rf, 'symmetry') and abs(rf.symmetry) > 0.1: # Even/Odd symmetry detected
                    if rf.symmetry > 0: # Even dominated
                        weights["polynomial"] += 8.0
                    else: # Odd dominated
                        weights["polynomial"] += 4.0
                else:
                    weights["polynomial"] += 4.0

        # Construct a weighted template list based on family weights
        weighted_templates = []
        for fam, templates in self.families.items():
            w = weights[fam]
            # Each template in this family gets weight w / len(templates)
            for t in templates:
                weighted_templates.append((t, w / len(templates)))
                
        temps, temp_weights = zip(*weighted_templates)
        temp_weights = np.array(temp_weights)
        temp_weights /= np.sum(temp_weights)
        
        # 50/50 mix of uniform and weighted probabilities to ensure structural diversity and prevent starvation of families
        uniform_weights = np.ones_like(temp_weights) / len(temp_weights)
        mix_weights = 0.5 * uniform_weights + 0.5 * temp_weights
        mix_weights /= np.sum(mix_weights)
        
        # Stochastic template bank — full 2× oversample (restored; B4 must not replace this)
        n_templates_to_process = len(self._templates) * 2
        chosen_templates = rng.choice(temps, size=n_templates_to_process, p=mix_weights, replace=True)

        for template in chosen_templates:
            needed_vars = 1
            if "{v2}" in template:
                needed_vars = 2
            needed_consts = 0
            if "{c1}" in template:
                needed_consts = 1

            if len(vars_available) >= needed_vars:
                v_chosen = rng.choice(vars_available, size=needed_vars, replace=False)
                fmt_dict = {f"v{i+1}": v_chosen[i] for i in range(needed_vars)}

                if needed_consts > 0:
                    if len(constants_available) >= needed_consts:
                        c_chosen = rng.choice(constants_available, size=needed_consts, replace=False)
                        fmt_dict["c1"] = c_chosen[0]
                    else:
                        fmt_dict["c1"] = "theta_9"

                try:
                    candidates.append(template.format(**fmt_dict))
                except Exception:
                    pass

        # Refinement of Previous Best Corrections
        if context.previous_best:
            sorted_prev = sorted(context.previous_best, key=lambda x: x[1])
            n_refine = int(context.n_candidates * 0.3)
            for _ in range(n_refine):
                base_expr, _ = sorted_prev[rng.choice(len(sorted_prev))]
                if "theta_0" in base_expr and rng.choice([True, False]):
                    mut = base_expr.replace("theta_0", "theta_1", 1)
                else:
                    mut = f"({base_expr})**2" if "**2" not in base_expr else base_expr
                candidates.append(mut)

        # 4. Clean, validate, and select unique candidates
        seen = set()
        unique_candidates = []
        for cand in candidates:
            try:
                sp.sympify(cand, locals=sym_locals)
                if cand not in seen:
                    seen.add(cand)
                    unique_candidates.append(cand)
            except Exception:
                pass
                
        # Pad with basic fallback terms if necessary
        attempts = 0
        while len(unique_candidates) < context.n_candidates and attempts < 100:
            attempts += 1
            v = rng.choice(vars_available)
            fallback = f"theta_0 * {v}"
            if fallback not in seen:
                seen.add(fallback)
                unique_candidates.append(fallback)

        return unique_candidates[:context.n_candidates]



class CorrectionGeminiProposer(BaseProposer):
    """Generates candidate physical correction terms Δ using Google Gemini API."""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    def _select_mode(self, iteration: int, stuck_count: int) -> Tuple[str, str]:
        g = GeminiProposer("")
        return g._select_mode(iteration, stuck_count)

    def propose(self, context: ProposalContext) -> List[str]:
        if not self.api_key:
            raise ValueError("Gemini API key is required.")
        
        prompt = self.get_prompt_template(context)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            g = GeminiProposer("")
            return g._parse_response(text, context.n_candidates)
            
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            logger.error(f"Gemini API Error: {e.code} - {err_msg}")
            raise RuntimeError(f"Gemini API Call failed with code {e.code}: {err_msg}")
        except Exception as e:
            logger.error(f"Failed to connect to Gemini API: {e}")
            raise e

    def get_prompt_template(self, context: ProposalContext) -> str:
        mode, mode_desc = self._select_mode(context.iteration, context.stuck_count)
        percentile_str, exponent_hint_str = _build_rich_context_strings(context)
        
        units_str = ", ".join([f"'{k}' ({v})" for k, v in context.variables_with_units.items()]) if context.variables_with_units else "None"

        return f"""You are a theoretical physicist discovering mathematical corrections to classical laws.
Search Mode: {mode.upper()} ({mode_desc})

KNOWN CLASSICAL LAW: {context.classical_expr}
Observed Anomaly: {context.anomaly_description}
Variables with Units: {units_str}
Absolute residual magnitude percentiles: {percentile_str}
{exponent_hint_str + chr(10) if exponent_hint_str else ''}

YOUR TASK: Propose exactly {context.n_candidates} dimensionless correction terms Δ such that:
    y_true ≈ y_classical × (1 + Δ)     [multiplicative correction]
    OR
    y_true ≈ y_classical + Δ           [additive correction]

HARD CONSTRAINTS (violation = immediate rejection):
1. Δ MUST be dimensionless when using ratios like (v/c), (r/λ), (x/x₀)
2. Reduced Limit check: As {context.classical_limit_condition or 'classical limit'} is reached, Δ must approach 0.
3. Use 'theta_0', 'theta_1', 'theta_2', ... for free parameter symbols (which fit constants).
4. Maximum complexity: {context.max_nodes} AST nodes.

         Output ONLY raw SymPy-parseable correction expression strings, one per line. No markdown, no comments, no explanation."""


class HybridCorrectionProposer(BaseProposer):
    """Combines Mock templates, Grammar-based candidates, and LLM-generated candidates."""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", seed: int = 42):
        from adcd.grammar_proposer import GrammarProposer
        self.mock = CorrectionMockProposer(seed=seed)
        self.grammar = GrammarProposer(seed=seed)
        self.gemini = CorrectionGeminiProposer(api_key=api_key, model_name=model_name)
        self.sources = {}

    def propose(self, context: ProposalContext) -> List[str]:
        # Reset sources mapping for this iteration
        self.sources = {}
        
        # Divide candidates between Gemini, Grammar, and Mock
        n_gemini = int(context.n_candidates * 0.4)
        n_grammar = int(context.n_candidates * 0.4)
        n_mock = max(1, context.n_candidates - n_gemini - n_grammar)
        
        # Ensure we have at least 1 for each if candidates is large enough
        if context.n_candidates >= 3:
            n_gemini = max(1, n_gemini)
            n_grammar = max(1, n_grammar)
        
        # Propose mock candidates
        mock_context = replace(context, n_candidates=n_mock)
        mock_candidates = self.mock.propose(mock_context)
        
        # Propose grammar candidates
        grammar_context = replace(context, n_candidates=n_grammar)
        grammar_candidates = self.grammar.propose(grammar_context)
        
        # Propose gemini candidates
        gemini_candidates = []
        try:
            gemini_context = replace(context, n_candidates=n_gemini)
            gemini_candidates = self.gemini.propose(gemini_context)
        except Exception as e:
            logger.warning(f"Gemini proposer failed: {e}. Falling back to more grammar and mock candidates.")
            # If Gemini fails, we shift its allocation to grammar and mock
            n_extra = n_gemini
            n_extra_grammar = n_extra // 2
            n_extra_mock = n_extra - n_extra_grammar
            
            grammar_context = replace(context, n_candidates=n_grammar + n_extra_grammar)
            grammar_candidates = self.grammar.propose(grammar_context)
            
            mock_context = replace(context, n_candidates=n_mock + n_extra_mock)
            mock_candidates = self.mock.propose(mock_context)

        # Track sources
        for c in gemini_candidates:
            self.sources[c] = "gemini"
        for c in grammar_candidates:
            if c not in self.sources:
                self.sources[c] = "grammar"
        for c in mock_candidates:
            if c not in self.sources:
                self.sources[c] = "mock"

        # Merge and deduplicate, keeping order: Gemini first, then Grammar, then Mock
        seen = set()
        unique = []
        for c in gemini_candidates + grammar_candidates + mock_candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:context.n_candidates]


