# ADCD Clean Development Plan
## Post-Rollback to 6c8c887 (CI 3/3 Stable)

---

## KONTEKS PROYEK

**Framework:** ADCD (Anomaly-Driven Correction Discovery)  
**Current stable commit:** `6c8c887` (Fix linting errors, CI 3/3 PASS)  
**Goal:** Add Phase 1 hardening + Phase 3 Bayesian output cleanly, tanpa bug Phase 2

**Pipeline saat ini (6c8c887):**
```
Proposer (Mock/Grammar) → Physics Gates (AST/Dim/ARC) → JAX L-BFGS-B → BIC Reranking
```

**Yang ADA di 6c8c887:**
- GrammarProposer dengan Buckingham-Pi ratios
- 9 blind scenarios (Blind-1 s/d 9, single-variable)
- Original 9 benchmark scenarios (paper 82.8%)
- CI 3/3 passing

**Yang BELUM ADA (akan ditambahkan):**
- ARC numerical pre-filter
- Correlation-based ratio ranking
- Bivariate templates (extended mode)
- Benchmark 3-column comparison
- Bayesian posterior output
- Identifiability analysis

---

## ATURAN WAJIB SEBELUM MULAI

```
1. Baca setiap file yang akan diubah SEPENUHNYA sebelum menulis kode
2. pytest tests/ -v HARUS PASS sebelum setiap commit
3. Setiap task = SATU commit, maksimal 2 file yang diubah
4. Jangan pernah ubah: physics gates (AST/Dim/ARC), BIC formula, JAXOptimizer core
5. Jangan campurkan task dari Phase yang berbeda dalam 1 commit
6. Jika CI gagal setelah commit: STOP, debug dulu, jangan lanjut
```

---

## PHASE 1 HARDENING

### Task P1-1: Label Fix (PALING MUDAH, MULAI DARI SINI)
**File:** `src/adcd/anomaly_scenarios.py`  
**Perubahan:** 2 baris saja

```python
# Cari dan ubah:
# Blind-1 Van der Waals
correction_class="rational"   →   correction_class="power_law"

# Blind-12 Thermal Radiation 2D  
correction_class="rational"   →   correction_class="power_law"
```

**Alasan:** `classify_structure()` sudah benar secara matematis.
`n²/V²` adalah power law (monomial dengan pangkat negatif), BUKAN rational function.
Rational function = P(x)/Q(x) dimana Q adalah Add seperti `x/(x+a)`.
Net Radiation `-(θ₀/T)⁴` juga power_law dan classifier sudah benar untuk itu.
Mengubah classifier justru akan merusak Net Radiation di original 9 scenarios.

**Verification:**
```python
# Setelah fix, pastikan:
assert classify_structure(sp.sympify("theta_0 * n**2 / V**2")) == "power_law"
assert classify_structure(sp.sympify("theta_0 * exp(-r/theta_1)")) == "exponential"  # tidak berubah
assert classify_structure(sp.sympify("-(theta_0/T)**4")) == "power_law"  # tidak berubah
```

**Success criteria:**
- [ ] pytest tests/ pass
- [ ] Blind-1 dan Blind-12 correction_class = "power_law"
- [ ] Net Radiation dan Casimir masih "power_law" (tidak berubah)

---

### Task P1-2: ARC Numerical Pre-filter di GrammarProposer
**File:** `src/adcd/grammar_proposer.py`

**Tambahkan method baru** `_numerical_arc_prefilter()` ke class GrammarProposer.

```python
def _numerical_arc_prefilter(
    self,
    candidates: List[str],
    context: ProposalContext,
    tol: float = 0.05,
    n_samples: int = 10,
) -> List[str]:
    """
    Numerical ARC pre-filter: reject candidates that don't vanish at classical limit.
    
    SymPy's symbolic limit fails on parameterized expressions like exp(-r/theta_0)
    because it can't determine the sign of theta_0. This numerical pre-filter
    evaluates the expression near the classical limit and rejects those that
    don't vanish, using random theta samples to handle parameterization.
    
    Replaces the need for 4x oversampling budget.
    """
    import numpy as np
    
    if not context.classical_limit_variable or not context.classical_limit_direction:
        return candidates
    
    limit_var = context.classical_limit_variable
    limit_dir = context.classical_limit_direction
    
    # Handle both string and list formats
    if isinstance(limit_var, str):
        limit_vars = [limit_var]
        limit_dirs = [limit_dir]
    else:
        limit_vars = limit_var
        limit_dirs = limit_dir
    
    rng = np.random.default_rng(self.seed)
    passed = []
    
    for cand_str in candidates:
        try:
            expr = sp.sympify(cand_str)
            phys_syms = [s for s in expr.free_symbols 
                         if str(s) in context.variable_names]
            theta_syms = [s for s in expr.free_symbols 
                          if str(s).startswith("theta")]
            
            vanishes = True
            for limit_v, limit_d in zip(limit_vars, limit_dirs):
                limit_sym = sp.Symbol(limit_v)
                # Set limit value
                if limit_d == "0":
                    limit_val = 1e-6
                elif limit_d in ("oo", "inf", "+oo"):
                    limit_val = 1e8
                else:
                    try:
                        limit_val = float(limit_d) * 0.01
                    except:
                        continue
                
                # Test with n_samples random theta values
                any_vanish = False
                for _ in range(n_samples):
                    subs = {}
                    for s in phys_syms:
                        if str(s) == limit_v:
                            subs[s] = limit_val
                        else:
                            # Use midpoint of variable range
                            subs[s] = 1.0
                    for s in theta_syms:
                        subs[s] = rng.uniform(0.1, 10.0)
                    
                    try:
                        val = float(complex(expr.subs(subs)).real)
                        if np.isfinite(val) and abs(val) < tol:
                            any_vanish = True
                            break
                    except Exception:
                        continue
                
                if not any_vanish:
                    vanishes = False
                    break
            
            if vanishes:
                passed.append(cand_str)
                
        except Exception:
            continue
    
    logger.info(
        f"[GrammarProposer] ARC pre-filter: "
        f"rejected {len(candidates)-len(passed)}/{len(candidates)} candidates"
    )
    return passed
```

**Panggil di akhir `propose()` method, sebelum `return`:**
```python
# Setelah selected_candidates dipilih:
selected_candidates = self._numerical_arc_prefilter(selected_candidates, context)
logger.info(f"[GrammarProposer] After ARC pre-filter: {len(selected_candidates)} candidates")
return selected_candidates[:n_budget]
```

**PENTING:** Tambahkan `n_oversample = n_budget * 4` sebelum selection loop
karena ~75% kandidat akan direject oleh ARC pre-filter:
```python
n_oversample = n_budget * 4  # oversample untuk kompensasi ARC rejection rate
```

**Success criteria:**
- [ ] pytest tests/ pass
- [ ] Grammar proposer log menunjukkan "ARC pre-filter: rejected X/Y candidates"
- [ ] Kandidat yang survive ARC pre-filter lebih relevan (test manual dengan 1 scenario)

---

### Task P1-3: Correlation-Based Ratio Ranking
**File:** `src/adcd/grammar_proposer.py`

**Tambahkan dua optional field ke `ProposalContext`** (dataclass di `llm_proposer.py`):
```python
@dataclass
class ProposalContext:
    # ... existing fields ...
    X_data: Optional[Dict[str, np.ndarray]] = None      # tambahkan ini
    residual_data: Optional[np.ndarray] = None           # tambahkan ini
```

**Di `GrammarProposer.propose()`, setelah `unique_ratios` di-generate:**
```python
# Correlation-based ratio ranking
# Rank ratios by absolute Pearson correlation with residual
if context.X_data is not None and context.residual_data is not None:
    scored_ratios = []
    for r_expr in unique_ratios:
        try:
            phys_syms = [s for s in r_expr.free_symbols 
                         if str(s) in context.variable_names]
            if not phys_syms:
                scored_ratios.append((r_expr, 0.0))
                continue
            # Substitute theta with 1.0 for evaluation
            subs = {s: 1.0 for s in r_expr.free_symbols 
                    if str(s).startswith("theta")}
            r_sub = r_expr.subs(subs)
            fn = sp.lambdify(phys_syms, r_sub, modules=["numpy"])
            args = [context.X_data[str(s)] for s in phys_syms]
            r_vals = np.asarray(fn(*args), dtype=float)
            if not np.all(np.isfinite(r_vals)):
                scored_ratios.append((r_expr, 0.0))
                continue
            from scipy.stats import pearsonr
            corr, _ = pearsonr(r_vals, context.residual_data)
            scored_ratios.append((r_expr, abs(corr) if np.isfinite(corr) else 0.0))
        except Exception:
            scored_ratios.append((r_expr, 0.0))
    
    scored_ratios.sort(key=lambda x: x[1], reverse=True)
    unique_ratios = [r for r, _ in scored_ratios[:10]]  # top-10 only
    logger.info(
        f"[GrammarProposer] Correlation-ranked {len(scored_ratios)} ratios, "
        f"top-10 selected. Best |r|="
        f"{scored_ratios[0][1]:.3f}" if scored_ratios else "No ratios"
    )
```

**Di `correction_orchestrator.py`, saat membuat ProposalContext, tambahkan:**
```python
context = ProposalContext(
    # ... existing fields ...
    X_data=X,                    # tambahkan
    residual_data=residual,      # tambahkan
)
```

**Success criteria:**
- [ ] pytest tests/ pass
- [ ] Log menunjukkan "Correlation-ranked X ratios, top-10 selected. Best |r|=Y"
- [ ] Best |r| > 0.3 untuk scenarios dengan clear signal

---

### Task P1-4: Extended Bivariate Templates
**File:** `src/adcd/llm_proposer.py`

**Di `CorrectionMockProposer`, tambahkan method `_extend_templates()`** yang dipanggil
saat `extended=True`. Tambahkan template berikut:

```python
def _extend_templates(self, variable_names: List[str]) -> List[str]:
    """
    Extended bivariate templates for multi-variable corrections.
    All templates are ARC-safe by construction (vanish at classical limit).
    Only called when extended=True.
    """
    if len(variable_names) < 2:
        return []
    
    v1, v2 = variable_names[0], variable_names[1]
    templates = [
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
    return templates
```

**PENTING:** Sebelum menambahkan template ini ke pool, verifikasi setiap template
terhadap ARC gate untuk minimal satu scenario. Jangan add template yang fail ARC.

**Success criteria:**
- [ ] pytest tests/ pass
- [ ] `CorrectionMockProposer(extended=True).propose(context)` menghasilkan
      minimal 3 bivariate candidates untuk scenario dengan 2+ variabel
- [ ] Semua bivariate templates lulus ARC gate untuk Blind-10 Yukawa-2D

---

### Task P1-5: Benchmark Script 3-Column (GENIUS ADDITION)
**File:** `run_grammar_blind_benchmark.py` (rewrite dari scratch)

Script ini harus menghasilkan **tiga kolom perbandingan yang fair** dan
**diagnostic output per scenario** untuk paper reporting.

```python
"""
ADCD Blind Generalization Benchmark — 3-Column Comparison
Tests: Mock (base) | Mock (extended) | Grammar (standalone)
on all blind scenarios across 4 noise levels.
"""

import json
import time
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.llm_proposer import CorrectionMockProposer
from adcd.grammar_proposer import GrammarProposer
from adcd.dimensional_checker import DimensionalChecker
# ... imports

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
SEED = 42

def run_single(scenario, proposer, noise, seed):
    """Run discovery for one scenario+proposer+noise combination."""
    # ... orchestrator setup
    result = orchestrator.search_correction(scenario, noise_level=noise)
    return {
        "scenario": scenario.name,
        "proposer": proposer_name,
        "noise": noise,
        "discovered_expr": result.best_expr,
        "nmse_residual": result.best_nmse_residual,
        "class_match": result.evaluation.class_match,
        "success": result.evaluation.class_match and result.best_nmse_residual < 0.1,
        "n_proposed": sum(h.n_proposed for h in result.history),
        "n_survived_gates": sum(h.n_survived_stage1 for h in result.history),
        "wall_seconds": result.total_time_seconds,
    }

def main():
    blind_scenarios = [s for s in get_all_scenarios() if s.tier == "blind"]
    
    proposers = {
        "Mock (base)":     CorrectionMockProposer(seed=SEED),
        "Mock (extended)": CorrectionMockProposer(seed=SEED, extended=True),
        "Grammar":         GrammarProposer(checker=DimensionalChecker(), seed=SEED),
    }
    
    results = []
    for scenario in blind_scenarios:
        for p_name, proposer in proposers.items():
            for noise in NOISE_LEVELS:
                r = run_single(scenario, proposer, noise, SEED)
                r["proposer"] = p_name
                results.append(r)
                # Print immediate diagnostic
                status = "SUCCESS" if r["success"] else "FAILED"
                print(f"{status:7} | {p_name:20} | {scenario.name:35} | "
                      f"noise={noise:.0%} | NMSE={r['nmse_residual']:.3e} | "
                      f"expr={r['discovered_expr']}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Success Rate per Proposer")
    print("="*80)
    for p_name in proposers:
        p_results = [r for r in results if r["proposer"] == p_name and r["noise"] == 0.05]
        success = sum(r["success"] for r in p_results)
        total = len(p_results)
        print(f"{p_name:20}: {success}/{total} ({100*success/total:.1f}%) at 5% noise")
    
    # Save JSON
    with open("blind_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to blind_benchmark_results.json")

if __name__ == "__main__":
    main()
```

**Target output untuk paper:**
```
| Proposer       | 0% noise | 1% noise | 5% noise | 10% noise |
|----------------|----------|----------|----------|-----------|
| Mock (base)    | X/9      | X/9      | X/9      | X/9       |
| Mock (extended)| X/9      | X/9      | X/9      | X/9       |
| Grammar        | X/9      | X/9      | X/9      | X/9       |
```

**Success criteria:**
- [ ] Script berjalan tanpa error untuk semua 9 blind scenarios
- [ ] Output JSON tersimpan dengan benar
- [ ] Grammar >= Mock (base) pada minimal 1 scenario (proves Grammar value)

---

### GENIUS ADDITION — Task P1-6: Residual Asymptotic Signature (RAS)
**File:** `src/adcd/residual_analyzer.py`

**Ide:** Sebelum proposer dijalankan, hitung **leading-order exponent** dari
residual δ saat x → x₀ (classical limit). Ini adalah 6th residual feature yang
sangat diagnostik dan belum ada di framework manapun.

**Fisikanya:** Koreksi fisika hampir selalu bisa ditulis:
```
δ(x) ≈ C · (x - x₀)^n + higher order terms   saat x → x₀
```
Dengan log-log regression dekat x₀, kita bisa estimate n.

- n ≈ 1 → koreksi polynomial/linear
- n ≈ 2 → koreksi polynomial/quadratic  
- n ≈ 0.5 → koreksi power_law (non-integer)
- n → ∞ (decay) → koreksi exponential
- n berosilasi → trigonometric

**Implementasi:**
```python
def compute_ras(self, x_vals: np.ndarray, delta_vals: np.ndarray, 
                limit_val: float) -> dict:
    """
    Residual Asymptotic Signature: estimates leading-order behavior
    of residual as x → classical limit.
    
    Returns:
        {
          "leading_exponent": float,  # n in δ ~ (x-x₀)^n
          "leading_coeff": float,     # C
          "fit_quality": float,       # R² of log-log fit
          "suggested_class": str,     # "polynomial", "power_law", "exponential"
        }
    """
    # Take points nearest to classical limit (bottom 20%)
    dist_to_limit = np.abs(x_vals - limit_val)
    thresh = np.percentile(dist_to_limit, 20)
    mask = dist_to_limit <= thresh
    
    x_near = x_vals[mask]
    d_near = np.abs(delta_vals[mask])
    
    # Filter zeros and invalid
    valid = (d_near > 1e-15) & (dist_to_limit[mask] > 1e-10)
    if valid.sum() < 5:
        return {"leading_exponent": None, "fit_quality": 0.0, 
                "suggested_class": "unknown"}
    
    log_dist = np.log(dist_to_limit[mask][valid])
    log_delta = np.log(d_near[valid])
    
    # Linear regression in log-log space
    coeffs = np.polyfit(log_dist, log_delta, 1)
    n_estimate = coeffs[0]
    residuals = log_delta - np.polyval(coeffs, log_dist)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_delta - log_delta.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    
    # Classify
    if r2 < 0.5:
        suggested = "exponential"  # poor power-law fit → likely exp
    elif abs(n_estimate - round(n_estimate)) < 0.15:
        suggested = "polynomial"   # integer exponent
    else:
        suggested = "power_law"    # non-integer exponent
    
    return {
        "leading_exponent": float(n_estimate),
        "leading_coeff": float(np.exp(coeffs[1])),
        "fit_quality": float(r2),
        "suggested_class": suggested,
    }
```

**Integrasi ke ResidualFeatures dataclass:**
```python
@dataclass
class ResidualFeatures:
    # ... existing 5 features ...
    leading_exponent: Optional[float] = None      # NEW: dari RAS
    ras_suggested_class: Optional[str] = None      # NEW: class hint dari RAS
    ras_fit_quality: Optional[float] = None        # NEW: confidence RAS
```

**Integrasi ke proposer weighting:**
```python
# Di MockProposer dan GrammarProposer, setelah residual features didapat:
if rf.ras_suggested_class == "exponential":
    # Boost exponential templates weight 3x
elif rf.ras_suggested_class == "polynomial":
    # Boost polynomial templates weight 3x
elif rf.ras_suggested_class == "power_law":
    # Boost power_law templates weight 3x
```

**Kenapa ini genius untuk paper:**
Ini adalah kontribusi algoritmik baru yang belum ada di AI Feynman, PySR, atau PhySO.
Paper bisa claim: *"ADCD uniquely combines asymptotic expansion analysis (RAS)
with grammar-based correction search, directly emulating how physicists
derive perturbative corrections in theoretical physics."*

**Success criteria:**
- [ ] pytest tests/ pass
- [ ] RAS memberikan `leading_exponent` yang benar untuk Relativistic KE
      (v → 0, δ ~ v², sehingga n ≈ 2)
- [ ] RAS `suggested_class` match dengan `correction_class` untuk ≥7/9
      original scenarios

---

## PHASE 3: BAYESIAN OUTPUT

### Task P3-1: BayesianReranker
**File:** `src/adcd/bayesian_ranker.py` (FILE BARU)

```python
"""
Bayesian posterior estimation over discovered correction candidates.

Uses BIC weight approximation: posterior_i ∝ exp(-ΔBIC_i / 2)
This is the well-established Schwarz approximation to Bayes factors
(Kass & Raftery 1995, JASA).

Does NOT require MCMC or new infrastructure — uses BIC scores
already computed by the pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BayesianCorrectionOutput:
    """
    Bayesian posterior distribution over candidate corrections.
    
    candidates: list of (expression_str, bic_score) tuples, sorted by BIC
    posterior_weights: normalized posterior probability for each candidate
    correction_class_probs: aggregated probability per functional family
    is_ambiguous: True if top candidate weight < 3x second candidate weight
    evidence_label: "decisive" | "strong" | "substantial" | "weak" | "ambiguous"
    posterior_entropy: Shannon entropy of posterior (bits)
    """
    candidates: List[Tuple[str, float]]
    posterior_weights: List[float]
    correction_class_probs: dict
    is_ambiguous: bool
    evidence_label: str
    posterior_entropy: float
    best_expr: str
    best_weight: float


class BayesianReranker:
    """
    Converts BIC-ranked candidates to Bayesian posterior distribution.
    
    The BIC weight approximation is:
        w_i = exp(-ΔBIC_i / 2) / Σ exp(-ΔBIC_j / 2)
    where ΔBIC_i = BIC_i - BIC_min.
    
    This is equivalent to the Bayesian Information Criterion model averaging
    used in statistical model selection (Burnham & Anderson 2002).
    """
    
    EVIDENCE_THRESHOLDS = {
        "decisive":    150.0,   # ΔBIC > 10 → weight ratio > 150
        "very strong":  20.0,   # ΔBIC > 6
        "strong":        7.4,   # ΔBIC > 4  
        "substantial":   3.0,   # ΔBIC > 2.2
        "weak":          1.0,   # weight ratio > 1
    }
    
    def __init__(self, threshold_ratio: float = 0.05):
        """
        threshold_ratio: minimum posterior weight to include candidate.
                         Candidates below this are pruned from output.
        """
        self.threshold_ratio = threshold_ratio
    
    def rank(
        self, 
        candidates_with_bic: List[Tuple[str, float]]
    ) -> BayesianCorrectionOutput:
        """
        Convert BIC scores to posterior weights.
        
        Args:
            candidates_with_bic: List of (expr_str, bic_score) tuples
            
        Returns:
            BayesianCorrectionOutput with full posterior distribution
        """
        if not candidates_with_bic:
            raise ValueError("No candidates provided to BayesianReranker")
        
        # Sort by BIC ascending (lower BIC = better)
        sorted_cands = sorted(candidates_with_bic, key=lambda x: x[1])
        exprs = [c[0] for c in sorted_cands]
        bics = np.array([c[1] for c in sorted_cands])
        
        # Compute BIC weights: w_i ∝ exp(-ΔBIC_i / 2)
        delta_bic = bics - bics.min()
        log_weights = -0.5 * delta_bic
        # Numerical stability: subtract max before exp
        log_weights -= log_weights.max()
        raw_weights = np.exp(log_weights)
        
        # Prune low-weight candidates
        threshold = raw_weights.max() * self.threshold_ratio
        mask = raw_weights >= threshold
        exprs_pruned = [e for e, m in zip(exprs, mask) if m]
        weights_pruned = raw_weights[mask]
        
        # Normalize
        weights_norm = weights_pruned / weights_pruned.sum()
        
        # Compute posterior entropy (bits)
        entropy = float(-np.sum(
            weights_norm * np.log2(weights_norm + 1e-15)
        ))
        
        # Aggregate by functional family
        from adcd.metrics import classify_structure
        import sympy as sp
        
        class_probs = {}
        for expr_str, w in zip(exprs_pruned, weights_norm):
            try:
                fam = classify_structure(sp.sympify(expr_str))
            except Exception:
                fam = "unknown"
            class_probs[fam] = class_probs.get(fam, 0.0) + float(w)
        
        # Determine evidence label
        if len(weights_norm) >= 2:
            weight_ratio = float(weights_norm[0] / weights_norm[1])
        else:
            weight_ratio = float("inf")
        
        evidence_label = "ambiguous"
        for label, threshold in self.EVIDENCE_THRESHOLDS.items():
            if weight_ratio >= threshold:
                evidence_label = label
                break
        
        is_ambiguous = weight_ratio < 3.0
        
        return BayesianCorrectionOutput(
            candidates=list(zip(exprs_pruned, bics[mask].tolist())),
            posterior_weights=weights_norm.tolist(),
            correction_class_probs=class_probs,
            is_ambiguous=is_ambiguous,
            evidence_label=evidence_label,
            posterior_entropy=entropy,
            best_expr=exprs_pruned[0],
            best_weight=float(weights_norm[0]),
        )
```

**Unit tests (tambahkan ke `tests/test_bayesian_ranker.py`):**
```python
def test_decisive_evidence():
    """Single dominant candidate → decisive evidence."""
    candidates = [("theta_0 * v**2", -1000.0), ("theta_0 * exp(-v)", -990.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.evidence_label in ("decisive", "very strong", "strong")
    assert not out.is_ambiguous
    assert out.best_expr == "theta_0 * v**2"

def test_ambiguous_evidence():
    """Nearly equal BIC → ambiguous."""
    candidates = [("theta_0 * v**2", -1000.0), ("theta_0 * v", -999.5)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.is_ambiguous

def test_posterior_sums_to_one():
    candidates = [("a", -100.0), ("b", -95.0), ("c", -80.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert abs(sum(out.posterior_weights) - 1.0) < 1e-10

def test_entropy_range():
    """Entropy should be >= 0."""
    candidates = [("a", -100.0), ("b", -95.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.posterior_entropy >= 0.0
```

**Success criteria:**
- [ ] pytest tests/test_bayesian_ranker.py pass (semua 4 tests)
- [ ] pytest tests/ pass (tidak ada regression)

---

### Task P3-2: IdentifiabilityAnalyzer
**File:** `src/adcd/identifiability.py` (FILE BARU)

```python
"""
Identifiability analysis for discovered corrections.

Diagnoses WHY a correction cannot be uniquely identified from data,
converting "system failed" narratives into precise, scientifically
meaningful statements about data information limits.

Three failure modes:
1. model_degeneracy: top-2 candidates statistically indistinguishable
2. low_snr: correction magnitude buried in noise
3. undetectable_magnitude: |Δ/y_classical| < threshold
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IdentifiabilityReport:
    """
    Identifiability analysis result.
    
    is_identifiable: True if correction can be reliably identified
    failure_mode: "model_degeneracy" | "low_snr" | "undetectable_magnitude" | None
    snr: Signal-to-noise ratio of the correction
    weight_ratio: Posterior weight ratio of top-2 candidates
    relative_magnitude: |Δ|/|y_classical| median
    summary: Human-readable summary for paper reporting
    """
    is_identifiable: bool
    failure_mode: Optional[str]
    snr: float
    weight_ratio: float
    relative_magnitude: float
    summary: str


class IdentifiabilityAnalyzer:
    """
    Analyzes whether the discovered correction is identifiable from data.
    
    Inspired by identifiability analysis in statistics and physics:
    - Fisher information / Cramér-Rao bound (parameter identifiability)
    - Rissanen's MDL (information-theoretic identifiability)
    - Physics: Screened Coulomb vs Yukawa are indistinguishable below SNR threshold
    """
    
    SNR_THRESHOLD = 1.0
    WEIGHT_RATIO_THRESHOLD = 3.0
    MAGNITUDE_THRESHOLD = 1e-10
    
    def analyze(
        self,
        bayesian_output,        # BayesianCorrectionOutput
        residual: np.ndarray,
        y_classical: np.ndarray,
        noise_level: float = 0.0,
    ) -> IdentifiabilityReport:
        
        # Compute SNR: correction signal vs noise
        correction_magnitude = float(np.std(residual))
        noise_magnitude = float(noise_level * np.std(y_classical)) + 1e-15
        snr = correction_magnitude / noise_magnitude
        
        # Compute relative magnitude
        relative_magnitude = float(np.median(
            np.abs(residual) / (np.abs(y_classical) + 1e-15)
        ))
        
        # Compute weight ratio from posterior
        weights = bayesian_output.posterior_weights
        if len(weights) >= 2:
            weight_ratio = float(weights[0] / (weights[1] + 1e-15))
        else:
            weight_ratio = float("inf")
        
        # Diagnose failure mode
        failure_mode = None
        is_identifiable = True
        
        if relative_magnitude < self.MAGNITUDE_THRESHOLD:
            failure_mode = "undetectable_magnitude"
            is_identifiable = False
        elif snr < self.SNR_THRESHOLD:
            failure_mode = "low_snr"
            is_identifiable = False
        elif weight_ratio < self.WEIGHT_RATIO_THRESHOLD:
            failure_mode = "model_degeneracy"
            is_identifiable = False
        
        # Build human-readable summary
        if is_identifiable:
            summary = (
                f"IDENTIFIABLE ({bayesian_output.evidence_label} evidence, "
                f"SNR={snr:.1f}, weight_ratio={weight_ratio:.1f})"
            )
        else:
            if failure_mode == "model_degeneracy":
                summary = (
                    f"NOT IDENTIFIABLE: model_degeneracy "
                    f"(weight_ratio={weight_ratio:.1f} < {self.WEIGHT_RATIO_THRESHOLD}, "
                    f"top candidates are statistically indistinguishable)"
                )
            elif failure_mode == "low_snr":
                summary = (
                    f"NOT IDENTIFIABLE: low_snr "
                    f"(SNR={snr:.2f} < {self.SNR_THRESHOLD}, "
                    f"correction buried in observational noise)"
                )
            else:
                summary = (
                    f"NOT IDENTIFIABLE: undetectable_magnitude "
                    f"(|Δ/y|={relative_magnitude:.2e} < {self.MAGNITUDE_THRESHOLD})"
                )
        
        return IdentifiabilityReport(
            is_identifiable=is_identifiable,
            failure_mode=failure_mode,
            snr=snr,
            weight_ratio=weight_ratio,
            relative_magnitude=relative_magnitude,
            summary=summary,
        )
```

**Unit tests (tambahkan ke `tests/test_identifiability.py`):**
```python
def test_identifiable_high_snr():
    report = analyzer.analyze(bayesian_high_confidence, residual_strong, y_classical)
    assert report.is_identifiable
    assert report.failure_mode is None

def test_not_identifiable_low_snr():
    report = analyzer.analyze(bayesian_ambiguous, residual_weak, y_classical, noise_level=0.5)
    assert not report.is_identifiable
    assert report.failure_mode == "low_snr"

def test_not_identifiable_model_degeneracy():
    report = analyzer.analyze(bayesian_ambiguous, residual_moderate, y_classical)
    assert not report.is_identifiable
    assert report.failure_mode == "model_degeneracy"
```

**Success criteria:**
- [ ] pytest tests/test_identifiability.py pass
- [ ] pytest tests/ pass (tidak ada regression)

---

### Task P3-3: Integrasi ke Orchestrator (MINIMAL TOUCH)
**File:** `src/adcd/correction_orchestrator.py`

Ini adalah satu-satunya task yang menyentuh orchestrator. Perubahan harus
**purely additive** — tidak mengubah logika existing, hanya menambahkan output.

**1. Tambahkan field ke `CorrectionSearchResult`:**
```python
@dataclass
class CorrectionSearchResult:
    # ... existing fields TIDAK DIUBAH ...
    
    # Phase 3 additions (optional, default None untuk backward compatibility)
    bayesian_output: Optional["BayesianCorrectionOutput"] = None
    identifiability_report: Optional["IdentifiabilityReport"] = None
```

**2. Di akhir `search_correction()`, sebelum `return`:**
```python
# Phase 3: Bayesian posterior (purely additive, tidak ubah best_expr)
bayesian_out = None
ident_report = None

if all_candidates_bic and len(all_candidates_bic) >= 2:
    try:
        from adcd.bayesian_ranker import BayesianReranker
        from adcd.identifiability import IdentifiabilityAnalyzer
        
        reranker = BayesianReranker(threshold_ratio=0.05)
        bayesian_out = reranker.rank(all_candidates_bic)
        
        analyzer = IdentifiabilityAnalyzer()
        ident_report = analyzer.analyze(
            bayesian_output=bayesian_out,
            residual=residual,
            y_classical=y_classical,
            noise_level=noise_level,
        )
        logger.info(f"[Phase3] {ident_report.summary}")
    except Exception as e:
        logger.warning(f"[Phase3] Bayesian analysis failed: {e}")
        # Phase 3 failure NEVER propagates to main result

return CorrectionSearchResult(
    # ... existing fields unchanged ...
    bayesian_output=bayesian_out,
    identifiability_report=ident_report,
)
```

**PENTING:** `all_candidates_bic` adalah list yang dikumpulkan selama search loop.
Verifikasi bahwa variabel ini sudah ada di orchestrator. Jika belum, tambahkan
accumulation loop:
```python
all_candidates_bic = []  # inisialisasi sebelum iterasi loop
# Di dalam loop, setelah BIC dihitung:
all_candidates_bic.append((best_expr, best_bic))
```

**Success criteria:**
- [ ] pytest tests/ pass (tidak ada regression di semua tests existing)
- [ ] `CorrectionSearchResult` punya field `bayesian_output` dan `identifiability_report`
- [ ] Ketika Bayesian analysis gagal, main result TETAP dikembalikan dengan benar
- [ ] Log menunjukkan "[Phase3] IDENTIFIABLE/NOT IDENTIFIABLE ..."

---

## URUTAN EKSEKUSI

```
P1-1 (label fix, 2 lines)     → commit → pytest ✓
    ↓
P1-2 (ARC pre-filter)          → commit → pytest ✓
    ↓
P1-3 (correlation ranking)     → commit → pytest ✓
    ↓
P1-5 (benchmark script)        → commit → run benchmark → catat baseline
    ↓
P1-4 (bivariate templates)     → commit → pytest ✓ → verify di benchmark
    ↓
P1-6 (RAS - genius addition)   → commit → pytest ✓ → verify RAS accuracy
    ↓
P3-1 (BayesianReranker)        → commit → pytest ✓ (new tests)
    ↓
P3-2 (IdentifiabilityAnalyzer) → commit → pytest ✓ (new tests)
    ↓
P3-3 (Integration)             → commit → pytest ✓ → run full benchmark
```

**CI harus HIJAU setelah setiap commit. Jika merah: STOP dan debug.**

---

## PHASE 2: ARSITEKTUR MASA DEPAN (JANGAN IMPLEMENTASI SEKARANG)

### Mengapa Phase 2 Ditunda

Phase 2 (multivariable corrections) memerlukan perubahan arsitektur yang signifikan.
Berdasarkan analisis mendalam dari bug-bug yang ditemukan, implementasi yang terburu-buru
menghasilkan 0/4 dengan 10 bugs terdokumentasi. Phase 2 yang benar membutuhkan:

1. Fundamental design decision yang belum diselesaikan
2. Test suite yang dibuat SEBELUM implementasi (TDD)
3. Minimal 2-3 hari development focused tanpa gangguan Phase 3

### Arsitektur yang Benar untuk Phase 2

**Prinsip:** Decompose multivariable correction menjadi produk/jumlah 1D corrections.

```
Δ(x₁, x₂) ≈ f(x₁) · g(x₂)   [multiplicative separable]
           OR f(x₁) + g(x₂)   [additive separable]
           OR f(x₁/x₂)         [ratio-based, via Buckingham-Pi]
```

**Step 1:** ResidualFactorizer test separability
- Jika separable → jalankan 2x 1D ADCD, kalikan/tambahkan hasilnya
- Jika tidak separable → gunakan Buckingham-Pi product templates

**Step 2:** DimensionalChecker extension untuk multivariable
- Setiap faktor harus dimensionless secara independen
- JANGAN bypass dimensional gate — perluas untuk handle faktor terpisah

**Step 3:** ARC gate extension untuk multi-limit
- Test setiap limit secara INDEPENDEN (bukan simultaneous)
- Correction harus vanish di setiap limit variable secara terpisah

**Step 4:** Scenario design yang tepat
- Semua θ parameters harus dimensionless by design
- Gunakan known physical constants sebagai reference scales
- Setiap scenario harus punya documented ground truth yang accessible

### Pre-conditions untuk Mulai Phase 2

```
□ Phase 1 + 3 complete dan CI 3/3 passing
□ Blind benchmark ≥ 7/9 untuk single-variable scenarios  
□ Test suite untuk ResidualFactorizer ditulis dulu (TDD)
□ Scenario definitions reviewed untuk θ-dimensionless consistency
□ Minimum 1 working day dedicated tanpa switching task
```

---

## DEFINISI SELESAI

**Phase 1 + 3 DONE ketika:**
```
□ pytest tests/ = 3/3 CI PASS (semua tests hijau)
□ Blind benchmark ≥ 7/9 untuk Mock (extended) pada 5% noise
□ Grammar > Mock (base) pada minimal 1 scenario
□ BayesianReranker output tersedia di CorrectionSearchResult
□ IdentifiabilityAnalyzer output tersedia
□ Paper main.tex ter-compile dengan:
    - Afiliasi: "Independent Researcher, Indonesia"
    - Section baru Phase 3 Bayesian output
    - Updated blind test results (X/9 vs previous 1/3)
    - RAS dijelaskan sebagai novel contribution
```

---

## CATATAN UNTUK PAPER

### Kontribusi Baru yang Bisa Diklaim di Paper

1. **Correction-first paradigm** (sudah ada) — ADCD vs tabula rasa SR

2. **Physics-gated search** (sudah ada) — AST/Dim/ARC gates

3. **Residual Asymptotic Signature (RAS)** (BARU dari P1-6):
   *"First symbolic regression framework to compute leading-order asymptotic
   expansion of residual as a structural prior, directly emulating perturbative
   expansion methods in theoretical physics."*

4. **Bayesian posterior over functional families** (BARU dari P3):
   *"First physics-constrained SR framework to provide honest uncertainty
   quantification via BIC-weight posterior, distinguishing decisive recovery
   from model degeneracy."*

5. **Identifiability analysis** (BARU dari P3):
   *"Novel contribution: systematic diagnosis of why a correction cannot be
   identified (model_degeneracy vs low_snr vs undetectable_magnitude),
   converting failure narratives into precise scientific statements."*

---

## ATURAN ABSOLUTE YANG TIDAK BOLEH DILANGGAR

```
JANGAN ubah:     pipeline.py physics gates (AST/Dim/ARC)
JANGAN ubah:     BIC formula: N*ln(NMSE) + k*ln(N)
JANGAN ubah:     JAXOptimizer core L-BFGS-B
JANGAN ubah:     Original 9 benchmark scenarios
JANGAN commit:   Kalau pytest merah
JANGAN lanjut:   Phase 2 sampai Phase 1+3 benar-benar done
JANGAN campurkan: Task dari Phase berbeda dalam 1 commit
```
