# ADCD Phase 2: Multivariable Correction Discovery
## Genius-Level Complete Plan — Learned from All Previous Mistakes

---

## RETROSPECTIVE: MENGAPA PHASE 2 SEBELUMNYA GAGAL

Sebelum menulis satu baris kode, pahami 5 kesalahan fundamental:

```
Mistake 1 — Architecture before tests:
  Phase 2 diimplementasi tanpa test suite end-to-end.
  Bug baru tidak terdeteksi sampai benchmark jalan.
  → FIX: TDD. Tulis semua tests SEBELUM implementasi.

Mistake 2 — θ berdimensi:
  Scenarios mendefinisikan correction_expr dengan θ yang
  punya satuan fisik (θ₀=[L], θ₁=[M]). ADCD model
  mengasumsikan semua θ dimensionless.
  → FIX: Semua scenarios WAJIB θ-dimensionless dari awal.

Mistake 3 — ARC simultaneous:
  Pre-filter men-set SEMUA limit variables sekaligus.
  Correction (T→∞) AND (A→0) dites bersamaan, bukan terpisah.
  → FIX: Test setiap limit variable INDEPENDEN, satu per satu.

Mistake 4 — Dimensional gate terlalu broad:
  Pipeline.py di-bypass untuk SEMUA has_params=True,
  bukan hanya multivariable scenarios.
  → FIX: Bypass HANYA untuk is_multivariable AND has_params.
  Better yet: ganti paradigm — check per factor, bukan total.

Mistake 5 — Simplify targets bukan fix tools:
  Blind-12 di-simplify dari power law ke bilinear θ₀·A·T
  agar lebih mudah di-solve. Ini bukan improvement genuine.
  → FIX: Design scenarios yang physically meaningful dan
  konsisten dengan ADCD model. Jangan compromise ground truth.
```

---

## PRINSIP ARSITEKTUR BARU

### Insight Fundamental: Buckingham-Pi sebagai Foundation

Koreksi fisika multivariable SELALU bisa ditulis dalam
**Buckingham-Pi dimensionless groups** (Π₁, Π₂, ..., Πₙ₋ₖ).

```
Untuk n variabel fisik dengan k dimensi fundamental:
  Ada n-k kelompok dimensionless Πᵢ (Buckingham-Pi theorem)

Contoh:
  Variabel: {m[M], M[M], r[L], G[L³M⁻¹T⁻²]}
  Dimensi fundamental: {M, L, T} → k=3
  n=4 variabel → n-k=1 Pi group: Π₁ = m/M (pure ratio!)

  Variabel: {v[LT⁻¹], ρ[ML⁻³], b[MT⁻¹], v_ref[LT⁻¹], ρ_ref[ML⁻³]}
  Π₁ = v/v_ref, Π₂ = ρ/ρ_ref → 2D search di (Π₁, Π₂) space
```

**Implikasinya:**
- Semua kandidat yang digenerate DALAM Pi-group space otomatis dimensionless
- Tidak perlu dimensional gate bypass — semua pass by construction
- ARC gate berlaku secara natural pada masing-masing Πᵢ

### Insight Genius: Product Decomposition

**Teorema:** Jika Δ = f(Π₁) · g(Π₂) dimana:
- f(Π₁) → 0 ketika Π₁ → 0 (classical limit untuk x₁)
- g(Π₂) terbatas (finite) ketika Π₁ → 0

Maka Δ → 0 ketika Π₁ → 0. **ARC-compliant by construction.**

Ini berarti: **produk dua koreksi 1D yang masing-masing ARC-safe
secara otomatis menghasilkan koreksi 2D yang ARC-safe.**

```python
# Contoh:
f(Π₁) = θ₀ · (m/M)           # vanish di m→0  ✓
g(Π₂) = exp(-r/θ₁)           # finite di m→0   ✓
Δ = f · g = θ₀·(m/M)·exp(-r/θ₁)  # vanishes at m→0 ✓
                                   # vanishes at r→∞  ✓ (exp term)
```

### Insight Genius: Pi-Sparse Decomposition (PSAD) — NOVEL CONTRIBUTION

**Ide yang belum ada di AI Feynman, PySR, atau PhySO:**

Sebelum searching, lakukan **Sparse Pi-Group Selection**:
1. Compute semua Π groups dari Buckingham-Pi
2. Compute Pearson correlation |ρ(Πᵢ, δ)| dan |ρ(Πᵢ·Πⱼ, δ)|
3. Run sparse regression (Lasso) untuk menemukan subset Π minimal
4. Search HANYA dalam subspace Π yang terpilih

Ini adalah **sparse dimensionless variable selection** — secara langsung
melawan curse of dimensionality dalam multivariable SR.

Paper claim yang bisa dibuat:
*"ADCD Phase 2 introduces Pi-Sparse Anomaly Decomposition (PSAD), the first
method to perform automatic sparse selection of Buckingham-Pi groups before
symbolic search, reducing the effective dimensionality of the correction
subspace without loss of physical consistency."*

---

## SCENARIO DESIGN (HARUS SELESAI SEBELUM CODING)

### Aturan Desain Scenario

```
WAJIB:
□ Semua correction_expr dalam bentuk θ-dimensionless
  (θ adalah pure scaling factor, bukan satuan fisik)
□ Koreksi menggunakan Pi groups atau known constants sebagai reference
□ Koreksi vanish di SETIAP classical limit secara INDEPENDEN
□ correction_class dapat diverifikasi dengan classify_structure()
□ Ground truth REACHABLE oleh grammar (product of 1D ARC-safe forms)

DILARANG:
□ θ₀ dengan satuan [L], [M], [T], dsb
□ Koreksi yang hanya vanish ketika SEMUA limits diterapkan bersamaan
□ Simplify ground truth agar lebih mudah di-solve
```

### 4 Validated Multivariable Scenarios

**MV-1: Yukawa Mass-Ratio Screening**
```python
AnomalyScenario(
    name="MV-1: Yukawa Mass-Ratio",
    classical_expr="G * m * M / r**2",
    classical_variables=["m", "M", "r"],
    classical_constants={"G": 6.6743e-11, "r_0": 2.5},  # r_0 = known scale [m]
    correction_type="multiplicative",
    # Pi groups: Π₁ = m/M (mass ratio), Π₂ = r/r_0
    # Δ = θ₀ · (m/M) · exp(-r/r_0)
    # Vanishes: m→0 (Π₁→0) ✓ and r→∞ (exp→0) ✓
    correction_expr="theta_0 * (m / M) * exp(-r / r_0)",
    correction_constants={"theta_0": 0.50},  # θ pure dimensionless
    correction_class="exponential",
    variables_with_units={"m": "kg", "M": "kg", "r": "m"},
    classical_limit_variable=["r", "m"],
    classical_limit_direction=["oo", "0"],
)
```

**MV-2: Plasma Density-Temperature Correction**
```python
AnomalyScenario(
    name="MV-2: Plasma Correction",
    classical_expr="n * k_B * T",  # ideal gas pressure
    classical_variables=["n", "T"],
    classical_constants={"k_B": 1.38e-23, "n_ref": 1e20, "T_ref": 1000.0},
    correction_type="multiplicative",
    # Pi groups: Π₁ = n/n_ref, Π₂ = T/T_ref
    # Δ = θ₀ · (n/n_ref) · (T_ref/T)^0.5
    # Vanishes: n→0 (Π₁→0) ✓ and T→∞ (Π₂→∞, 1/√T→0) ✓
    correction_expr="theta_0 * (n / n_ref) * (T_ref / T)**0.5",
    correction_constants={"theta_0": 0.3},
    correction_class="power_law",
    variables_with_units={"n": "1/m^3", "T": "K"},
    classical_limit_variable=["n", "T"],
    classical_limit_direction=["0", "oo"],
)
```

**MV-3: Turbulent Drag 2D**
```python
AnomalyScenario(
    name="MV-3: Turbulent Drag 2D",
    classical_expr="b * v",  # Stokes drag
    classical_variables=["v", "rho"],
    classical_constants={"b": 1.0, "v_ref": 10.0, "rho_ref": 1.0},
    correction_type="additive",
    # Pi groups: Π₁ = v/v_ref, Π₂ = ρ/ρ_ref
    # Δ = θ₀ · (v/v_ref)² · (ρ/ρ_ref)
    # Vanishes: v→0 (Π₁→0) ✓ and ρ→0 (Π₂→0) ✓
    correction_expr="theta_0 * (v / v_ref)**2 * (rho / rho_ref)",
    correction_constants={"theta_0": 0.5},
    correction_class="polynomial",
    variables_with_units={"v": "m/s", "rho": "kg/m^3"},
    classical_limit_variable=["v", "rho"],
    classical_limit_direction=["0", "0"],
)
```

**MV-4: Van der Waals Pressure Correction**
```python
AnomalyScenario(
    name="MV-4: Van der Waals 2D",
    classical_expr="n * k_B * T / V",  # ideal gas pressure
    classical_variables=["n", "V"],
    classical_constants={"k_B": 1.38e-23, "T": 300.0,
                         "n_ref": 1.0, "V_ref": 1.0},
    correction_type="additive",
    # Pi groups: Π₁ = n/n_ref, Π₂ = V/V_ref
    # Δ = θ₀ · (n/n_ref)² / (V/V_ref)² = van der Waals a*n²/V² term
    # Vanishes: n→0 (Π₁→0) ✓ and V→∞ (1/Π₂→0) ✓
    correction_expr="theta_0 * (n / n_ref)**2 / (V / V_ref)**2",
    correction_constants={"theta_0": 0.1},
    correction_class="power_law",
    variables_with_units={"n": "mol", "V": "m^3"},
    classical_limit_variable=["n", "V"],
    classical_limit_direction=["0", "oo"],
)
```

**Verifikasi scenarios sebelum coding:**
```python
# Script verifikasi (jalankan manual):
for sc in [MV1, MV2, MV3, MV4]:
    expr = sp.sympify(sc.correction_expr)
    # 1. Semua θ adalah pure symbol (bukan dalam satuan fisik):
    theta_syms = [s for s in expr.free_symbols if str(s).startswith("theta")]
    assert len(theta_syms) >= 1

    # 2. ARC independent per variable:
    for var, direction in zip(sc.classical_limit_variable,
                               sc.classical_limit_direction):
        lim_val = 0 if direction == "0" else sp.oo
        # Substitute theta=1, other vars=midpoint
        subs = {s: 1.0 for s in expr.free_symbols
                if str(s).startswith("theta")}
        for other_var in sc.classical_variables:
            if other_var != var:
                subs[sp.Symbol(other_var)] = 1.0
        lim_result = sp.limit(expr.subs(subs), sp.Symbol(var), lim_val)
        assert lim_result == 0, f"ARC FAIL: {sc.name}, var={var}"

    print(f"✅ {sc.name}: θ-dimensionless OK, ARC per-variable OK")
```

---

## TASK LIST (TDD ORDER — TESTS FIRST)

### Task MV-T0: Test Suite (TULIS INI DULU, SEBELUM APAPUN)

**File:** `tests/test_phase2_components.py` (FILE BARU)

Tulis tests untuk semua komponen yang AKAN dibuat:

```python
"""
Phase 2 component tests. Written BEFORE implementation (TDD).
All tests start as EXPECTED_FAIL, become PASS as implementation progresses.
"""

# === Tests untuk BuckinghamPiEngine ===
class TestBuckinghamPiEngine:
    def test_mass_ratio_scenario(self):
        """m/M adalah satu-satunya Pi group untuk Yukawa Mass-Ratio."""
        engine = BuckinghamPiEngine()
        engine.register("m", [1, 0, 0])   # [M, L, T]
        engine.register("M", [1, 0, 0])
        engine.register("r", [0, 1, 0])
        engine.register("r_0", [0, 1, 0])  # known constant
        pi_groups = engine.compute_pi_groups()
        # Should find m/M and r/r_0
        assert len(pi_groups) == 2

    def test_turbulent_drag_scenario(self):
        """v/v_ref dan ρ/ρ_ref untuk turbulent drag."""
        engine = BuckinghamPiEngine()
        engine.register("v", [0, 1, -1])
        engine.register("v_ref", [0, 1, -1])
        engine.register("rho", [1, -3, 0])
        engine.register("rho_ref", [1, -3, 0])
        pi_groups = engine.compute_pi_groups()
        assert len(pi_groups) == 2
        # Both groups should be dimensionless ratios


# === Tests untuk SequentialARCChecker ===
class TestSequentialARCChecker:
    def test_independent_limit_pass(self):
        """(m/M)*exp(-r/r_0) vanishes at m→0 independently."""
        checker = SequentialARCChecker()
        expr = sp.sympify("theta_0 * (m/M) * exp(-r/r_0)")
        result = checker.check(
            expr,
            limit_vars=["r", "m"],
            limit_dirs=["oo", "0"],
            constants={"r_0": 2.5}
        )
        assert result.passes
        assert result.vanishing_at == ["m→0", "r→∞"]

    def test_simultaneous_only_fails(self):
        """Expression that only vanishes when BOTH limits applied simultaneously
        should FAIL sequential check."""
        checker = SequentialARCChecker()
        # This expression only vanishes when BOTH x→0 AND y→0
        expr = sp.sympify("theta_0 * x * y")
        # But if x→0 independently: θ₀·0·y = 0 ✓ (actually passes!)
        # A better failing example:
        expr_fail = sp.sympify("theta_0 * (x + y)")  # neither term alone → 0
        result = checker.check(
            expr_fail,
            limit_vars=["x", "y"],
            limit_dirs=["0", "0"],
            constants={}
        )
        # x→0: θ₀·(0+y) = θ₀·y ≠ 0 → FAIL
        assert not result.passes

    def test_product_of_arc_safe_forms(self):
        """Product of two ARC-safe 1D forms is ARC-safe 2D."""
        checker = SequentialARCChecker()
        expr = sp.sympify("theta_0 * (v/v_ref)**2 * (rho/rho_ref)")
        result = checker.check(
            expr,
            limit_vars=["v", "rho"],
            limit_dirs=["0", "0"],
            constants={"v_ref": 10.0, "rho_ref": 1.0}
        )
        assert result.passes  # v→0: v²→0 ✓; rho→0: rho→0 ✓


# === Tests untuk ResidualFactorizerV2 ===
class TestResidualFactorizerV2:
    def test_multiplicative_separable(self):
        """f(x)·g(y) terdeteksi sebagai multiplicative separable."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 5, 100)
        y = rng.uniform(0.1, 2, 100)
        X = {"v": x, "rho": y}
        delta = 0.5 * x**2 * y  # = θ₀·Π₁²·Π₂, perfectly separable
        factorizer = ResidualFactorizerV2()
        result = factorizer.test_separability(X, delta)
        assert result.factorization_type == "multiplicative"
        assert result.explained_variance > 0.95

    def test_additive_separable(self):
        X = {"v": x, "rho": y}
        delta = 0.3 * x**2 + 0.7 * np.exp(-y)
        result = ResidualFactorizerV2().test_separability(X, delta)
        assert result.factorization_type in ("additive", "multiplicative")

    def test_non_separable_returns_none(self):
        """Coupled non-separable correction."""
        X = {"v": x, "rho": y}
        delta = 0.5 * x * y / (x + y)  # not separable
        result = ResidualFactorizerV2().test_separability(X, delta)
        # Should detect non-separability with low explained_variance
        assert result.factorization_type == "none" or \
               result.explained_variance < 0.8


# === Integration test: end-to-end MV-1 ===
class TestMultivariableEndToEnd:
    def test_yukawa_mass_ratio_discovery(self):
        """ADCD discovers θ₀·(m/M)·exp(-r/r_0) at 0% noise."""
        scenario = get_mv_scenario("MV-1: Yukawa Mass-Ratio")
        result = run_adcd_mv(scenario, noise=0.0, seed=42)
        assert result.evaluation.class_match  # exponential class
        assert result.best_nmse_residual < 0.01
        # Check it's actually using both variables
        expr = sp.sympify(result.best_expr)
        vars_found = {str(s) for s in expr.free_symbols
                      if s.name in ["m", "M", "r"]}
        assert len(vars_found) >= 2  # genuinely multivariable
```

**CI harus FAIL pada semua tests ini sampai implementasi selesai.**
Ini adalah bukti TDD bekerja dengan benar.

---

### Task MV-1: BuckinghamPiEngine

**File:** `src/adcd/buckingham_pi.py` (FILE BARU)

**Apa yang dilakukan:**
- Menerima registry variabel + dimensi mereka
- Menghitung nullspace dari dimension matrix menggunakan SVD
- Mengembalikan daftar Buckingham-Pi groups sebagai SymPy expressions
- Digunakan oleh GrammarProposer untuk generate ratio candidates

**Implementasi:**
```python
class BuckinghamPiEngine:
    """
    Computes dimensionless Buckingham-Pi groups from registered variables.
    
    Uses SVD of the dimensional matrix to find the null space,
    which corresponds to dimensionless combinations.
    
    Based on: Buckingham (1914), Phys. Rev., 4(4), 345–376.
    """
    def __init__(self):
        self.registry = {}  # name → dimension_vector [M, L, T, ...]
    
    def register(self, name: str, dim_vector: List[int]):
        """Register a variable with its dimension vector."""
        self.registry[name] = np.array(dim_vector, dtype=float)
    
    def compute_pi_groups(self) -> List[sp.Expr]:
        """
        Compute all independent dimensionless Pi groups.
        
        Returns:
            List of SymPy expressions representing each Pi group.
            Each expression is dimensionless by construction.
        """
        names = list(self.registry.keys())
        dim_matrix = np.array([self.registry[n] for n in names]).T
        
        # SVD to find null space
        _, s, Vt = np.linalg.svd(dim_matrix)
        rank = np.sum(s > 1e-10)
        null_space = Vt[rank:].T  # columns are null space basis vectors
        
        pi_groups = []
        syms = {n: sp.Symbol(n) for n in names}
        
        for col in null_space.T:
            # Each column of null space → one Pi group
            # Πᵢ = ∏ xⱼ^(null_col_j)
            pi_expr = sp.Mul(*[
                syms[n] ** sp.Rational(int(round(exp * 12)), 12)
                for n, exp in zip(names, col)
                if abs(exp) > 1e-6
            ])
            # Only include if it involves at least 2 different variables
            free_vars = {str(s) for s in pi_expr.free_symbols}
            if len(free_vars) >= 2:
                pi_groups.append(pi_expr)
        
        return pi_groups
    
    def get_parameterized_ratios(self) -> List[sp.Expr]:
        """
        Get parameterized forms of Pi groups: Πᵢ/θ for use as grammar ratios.
        Free parameters θ absorb dimensional scaling.
        """
        pi_groups = self.compute_pi_groups()
        ratios = []
        for i, pi in enumerate(pi_groups):
            theta = sp.Symbol(f"theta_pi_{i}")
            ratios.append(pi / theta)       # Πᵢ/θ
            ratios.append(pi * theta)       # Πᵢ·θ (for inverse ratios)
        return ratios
```

**Unit tests:** Test semua dengan scenarios dari Task MV-T0.
**Success criteria:** pytest test_phase2_components.py::TestBuckinghamPiEngine PASS

---

### Task MV-2: SequentialARCChecker

**File:** `src/adcd/sequential_arc.py` (FILE BARU)

**Apa yang dilakukan:**
- Menggantikan simultaneous ARC pre-filter
- Mengevaluasi setiap limit variable SECARA INDEPENDEN
- Menggunakan numerical evaluation (seperti ARC pre-filter di Phase 1)
- Mengembalikan detailed report: passes=True/False + vanishing_at list

**Core logic:**
```python
def check(self, expr, limit_vars, limit_dirs, constants, tol=0.05, n_samples=10):
    """
    Check ARC compliance for each limit variable INDEPENDENTLY.
    
    For Δ = f(x₁, x₂) with limits x₁→0 and x₂→∞:
    - Test: set x₁→0 (small value), x₂=midpoint, θ=random
      If |Δ| < tol → passes for x₁
    - Test: set x₂→∞ (large value), x₁=midpoint, θ=random
      If |Δ| < tol → passes for x₂
    - BOTH must pass → expression is ARC-compliant
    
    KEY DIFFERENCE from Phase 2 (failed):
    Old: set ALL limit vars to extreme simultaneously
    New: test each limit var INDEPENDENTLY, others at midpoint
    """
    vanishing_at = []
    
    for limit_var, limit_dir in zip(limit_vars, limit_dirs):
        var_passes = False
        
        for _ in range(n_samples):
            subs = {}
            # Set the limit variable to extreme value
            if limit_dir == "0":
                subs[sp.Symbol(limit_var)] = 1e-7
            else:
                subs[sp.Symbol(limit_var)] = 1e8
            
            # Set ALL OTHER variables to midpoint (not extreme)
            for other_var in [v for v in limit_vars if v != limit_var]:
                subs[sp.Symbol(other_var)] = 1.0
            
            # Set physical constants
            for const_name, const_val in constants.items():
                subs[sp.Symbol(const_name)] = const_val
            
            # Random theta values
            for s in expr.free_symbols:
                if str(s).startswith("theta"):
                    subs[s] = rng.uniform(0.1, 5.0)
            
            try:
                val = float(complex(expr.subs(subs)).real)
                if np.isfinite(val) and abs(val) < tol:
                    var_passes = True
                    break
            except Exception:
                continue
        
        if not var_passes:
            return SequentialARCResult(passes=False, vanishing_at=vanishing_at,
                                        failing_var=limit_var)
        vanishing_at.append(f"{limit_var}→{limit_dir}")
    
    return SequentialARCResult(passes=True, vanishing_at=vanishing_at)
```

**Integration ke pipeline:**
- Tambahkan `SequentialARCChecker` ke `Stage1Pipeline.__init__()`
- Hanya dipanggil untuk multivariable scenarios
- 1D scenarios tetap menggunakan existing ARC scorer
- JANGAN ubah existing ARC gate untuk 1D

**Success criteria:** pytest test_phase2_components.py::TestSequentialARCChecker PASS

---

### Task MV-3: ResidualFactorizerV2

**File:** `src/adcd/residual_factorizer_v2.py` (FILE BARU)
(bukan modify yang lama — buat baru untuk keamanan)

**Apa yang dilakukan:**
- Test multiplicative separability: δ(x₁,x₂) ≈ f(x₁)·g(x₂)
- Test additive separability: δ(x₁,x₂) ≈ f(x₁)+g(x₂)
- Test ratio-based: δ(x₁,x₂) ≈ h(x₁/x₂)
- Menggunakan isotonic regression dan variance decomposition
- Return components untuk digunakan di strategy selection

**Key improvement over Phase 2 version:**
```python
def test_separability(self, X: Dict, delta: np.ndarray) -> FactorizationResult:
    """
    Tests separability using variance decomposition approach.
    
    For multiplicative: log(|δ|) = log(|f(x₁)|) + log(|g(x₂)|)
    Fit additive model in log space → find multiplicative separability.
    
    This is more robust than polynomial fit because:
    1. Works for exp, power law, and polynomial functions
    2. Handles non-monotone functions via rank correlation
    3. Reports partial R² for each variable
    """
    var_names = list(X.keys())
    if len(var_names) < 2:
        return FactorizationResult(factorization_type="none", ...)
    
    # Try multiplicative in log space
    log_delta = np.log(np.abs(delta) + 1e-15)
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer
    
    # Fit additive spline model in log space: log|δ| = s₁(x₁) + s₂(x₂)
    # This captures multiplicative separability
    X_arr = np.column_stack([X[v] for v in var_names])
    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False)
    X_spline = spline.fit_transform(X_arr)
    model = Ridge(alpha=0.01)
    model.fit(X_spline, log_delta)
    r2_mult = model.score(X_spline, log_delta)
    
    # Try additive directly
    model_add = Ridge(alpha=0.01)
    model_add.fit(X_spline, delta)
    r2_add = model_add.score(X_spline, delta)
    
    if r2_mult > 0.85:
        return FactorizationResult(
            factorization_type="multiplicative",
            explained_variance=r2_mult,
            components={v: ... for v in var_names}
        )
    elif r2_add > 0.85:
        return FactorizationResult(
            factorization_type="additive",
            explained_variance=r2_add,
            components={v: ... for v in var_names}
        )
    else:
        return FactorizationResult(factorization_type="none",
                                    explained_variance=max(r2_mult, r2_add))
```

**Note:** Perlu install scikit-learn jika belum ada.
**Success criteria:** pytest test_phase2_components.py::TestResidualFactorizerV2 PASS

---

### Task MV-4: ProductGrammar — Multivariable Template Generator

**File:** `src/adcd/product_grammar.py` (FILE BARU)

**Apa yang dilakukan:**
- Generate product templates: Δ = f(Π₁) · g(Π₂)
- Setiap faktor diambil dari ARC-safe 1D unary templates
- Product dari dua ARC-safe 1D forms → ARC-safe 2D by construction
- Tidak memerlukan dimensional gate bypass

**Core idea:**
```python
class ProductGrammar:
    """
    Generates multivariable correction templates as products of
    ARC-safe 1D correction forms applied to different Pi groups.
    
    Mathematical guarantee:
    If f(Π₁) → 0 as Π₁ → 0 (ARC-safe for x₁)
    And g(Π₂) is finite as Π₁ → 0
    Then f(Π₁)·g(Π₂) → 0 as Π₁ → 0 (ARC-safe for x₁)
    
    Similarly for x₂.
    """
    
    # ARC-safe 1D unary templates (vanish at Π→0)
    ARC_SAFE_UNARIES = [
        "theta_0 * R",           # linear, vanishes at R→0
        "theta_0 * R**2",        # quadratic, vanishes at R→0
        "theta_0 * R**theta_1",  # power law, vanishes at R→0 for θ₁>0
        "theta_0 * (exp(-R/theta_1) - 1)",  # exp correction, vanishes at R→0
        "theta_0 * R * exp(-R/theta_1)",    # exponential decay times R
        "theta_0 * log(1 + R/theta_1)",     # log correction, vanishes at R→0
        "theta_0 * R / (1 + R/theta_1)",    # rational, vanishes at R→0
        "theta_0 * tanh(R/theta_1)**2",     # tanh², vanishes at R→0
        "theta_0 * sin(R/theta_1)",         # sin, vanishes at R→0
    ]
    
    # ARC-safe at Π→∞ (vanish as Π increases)
    ARC_SAFE_UNARIES_INF = [
        "theta_0 * exp(-R/theta_1)",        # exp decay, vanishes at R→∞
        "theta_0 * (theta_1/R)",            # 1/R, vanishes at R→∞
        "theta_0 * (theta_1/R)**2",         # 1/R², vanishes at R→∞
        "theta_0 * (theta_1/R)**theta_2",   # power law decay
    ]
    
    def generate(self, pi_groups, limit_specs, n_candidates):
        """
        Generate product templates for multivariable scenarios.
        
        Args:
            pi_groups: list of dimensionless Pi groups from BuckinghamPiEngine
            limit_specs: list of (var_name, direction) for each classical limit
            n_candidates: how many product templates to generate
        """
        # For each Pi group, determine which ARC template family applies
        # based on the limit direction for that group
        per_group_templates = {}
        for pi_group in pi_groups:
            # Get the primary variable in this Pi group
            primary_var = self._get_primary_var(pi_group, limit_specs)
            limit_dir = self._get_limit_dir(primary_var, limit_specs)
            
            if limit_dir == "0":
                templates = self.ARC_SAFE_UNARIES
            else:  # "oo"
                templates = self.ARC_SAFE_UNARIES_INF
            
            per_group_templates[str(pi_group)] = (pi_group, templates)
        
        # Generate products: for each combination of (f from group1, g from group2)
        products = []
        for (pi1, templates1), (pi2, templates2) in combinations(
            per_group_templates.values(), 2
        ):
            for t1, t2 in product(templates1[:5], templates2[:5]):
                R1 = pi1
                R2 = pi2
                f = sp.sympify(t1.replace("R", f"({R1})"))
                g = sp.sympify(t2.replace("R", f"({R2})"))
                product_expr = f * g
                products.append(str(product_expr))
        
        return products[:n_candidates]
```

**Success criteria:**
- `ProductGrammar.generate()` untuk MV-1 menghasilkan template yang
  mengandung `m/M` DAN `r` dalam satu ekspresi
- Semua generated templates pass SequentialARCChecker

---

### Task MV-5: MultivariableOrchestrator

**File:** `src/adcd/multivar_orchestrator.py` (FILE BARU)

**Apa yang dilakukan:**
- Strategy selection berdasarkan ResidualFactorizerV2 output
- Koordinasi antara BuckinghamPiEngine, ProductGrammar, SequentialARCChecker
- Menghasilkan candidates untuk Stage1Pipeline
- TIDAK mengubah Stage1Pipeline — tetap menggunakan existing gates

**Strategy selection logic:**
```python
class MultivariableOrchestrator:
    """
    Orchestrates multivariable correction discovery.
    
    Strategy selection based on residual factorization type:
    
    multiplicative → Factorized 1D search:
      Run 1D ADCD on each component separately
      Combine: Δ = Δ₁(x₁) · Δ₂(x₂)
    
    additive → Factorized 1D search:
      Run 1D ADCD on each component separately
      Combine: Δ = Δ₁(x₁) + Δ₂(x₂)
    
    none → Product Grammar search:
      Use BuckinghamPiEngine + ProductGrammar
      Search directly in (Π₁, Π₂) space
      Use SequentialARCChecker instead of standard ARC
    
    Pi-Sparse selection (GENIUS — for all strategies):
      Before searching, select minimal Pi groups via Lasso
      Only include Pi groups with |ρ(Πᵢ, δ)| > threshold
    """
    
    def discover_multivariable_correction(self, scenario, X, delta, noise_level):
        # Step 1: Pi-Sparse selection
        pi_engine = BuckinghamPiEngine()
        pi_engine.register_from_scenario(scenario)
        pi_groups = pi_engine.compute_pi_groups()
        
        # Compute correlations (PSAD — genius contribution)
        pi_correlations = {}
        for pi in pi_groups:
            try:
                pi_vals = lambdify_pi(pi, X)
                corr = abs(pearsonr(pi_vals, delta)[0])
                pi_correlations[str(pi)] = corr
            except:
                pi_correlations[str(pi)] = 0.0
        
        # Select only Pi groups with |ρ| > 0.1
        relevant_pis = [pi for pi in pi_groups
                        if pi_correlations[str(pi)] > 0.1]
        
        if not relevant_pis:
            relevant_pis = pi_groups  # fallback: use all
        
        # Step 2: Factorization test
        factorizer = ResidualFactorizerV2()
        fact_result = factorizer.test_separability(X, delta)
        
        # Step 3: Strategy selection
        if fact_result.factorization_type in ("multiplicative", "additive"):
            return self._factorized_search(
                fact_result, scenario, X, delta, noise_level
            )
        else:
            return self._product_grammar_search(
                relevant_pis, scenario, X, delta, noise_level
            )
```

**Success criteria:**
- MV-1 Yukawa Mass-Ratio terdeteksi sebagai multiplicative separable
  (m-component: m/M, r-component: exp(-r/r_0))
- MV-3 Turbulent Drag terdeteksi sebagai multiplicative separable
  (v-component: v², rho-component: ρ)

---

### Task MV-6: Scenario Definitions

**File:** `src/adcd/anomaly_scenarios.py` (TAMBAHKAN 4 scenarios)

Tambahkan 4 MV scenarios yang sudah didesain di atas ke `get_all_scenarios()`.
**Jalankan verifikasi script sebelum commit.**

**Success criteria:**
```python
# Semua 4 scenarios pass:
python scripts/verify_mv_scenarios.py
# Output: ✅ MV-1, ✅ MV-2, ✅ MV-3, ✅ MV-4 — θ-dimensionless OK, ARC OK
```

---

### Task MV-7: Benchmark Script

**File:** `run_multivariable_benchmark.py` (REWRITE dari scratch)

```python
"""
ADCD Multivariable Benchmark — Clean version with correct comparisons.

Tests 4 multivariable scenarios with:
- ProductGrammar (new Phase 2 proposer)
- Mock (extended) as baseline

Criteria: class_match AND nmse_residual < 0.1
Reports: per-scenario, per-noise breakdown
"""

SCENARIOS = ["MV-1", "MV-2", "MV-3", "MV-4"]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
SEED = 42

# Run benchmark and save results
# At least 2/4 target at 0% noise (conservative)
# At least 1/4 target at 5% noise
```

**Success criteria (CONSERVATIVE — tidak overpromise):**
- ProductGrammar ≥ 2/4 (50%) at 0% noise
- ProductGrammar ≥ 1/4 (25%) at 5% noise

---

### Task MV-8: Paper Integration

**File:** `paper/main.tex`

Setelah benchmark hasil dikonfirmasi:

```
Section 5.6 (baru): Multivariable Correction Discovery
- Table: 4 MV scenarios, 4 noise levels, ProductGrammar vs Mock
- Highlight: PSAD (Pi-Sparse) sebagai novel contribution
- Honest reporting: if 2/4, report 2/4 with explanation

Section 2 Related Work:
- Tambahkan Buckingham-Pi theorem reference
- Bandingkan dengan PhySO (yang juga pakai unit constraints)

Section 7 Limitations:
- Update: Phase 2 terimplementasi untuk 4 MV scenarios
- New limitation: non-separable corrections require product grammar
- Future: extend to 3+ variable scenarios
```

---

## URUTAN EKSEKUSI DENGAN DEPENDENCIES

```
MV-T0 (tulis semua tests)         → FAIL expected
    ↓
MV-1 (BuckinghamPiEngine)          → pytest MV-T0::TestBuckinghamPiEngine PASS
    ↓
MV-2 (SequentialARCChecker)        → pytest MV-T0::TestSequentialARCChecker PASS
    ↓
MV-3 (ResidualFactorizerV2)        → pytest MV-T0::TestResidualFactorizerV2 PASS
    ↓
MV-6 (Scenario definitions)        → verify_mv_scenarios.py PASS
    ↓
MV-4 (ProductGrammar)              → pytest unit tests PASS
    ↓
MV-5 (MultivariableOrchestrator)   → pytest MV-T0::TestMultivariableEndToEnd PASS
    ↓
MV-7 (Benchmark)                   → run benchmark, capture results
    ↓
MV-8 (Paper update)                → compile PDF, verify claims
```

**CI harus HIJAU setelah setiap task. Jangan lanjut kalau merah.**

---

## ATURAN ABSOLUTE PHASE 2

```
WAJIB:
□ TDD — tulis test DULU, implementasi KEMUDIAN
□ Setiap θ dalam correction_expr adalah dimensionless scalar
□ Sequential ARC: test setiap limit variable INDEPENDEN
□ Verify each scenario dengan script sebelum commit
□ pytest 96+tests passing setelah setiap commit

DILARANG:
□ Mengubah existing physics gates (AST/Dim/ARC) untuk 1D scenarios
□ Menyederhanakan ground truth agar lebih mudah di-solve
□ Commit tanpa pytest hijau
□ Menggabungkan multiple tasks dalam 1 commit
□ Membuat assumptions tentang behavior tanpa membaca code dulu
□ Menyuruh AI agent mengeksekusi lebih dari 1 task sekaligus
```

---

## PRE-CONDITIONS UNTUK MULAI PHASE 2

```
□ Phase 1 + Phase 3: 96/96 tests PASS (✅ SUDAH DONE)
□ Original 9 scenarios: 82.8% CONFIRMED (✅ SUDAH VERIFIED)
□ Paper Phase 1+3 sudah di-submit ke arXiv DULU
□ Baca seluruh plan ini dengan teliti
□ Konfirmasi setiap scenario definition dengan verifikasi script
□ scikit-learn terinstall (untuk ResidualFactorizerV2)
```

---

## EKSPEKTASI REALISTIS

```
Target conservative:
  MV-1 (Yukawa Mass-Ratio): ≥ SUCCESS at 0-1% noise
  MV-2 (Plasma):            ≥ SUCCESS at 0% noise
  MV-3 (Turbulent Drag):    ≥ SUCCESS at 0-5% noise
  MV-4 (Van der Waals):     ≥ SUCCESS at 0% noise
  Overall: 3-4/4 at 0% noise, 2-3/4 at 5% noise

Ini jauh lebih achievable dari Phase 2 sebelumnya (0/4)
karena:
1. Scenarios designed untuk θ-dimensionless dari awal
2. Sequential ARC tidak salah-reject valid candidates
3. ProductGrammar generate ARC-safe forms by construction
4. BuckinghamPi eliminasi perlu dimensional gate bypass
5. TDD memastikan bugs terdeteksi early, bukan di benchmark
```
