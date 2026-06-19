# ADCD Experiment Plan — Rational & Honest

Focused roadmap for **Phase B–D** experiments only.  
Framework/tooling (CLI, PyPI, docs) is out of scope here.

---

## Taxonomy (apply to every run)

| Label | Meaning | Publishable as |
|-------|---------|----------------|
| **SYNTHETIC VALIDATION** | Data from known formula + known coefficients | Method demonstration |
| **REAL VALIDATION** | Real data + known answer (e.g. confirms Simple MOND) | Real-world validation |
| **REAL DISCOVERY** | Real data + form not assumed a priori | Strongest claim |
| **SIMULATED BENCHMARK** | Synthetic data mimicking real distribution | Internal test only — **never** real discovery |

Every script prints its label on stdout and writes it to JSON under `results/`.

---

## Experiment 1 — Muon g-2 QED Series

**File:** `src/adcd/experiments/muon_g2_validation.py`  
**Label:** `SYNTHETIC VALIDATION`  
**Claim:** iADCD recovers perturbative orders from a known QED expansion — not new physics.

### Design (rational)

1. Independent variable: `x = α/π` (not raw α — matches Schwinger expansion).
2. Classical baseline: `a_μ = 0` (Dirac g=2).
3. **Per-round proposer restriction:**
   - Round 1 → templates `{θ₀·x}` only (Schwinger order)
   - Round 2 → templates `{θ₀·x²}` only (Petermann-Sommerfield order)
   - Round k → `{θ₀·x^k}` only
4. Why narrow templates? Tests **iterative subtraction**, not whether mock proposer can accidentally fit `log`/`sin` proxies.

### Success criteria (three tiers)

| Tier | Test | Criterion |
|------|------|-----------|
| **A — Isolated** | Data = C_k x^k only | Each order recovered ± tolerance |
| **B — Residual** | Classical = exact Σ_{i<k} C_i x^i | Target order recovered |
| **C — Integrated** | Full iADCD loop | OLS readout; C2 may fail (correlated monomials) |
| **C+ — OLS projection** | iADCD + `subtraction_mode=ols_projection` | Same limitation as C unless `prior_subtractions` exact |
| **D — Simultaneous** | `np.linalg.lstsq` design matrix | Oracle ceiling — all orders at once |

| Order | Known C_k | Tolerance |
|-------|-----------|-----------|
| 1 | 0.500000 | ±5% |
| 2 | 0.765857 | ±20% |

**Note:** Tier C may fail order-2 when Tier A/B pass — that indicates
cross-order numerical pollution, not proposer failure. Report all three tiers honestly.

## Estimated runtime (Intel i5-class laptop, CPU JAX)

| Step | Command | Time |
|------|---------|------|
| Experiment tests | `pytest tests/test_experiments.py` | ~15–30 s |
| Muon g-2 only | `python -m adcd.experiments.muon_g2_validation` | ~10–15 s |
| SPARC only (real data cached) | `python -m adcd.experiments.sparc_stacking` | ~6–15 s |
| SPARC first download | + network | +10–30 s |
| **Full suite** | `python -m adcd.experiments.run_all` | **~20–40 s** |

Measured on this machine: **18.7 s** total (muon 9.6s + sparc 5.9s real 171 galaxies).

---

## Experiment 2 — SPARC MOND ν(x)

**Files:** `sparc_data.py`, `sparc_stacking.py`  
**Label:** `REAL` or `SIMULATED` (auto-detected, never hidden)

### Design (rational)

1. **Real path:** Download `MassModels_Lelli2016c.mrt` (case.edu mirror).
2. Stack all galaxies:
   - `V_bar = √(V_gas² + V_disk² + V_bulge²)`
   - `g_bar = V_bar² / r` (SI)
   - `x = g_bar / a₀`, `ν = (V_obs/V_bar)²`
3. Classical: `ν = 1` → discover correction `Δν = ν_obs − 1`, ARC: `Δν → 0` as `x → ∞`.
4. **MOND proposer:** sqrt/rational templates only (no trig/log pollution).
5. **Reference baseline:** Simple MOND NMSE reported alongside ADCD NMSE.

### Decision matrix (after run)

```
REAL + ADCD NMSE ≈ Simple MOND NMSE  →  VALIDATION
REAL + ADCD NMSE << Simple MOND       →  potential DISCOVERY (verify carefully)
SIMULATED only                        →  benchmark label, stop discovery claims
REAL + identifiability NOT            →  report as fundamental limit
```

### Run

```powershell
python -m adcd.experiments.sparc_stacking
# Output: results/sparc_discovery.json
```

### Manual data fallback

If astroweb is down, download in browser:  
https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt  
Save to: `data/sparc/MassModels_Lelli2016c.mrt`

---

## Verification gate (before trusting results)

```powershell
python -m pytest tests/test_experiments.py -q
python -m pytest tests/test_iadcd.py -q
python -m adcd.experiments.muon_g2_validation
python -m adcd.experiments.sparc_stacking
```

Check JSON files in `results/` for `data_label` / `data_source`.

---

## What we deliberately do NOT do

- Claim discovery from simulated SPARC fallback
- Use full `CorrectionMockProposer` on perturbative series (log/sin false positives)
- Hide failed coefficient recovery — report `passed: false` in JSON
- Mix experiment code into core `adcd/` pipeline (experiments live in `experiments/`)

---

## Next steps (Week 2–3)

1. Run muon g-2 with `max_order=3` after order-1/2 pass reliably
2. Obtain real SPARC file → confirm `data_source: REAL` and n_galaxies > 100
3. Add BIC comparison table: Simple MOND vs Standard MOND vs ADCD-discovered
4. Wire identifiability analyzer (Phase 3) into SPARC run report
