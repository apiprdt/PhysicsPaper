# ADCD Master Execution Plan
## From Framework to Genuine Discovery — Complete Roadmap

---

## PERINGATAN SEBELUM EKSEKUSI

**Baca ini dulu sebelum menjalankan apapun:**

### Red Flag 1: Test Count Drop (26 vs 105)

Dokumen Antigravity menyebut "Expected: 26 passed" tapi sebelumnya 105 tests.
Ini harus diverifikasi SEGERA:

```powershell
& "C:\Program Files\Python311\python.exe" -m pytest --co -q | Measure-Object -Line
```

Jika < 80 tests: ada tests yang hilang atau dihapus. STOP dan investigasi
sebelum lanjut apapun.

### Red Flag 2: SPARC "Simulated Fallback"

Dokumen menyebut: "Falls back to robust simulated SPARC distribution if
remote Astroweb server is unreachable."

Ini masalah serius:
```
SPARC real data   + discovery = genuinely meaningful
SPARC simulated   + discovery = circular (generate MOND-like, then find MOND)
```

Jika SPARC experiment berjalan di simulated data:
- "Discovery" tidak genuine
- Tidak bisa diklaim sebagai real-world validation
- Reviewer AKAN menolak kalau ini tidak dijelaskan dengan sangat jelas

Verifikasi dulu:
```powershell
& "C:\Program Files\Python311\python.exe" -c "
import urllib.request
try:
    urllib.request.urlopen('http://astroweb.cwru.edu/SPARC/', timeout=10)
    print('REAL DATA: Server accessible')
except:
    print('SIMULATED: Server unreachable, will use synthetic fallback')
"
```

### Red Flag 3: Muon g-2 Framing

Expected NMSE < 10⁻⁶ menggunakan synthetic data dari koefisien yang SUDAH
DIKETAHUI adalah demonstration, bukan discovery. Paper HARUS menyebutkan
ini dengan eksplisit atau reviewer akan reject.

---

## VERIFICATION GATE (WAJIB, SEBELUM SEMUA TASK)

```powershell
# Step V1: Full test count
& "C:\Program Files\Python311\python.exe" -m pytest -v 2>&1 | tail -5
# Target: semua tests pass, ideally 100+

# Step V2: SPARC connectivity
& "C:\Program Files\Python311\python.exe" -c "
import urllib.request
try:
    urllib.request.urlopen('http://astroweb.cwru.edu/SPARC/', timeout=15)
    print('PASS: Real SPARC accessible')
except Exception as e:
    print(f'FAIL: {e}')
    print('Action needed: download SPARC data manually')
"

# Step V3: Verify iADCD exists dan tidak circular
& "C:\Program Files\Python311\python.exe" -c "
from adcd.correction_orchestrator import CorrectionOrchestrator
import inspect
src = inspect.getsource(CorrectionOrchestrator)
has_iterative = 'iterative' in src.lower() or 'round' in src.lower()
print('iADCD in orchestrator:', has_iterative)
"

# Step V4: Verify SPARC uses real data path
& "C:\Program Files\Python311\python.exe" -c "
import inspect
from adcd.experiments import sparc_stacking
src = inspect.getsource(sparc_stacking)
print('Uses real URL:', 'astroweb.cwru.edu' in src)
print('Has fallback:', 'simulat' in src.lower() or 'fallback' in src.lower())
"
```

Report hasil V1-V4 sebelum melanjutkan ke task apapun.

---

## TAXONOMY PENTING: Discovery vs Demonstration vs Validation

Sebelum menulis satu baris kode atau klaim di paper, pahami perbedaan ini:

```
DISCOVERY (highest impact):
  Real experimental data + open scientific question
  ADCD finds form not previously established
  Example: SPARC real data → ν(x) form more precise than known MOND variants
  Example: unexplained particle physics cross-section → functional correction

VALIDATION (medium impact):
  Real data + known answer
  ADCD confirms form that theory predicts
  Example: SPARC real data → confirms Simple MOND ν(x) form
  Example: Mercury perihelion real data → confirms GR v²/c² correction

DEMONSTRATION (low impact, useful for benchmarking):
  Synthetic data generated from known formula → ADCD recovers formula
  Example: Muon g-2 with synthetic QED data → recovers C₁, C₂
  Example: SPARC with simulated MOND distribution → recovers MOND form

CIRCULAR (not publishable as discovery):
  Generate data from Formula A → "discover" Formula A
  Must be labeled as "synthetic benchmark" not "real-world application"
```

**Aturan paper:**
- DISCOVERY/VALIDATION: Bisa klaim sebagai real-world result
- DEMONSTRATION: Harus explicit label "synthetic validation"
- CIRCULAR: Tidak boleh diklaim sebagai discovery apapun

---

## PHASE 0: SUBMIT ARXIV (WEEK 1 — NON-NEGOTIABLE)

**Tidak ada yang lain sampai ini selesai.**

### Task 0.1: Final Paper Audit (2-3 jam)

Baca paper dari halaman 1 sampai akhir:

```
Check setiap claim:
□ Abstract: 82.8% (±7.7%) — verified ✅
□ Abstract: 94.4% at seed=42 — verified ✅
□ Section 5.3 blind test: apakah angka match dengan blind_benchmark_results.json?
□ Section 5.4 real-world: Mercury, Lamb shift, Muon, Blackbody — angka benar?
□ Section 5.6 Phase 2: 2/4 ProductGrammar, 3/4 Mock — angka benar?
□ Semua figure: apakah figure di PDF match description di caption?
□ Semua table: cross-check dengan JSON files di repo
□ Bibliography: apakah semua citations resolve?
□ Affiliation: "Independent Researcher, Indonesia" (BUKAN SMA!)
```

### Task 0.2: Code Availability Section

Tambahkan di paper sebelum Acknowledgments:

```latex
\section*{Code and Data Availability}
The ADCD framework is available as an open-source Python package at
\url{https://github.com/apiprdt/PhysicsPaper} and on PyPI as \texttt{adcd}.
All benchmark results are reproducible via the scripts in the repository.
Benchmark result files (\texttt{reproducibility\_results.json},
\texttt{blind\_benchmark\_results.json}) are included in the repository.
```

### Task 0.3: arXiv Bundle

```powershell
# Compile final PDF (3 passes untuk stable references)
cd "e:\Physics Project\paper"
& pdflatex main.tex
& bibtex main
& pdflatex main.tex
& pdflatex main.tex

# Verify PDF size dan date
Get-Item main.pdf | Select-Object Length, LastWriteTime
```

**arXiv submission details:**
- Primary: `cs.LG`
- Cross-list: `physics.comp-ph`, `physics.data-an`
- Title: "Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
  Symbolic Regression for Evolutionary Scientific Discovery"
- Comments: "Code: github.com/apiprdt/PhysicsPaper. 22 pages, 6 figures"

### Task 0.4: Git Tag

```bash
git tag -a v2.2.0 -m "ADCD v2.2.0: arXiv submission, Phase 1+2+3 complete"
git push origin v2.2.0
```

---

## PHASE B: iADCD — ITERATIVE CORRECTION DISCOVERY

**Ini adalah genuine algorithmic contribution baru. Tidak ada SR framework
yang melakukan multi-order iterative correction discovery.**

### Scientific Narrative yang Benar

```
iADCD Story (yang bisa diklaim):

Round 1: Classical baseline = 0
         Target = full anomaly signal
         Discover: Δ₁(x) = leading-order correction

Round 2: Classical baseline = Δ₁(x)
         Target = residual setelah subtract Δ₁
         Discover: Δ₂(x) = next-order correction

Round n: Repeat sampai NMSE < threshold atau SNR < 1

Historical parallel:
  Einstein 1905:  Δ₁ = (3/4)v²/c²  (leading relativistic correction)
  Dirac 1928:     Δ₂ = higher-order QED terms
  Schwinger 1948: Δ₃ = α/2π (anomalous magnetic moment)

ADCD automates this iterative refinement process.
```

### Task B.1: Verify iADCD Implementation

```python
# Test: does iADCD actually iterate?
# File: tests/test_iadcd_iteration.py

def test_iadcd_converges_faster_than_single_pass():
    """iADCD using 2 rounds should achieve lower NMSE than 1 round."""
    scenario = get_scenario("Relativistic KE")
    
    # Single-pass ADCD
    result_1 = run_adcd(scenario, noise=0.01, max_rounds=1)
    
    # Two-pass iADCD
    result_2 = run_adcd(scenario, noise=0.01, max_rounds=2)
    
    assert result_2.final_nmse <= result_1.final_nmse

def test_iadcd_series_structure():
    """iADCD on relativistic KE should find v²/c² then v⁴/c⁴."""
    scenario = get_scenario("Relativistic KE")
    result = run_adcd(scenario, noise=0.0, max_rounds=3)
    
    assert len(result.correction_series) >= 2
    assert "v**2" in result.correction_series[0] or "(v/c)**2" in result.correction_series[0]
```

### Task B.2: Muon g-2 Iterative Validation

**Label yang benar untuk paper:** "Synthetic Validation of iADCD
on the Known QED Perturbative Series"

```python
# src/adcd/experiments/muon_g2_validation.py

"""
Synthetic validation: iADCD recovers known QED perturbative series.

IMPORTANT: This uses SYNTHETIC data generated from known coefficients.
This is a VALIDATION experiment, not a discovery.
The scientific claim is: "iADCD can iteratively recover perturbative
corrections in the same order they were historically discovered."

NOT claimed: "iADCD discovers new physics from experimental data."
"""

QED_COEFFICIENTS = {
    1: 0.5,                # Schwinger (1948) — α/(2π)
    2: 0.765857425,        # Petermann-Sommerfield (1957)
    3: 24.05050964,        # Laporta-Remiddi (1996)
}

def validate_iadcd_on_qed(max_order=3, noise_level=0.0, seed=42):
    """
    Run iADCD to recover QED coefficients from synthetic data.
    
    Success criteria:
    - Round 1: |C₁_discovered - 0.5| / 0.5 < 5% (Schwinger term)
    - Round 2: C₂ structural class = polynomial, error < 20%
    - Final NMSE < 1e-4
    
    Returns:
        IterativeResult with correction_series, recovered_coefficients,
        and comparison_to_literature
    """
    ...

def print_validation_report(result):
    """
    Print honest comparison: discovered vs known coefficients.
    Format suitable for paper table.
    """
    print("=== iADCD QED VALIDATION (SYNTHETIC DATA) ===")
    print(f"{'Order':>5} | {'Discovered':>15} | {'Known':>15} | {'Error %':>8}")
    for i, (discovered, known) in enumerate(result.comparison, 1):
        error = abs(discovered - known) / abs(known) * 100
        print(f"{i:>5} | {discovered:>15.6f} | {known:>15.6f} | {error:>7.1f}%")
```

**Expected results untuk paper:**

| Order | Known Coeff | Expected ADCD Recovery | Tolerance |
|-------|-------------|------------------------|-----------|
| 1 (Schwinger) | 0.500000 | 0.499-0.501 | ±1% |
| 2 (Petermann) | 0.765857 | 0.73-0.80 | ±5% |
| 3 (Laporta) | 24.050510 | 20-28 | ±20% |

Jika error lebih besar dari tolerance: jangan hide ini, report honestly.

### Task B.3: Verify Muon g-2 Results

```powershell
& "C:\Program Files\Python311\python.exe" -m adcd.experiments.muon_g2_validation 2>&1 | Tee-Object -FilePath muon_g2_results.txt
cat muon_g2_results.txt
```

Check:
- [ ] Label "SYNTHETIC DATA" muncul di output
- [ ] Recovery error untuk C₁ < 5%
- [ ] Recovery error untuk C₂ < 20%
- [ ] Final NMSE < 1e-4 (bukan < 1e-6 yang terlalu kecil untuk noisy data)

---

## PHASE C: SPARC REAL DATA PIPELINE

**Ini adalah potential genuine discovery. Perlu perhatian paling besar.**

### C.1: Ensure Real Data Pipeline

Sebelum run experiment, pastikan real data digunakan:

```powershell
# Option 1: Test connectivity
& "C:\Program Files\Python311\python.exe" -c "
import urllib.request
try:
    url = 'http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt'
    urllib.request.urlretrieve(url, 'data/sparc/SPARC_real.mrt')
    print('SUCCESS: Real SPARC data downloaded')
except Exception as e:
    print(f'FAILED: {e}')
"

# Option 2: Jika server tidak accessible dari network ini,
# download via browser: http://astroweb.cwru.edu/SPARC/
# Save ke: e:/Physics Project/data/sparc/SPARC_Lelli2016c.mrt
```

**ATURAN PENTING:** Jika experiment harus menggunakan simulated fallback,
label output dengan "SIMULATED DATA" dan JANGAN claim sebagai real discovery.

### C.2: SPARC Stacking Approach (Kunci untuk Tractability)

```python
# src/adcd/experiments/sparc_stacking.py

"""
MOND discovery from SPARC galaxy rotation curves.

Key insight: Instead of fitting each galaxy separately,
stack all galaxies into dimensionless coordinates:
  x = g_bar / g_dagger  (dimensionless baryonic acceleration)
  ν = (V_obs / V_bar)²  (dimensionless velocity ratio squared)

This converts 175 separate datasets into ONE unified ADCD scenario
with ~5000-10000 data points and massive statistical power.

The classical baseline: ν = 1 (Newtonian limit at high x)
The correction: Δν(x) = ν_obs(x) - 1

ARC limit: x → ∞ (high baryonic acceleration)
  At x → ∞: ν → 1, so Δν → 0 ✓ (Newtonian limit recovered)

If ADCD discovers ν(x) ≈ Simple MOND: ν(x) = (1 + √(1+4/x))/2
  → This VALIDATES Simple MOND

If ADCD discovers a different form:
  → This POTENTIALLY contributes to MOND literature
"""

G_DAGGER = 1.2e-10  # Milgrom's constant [m/s²]

def stack_sparc_galaxies(sparc_data, quality_threshold=3,
                          min_points_per_galaxy=10):
    """
    Stack all SPARC galaxies into dimensionless (x, ν) space.
    
    Args:
        sparc_data: parsed SPARC data
        quality_threshold: minimum data quality flag (1=best, 3=acceptable)
        min_points_per_galaxy: exclude galaxies with fewer points
    
    Returns:
        (x_all, nu_all, uncertainties) — stacked dimensionless arrays
    """
    x_all, nu_all, sigma_all = [], [], []
    
    for galaxy in sparc_data:
        if galaxy.quality > quality_threshold:
            continue
        if len(galaxy.r) < min_points_per_galaxy:
            continue
        
        # Compute dimensionless quantities
        g_bar = galaxy.v_bar**2 / galaxy.r  # baryonic acceleration
        x = g_bar / G_DAGGER
        nu = (galaxy.v_obs / galaxy.v_bar)**2
        
        # Only include Newtonian-dominated regime clearly (for ARC limit)
        valid = (galaxy.v_bar > 0) & np.isfinite(nu) & (nu > 0) & (nu < 10)
        
        x_all.extend(x[valid])
        nu_all.extend(nu[valid])
        
        # Propagate velocity uncertainties to ν
        sigma_nu = 2 * galaxy.v_obs[valid] * galaxy.e_v_obs[valid] / galaxy.v_bar[valid]**2
        sigma_all.extend(sigma_nu)
    
    return np.array(x_all), np.array(nu_all), np.array(sigma_all)


def run_mond_discovery(seed=42, noise_override=None):
    """
    Run ADCD to discover MOND interpolating function ν(x).
    
    Scenario setup:
        Classical: ν = 1 (Newtonian prediction)
        Observed: ν_obs from stacked SPARC data
        Target: correction Δ = ν_obs - 1
        ARC: Δ → 0 as x → ∞
    
    Returns:
        SparcResult with:
          - discovered_nu_form: symbolic expression
          - comparison_to_mond_models: dict
          - bayesian_posterior: over MOND functional families
          - identifiability_report: is the form uniquely determined?
    """
    ...
```

### C.3: MOND Model Comparison Table

**Setelah SPARC discovery run, bandingkan dengan known MOND variants:**

| MOND Model | ν(x) formula | Expected BIC | ADCD rank |
|------------|--------------|--------------|-----------|
| Simple MOND | `(1 + √(1+4/x))/2` | ? | ? |
| Standard MOND | `1/√(1-exp(-√x))` | ? | ? |
| RAR (McGaugh) | fit ke data | ? | ? |
| ADCD-discovered | (to be filled) | ? | ? |

Tabel ini yang akan jadi centerpiece result di follow-up paper.

### C.4: SPARC Result Verification

```powershell
& "C:\Program Files\Python311\python.exe" -m adcd.experiments.sparc_stacking 2>&1 | Tee-Object -FilePath sparc_results.txt
cat sparc_results.txt
```

Check:
- [ ] Output menyebut "REAL DATA" atau "SIMULATED DATA" dengan jelas
- [ ] Jumlah galaxies yang digunakan: berapa? (target: >100 dari 175)
- [ ] Jumlah data points yang distacked: berapa? (target: >3000)
- [ ] Discovered ν(x) expression: apa bentuknya?
- [ ] BIC comparison: Simple MOND vs Standard MOND vs Discovered
- [ ] Identifiability report: IDENTIFIABLE atau NOT?
- [ ] Plot tersimpan: sparc_discovery_plot.png

---

## PHASE D: SCIENTIFIC ASSESSMENT

**Setelah Muon g-2 dan SPARC selesai, evaluate hasil secara jujur.**

### Decision Matrix

```
SPARC REAL DATA + discovery form ≠ known MOND variants:
  → STRONGEST result. Follow-up paper ke Nature Physics atau similar.
  → Title: "Data-driven discovery of MOND interpolating function
    from 175 galaxy rotation curves"

SPARC REAL DATA + discovered form ≈ Simple/Standard MOND:
  → VALIDATION result. Still publishable.
  → Title: "Physics-constrained symbolic regression independently
    validates MOND from rotation curve observations"
  → Shows ADCD can identify physically correct form without prior

SPARC REAL DATA + ADCD fails (identifiability: NOT IDENTIFIABLE):
  → HONEST result. Report as limitation.
  → "Rotation curve data is insufficient to identify ν(x) form
    at current noise levels (identifiability: low SNR)"
  → Still publishable as methods paper showing where AI hits fundamental limits

SPARC SIMULATED ONLY:
  → Cannot claim real-world discovery
  → Must solve network access issue first
  → Or find alternative real dataset
```

### Alternative Real Datasets (Jika SPARC Tidak Accessible)

```
Option 2: RAR (Radial Acceleration Relation) public data
  McGaugh, Lelli & Schombert 2016 data: available at
  https://astroweb.cwru.edu/SPARC/
  (Same server, different file format)

Option 3: Open Exoplanet Archive transit timing variations
  Known classical: Kepler's third law
  Anomaly: transit timing variations from planet-planet interactions
  Classical data available at: exoplanetarchive.ipac.caltech.edu

Option 4: NIST Atomic spectra database
  Known classical: Bohr model
  Anomaly: measured quantum defects (real QED corrections)
  Data available at: physics.nist.gov/asd
  This would be REAL QED data, not synthetic!
```

---

## PHASE E: PROFESSIONAL TOOL (AFTER SCIENCE IS DONE)

**Urutan ini penting: science drives tool development, not vice versa.**

### E.1: CLI Verification

```powershell
# Test bahwa CLI tidak crash
& "C:\Program Files\Python311\python.exe" src/adcd/cli.py --help
& "C:\Program Files\Python311\python.exe" src/adcd/cli.py --version
& "C:\Program Files\Python311\python.exe" src/adcd/cli.py list-scenarios
```

Jika CLI error: fix dulu sebelum dokumentasi.

### E.2: I/O Module Verification

```powershell
# Test CSV ingestion dengan synthetic data
& "C:\Program Files\Python311\python.exe" -c "
import numpy as np
import csv

# Generate synthetic test CSV
data = {'x': np.linspace(0.1, 10, 50), 'y_obs': np.linspace(0.1,10,50)**2}
with open('test_ingestion.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['x', 'y_obs', 'y_classical'])
    writer.writeheader()
    for i in range(50):
        writer.writerow({'x': data['x'][i],
                         'y_obs': data['y_obs'][i],
                         'y_classical': data['x'][i]})

# Test load
from adcd.io import load_csv
result = load_csv('test_ingestion.csv', y_obs_col='y_obs', y_classical_col='y_classical')
print('I/O test:', 'PASS' if result is not None else 'FAIL')
"
```

### E.3: Version Bump ke 2.3.0

**Alasan tidak 3.0.0:**
- Tidak ada breaking API change di core correction discovery
- CLI rename bisa backward-compatible dengan deprecation warning
- Belum ada citations untuk dijaga

```python
# pyproject.toml
version = "2.3.0"

# src/adcd/__init__.py
__version__ = "2.3.0"
```

### E.4: Documentation

```powershell
# Install mkdocs
& "C:\Program Files\Python311\python.exe" -m pip install mkdocs-material

# Build docs locally untuk verify
& "C:\Program Files\Python311\python.exe" -m mkdocs build

# Jika build sukses, deploy
& "C:\Program Files\Python311\python.exe" -m mkdocs gh-deploy --force
```

---

## PHASE F: FOLLOW-UP PAPER (WEEK 6-8)

**Ini yang menentukan scientific impact, bukan framework paper.**

### Title Options (berdasarkan hasil actual)

```
Jika SPARC genuine discovery:
"iADCD: Iterative Anomaly-Driven Correction Discovery —
 Rediscovering QED Perturbative Series and Learning the
 MOND Interpolating Function from Galaxy Rotation Curves"

Jika SPARC validation only:
"Physics-Constrained Iterative Symbolic Regression
 Independently Validates MOND from 175 Galaxy Rotation Curves
 Without Prior Model Assumptions"

Jika SPARC fails (honest):
"Iterative Anomaly-Driven Correction Discovery:
 Capabilities and Fundamental Limits in Real Physics Applications"
```

### Structure Aplikasi Paper

```
1. Abstract (discovery/validation/limits — honest framing)

2. Introduction
   - Problem: multi-order corrections in physics
   - Historical parallel: perturbative expansion in QED
   - iADCD as automated perturbative discovery

3. Method: iADCD
   - Algorithm: iterate ADCD, subtract, repeat
   - Stop criterion: NMSE < ε OR SNR < 1
   - Bayesian confidence at each round (Phase 3 langsung berguna!)

4. Validation: Muon g-2 QED Series (clearly labeled SYNTHETIC)
   - Table: recovered vs known coefficients
   - Shows iADCD works iteratively

5. Application: SPARC Galaxy Rotation Curves (REAL DATA)
   - Data: 175 galaxies, stacked to (x, ν) space
   - Discovery/Validation of ν(x) form
   - Bayesian model comparison: Simple MOND vs Standard MOND vs Discovered
   - Identifiability analysis

6. Discussion
   - Where iADCD works (clean signal, 1D correction)
   - Where it fails (high noise, coupled multi-variable)
   - Information-theoretic limits (Identifiability analyzer — Phase 3 langsung relevant!)

7. Conclusion
```

### Target Venue

```
NeurIPS AI4Science Workshop 2026:
  Deadline: Biasanya Juli-Agustus 2026
  URL: ai4sciencecommunity.github.io
  Cocok untuk: methods paper dengan physics application

ICLR 2027:
  Deadline: Oktober 2026
  Lebih prestisius, butuh hasil lebih kuat

Journal of Machine Learning Research (JMLR):
  Open access, no deadline
  Cocok untuk: comprehensive methods + results paper

arXiv preprint sekarang → Submit ke venue setelah experiment selesai
```

---

## COMPLETE TIMELINE

```
WEEK 1 (SEKARANG):
━━━━━━━━━━━━━━━━━
□ Verification Gate (V1-V4)
□ Fix test count jika < 80
□ Final paper audit
□ Fix affiliation → "Independent Researcher, Indonesia"
□ Compile PDF final
□ SUBMIT ARXIV
□ Git tag v2.2.0

WEEK 2:
━━━━━━
□ Verify SPARC real data accessible
□ Jika tidak: download manual atau gunakan alternative (NIST ASD)
□ Verify iADCD iteration genuinely works (test B.1)
□ Run Muon g-2 validation → save results ke JSON

WEEK 3:
━━━━━━
□ Run SPARC stacking experiment (real data)
□ Scientific assessment (decision matrix)
□ Generate publication figures (Muon g-2 + SPARC)

WEEK 4:
━━━━━━
□ CLI verification dan polish
□ I/O module testing
□ MkDocs documentation deploy

WEEK 5:
━━━━━━
□ Start drafting application paper
□ Version bump 2.3.0 + PyPI

WEEK 6-8:
━━━━━━━━
□ Complete application paper
□ Submit to target venue
□ Community announcement
```

---

## DEFINISI SUKSES

```
Minimum success (paper yang defensible):
□ arXiv submitted dengan correct affiliation
□ iADCD validated pada Muon g-2 synthetic dengan error < 20% per order
□ SPARC experiment run (real atau simulated, CLEARLY LABELED)
□ 105+ tests passing
□ version 2.3.0 on PyPI

Medium success (strong follow-up paper):
□ SPARC dengan REAL data
□ ν(x) form discovered dan compared vs Simple/Standard MOND
□ Bayesian posterior menunjukkan decisive evidence untuk salah satu form
□ Application paper submitted ke workshop

Maximum success (nature-level impact):
□ SPARC discovery: form yang ditemukan ADCD ≠ Simple/Standard MOND
□ Form baru memberikan better fit secara statistik
□ Prediksi untuk galaxy yang belum dilihat
□ Peer validation dan citation
```

---

## ATURAN FINAL

```
TIDAK BOLEH:
□ Claim "discovery" dari synthetic/simulated data
□ Submit paper dengan affiliation "SMA Negeri 23"
□ Run SPARC experiment dan call it "real" kalau server unreachable
□ Menambahkan Phase baru ke framework sebelum follow-up paper selesai

WAJIB:
□ Label setiap experiment: "REAL DATA" atau "SYNTHETIC VALIDATION"
□ Verification Gate sebelum eksekusi apapun
□ Test count ≥ previous count sebelum commit baru
□ Every claim di paper harus ada angka yang bisa di-verify dari JSON files
```
