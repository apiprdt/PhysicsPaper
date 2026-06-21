# Master Plan: ADCD Paper — Complete & Actionable

---

## 1. Identitas Paper

| Field | Value |
|---|---|
| **Judul** | *Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery* |
| **Type** | Methodology paper dengan empirical contributions (BUKAN pure physics discovery paper) |
| **Length** | ~14,000–16,000 kata + appendix (upgrade dari versi current ~10,000 kata) |
| **Target** | arXiv FIRST (cs.LG + astro-ph.IM) → lalu TMLR (rolling, no deadline) atau NeurIPS 2027 / ICLR 2027 |

---

## 2. Perubahan Struktural dari Paper Saat Ini

### KEEP (tidak perlu diubah substansial)

```
├── Section 1: Introduction          ← minor update
├── Section 2: Related Work         ← tambah 2 referensi
├── Section 3: ADCD Framework        ← keep
├── Section 4: Experimental Setup   ← keep
├── Section 5.1–5.5: Synthetic + PySR + Blind  ← keep
├── Section 6: Ablation              ← keep
└── Section 7: Limitations          ← tambah 3 subsection baru
```

### MODIFY (perlu rewriting)

```
├── Abstract          ← rewrite total
├── Section 5.6: SPARC ← tambah algebraic equivalence
└── Section 8: Conclusion ← rewrite
```

### ADD (baru sama sekali)

```
├── Section 5.7: Wide Binary Cross-Validation
├── Section 5.8: Cosmological Probes (fσ₈ + H(z))
└── Section 5.9: Structural Dichotomy Finding
```

---

## 3. Plan Per-Section (Detail Penuh)

---

### ABSTRACT — Rewrite Total

#### Yang harus ada

**Paragraph 1 — Problem & Method:**

> *Traditional SR discovers equations from scratch. We present ADCD, a correction-first framework that...*

**Paragraph 2 — Synthetic benchmark (claim utama):**

> *Mean structural recovery 82.8% (±7.7%) across 5 seeds, 94.4% at reference seed=42 (highest-performing seed — disclosed explicitly). Outperforms PySR by 77.8 pp at 5% noise.*

**Paragraph 3 — Real-world: SPARC (honest framing):**

> *On SPARC (N=3342), ADCD autonomously discovers a 2-parameter member of the Simple MOND interpolating family with transition parameter c≈0.27, achieving 41% NMSE reduction over zero-parameter canonical forms and statistical parity with domain-expert 2-param forms.*

**Paragraph 4 — NEW: Cosmological probes:**

> *Extended to cosmological datasets (63 fσ₈ growth-rate measurements, 34 H(z) cosmic chronometer points), ADCD finds no z-dependent functional correction beyond a constant amplitude rescaling in 5 independent tests — a result that constrains but does not determine the physical origin of S8 tension.*

**Paragraph 5 — Structural dichotomy (dengan caveat):**

> *These results suggest a structural pattern: anomalies in individual gravitational systems exhibit functional corrections recoverable by ADCD, while large-scale cosmological tensions appear as amplitude offsets. Whether this reflects genuine physics dichotomy or current survey precision limits is an open question we explicitly flag.*

#### Yang TIDAK boleh ada di abstract

- Klaim "novel functional form" untuk SPARC
- Klaim "ADCD resolves S8 tension"
- Angka yang tidak didukung bukti

---

### SECTION 1: Introduction — Minor Update

Tambahkan satu paragraf di akhir intro tentang **extended scope**:

> *In this work, we extend ADCD beyond synthetic validation to three classes of real-world anomaly: galactic rotation curves, stellar system kinematics, and large-scale cosmological structure growth. Each class yields qualitatively different outcomes, revealing a structural pattern in how anomalies manifest across physical scales.*

Tidak perlu ubah yang lain.

---

### SECTION 2: Related Work — Tambah 2 Referensi

Tambahkan ke bagian SR:

- Famaey & McGaugh 2012 (Living Reviews) — untuk konteks MOND family
- Alestas et al. 2022 — untuk konteks fσ₈ compilation

---

### SECTION 3 & 4: Framework & Setup — Keep

Tidak ada yang perlu diubah. Solid.

---

### SECTION 5.6: SPARC — Modify dengan Algebraic Disclosure

Ini bagian yang paling penting untuk diupdate. Tambahkan subsection baru.

#### 5.6.1 Algebraic Identification of Discovered Form

Tuliskan:

```
ADCD form:   θ₀(√(1 + θ₁/x) − 1) + 1
           = (1−θ₀) + θ₀·√(1 + θ₁/x)
           = a + b·√(1 + c/x)
             dengan a = −0.83, b = 1.83, c = 0.27, a+b=1

Simple MOND (Famaey & Binney 2005):
             a + b·√(1 + c/x)
             dengan a = 0.5, b = 0.5, c = 4.0

→ Same algebraic family. Different parameters.
→ Key distinction: c = 0.27 vs c = 4.0 (factor of 15)
  Transition scale berbeda drastis.
```

#### 5.6.2 Parameter Degeneracy Analysis

**Tabel:**

| Cut | θ₀ | θ₁ | θ₀·√θ₁ |
|---|---|---|---|
| Baseline | 1.83 | 0.262 | 0.935 |
| Strict | 1.61 | 0.324 | 0.918 |
| Ultra-strict | 1.20 | 0.468 | 0.819 |

**Interpretasi:**

- θ₀√θ₁ ≈ 0.82–0.94 — kira-kira konstan
- Deep-MOND prediction: ν ≈ θ₀√(θ₁/x) ≈ 0.9/√x
- Simple MOND canonical: ν ≈ 1.0/√x
- ~6.5% difference in deep-MOND asymptote

#### 5.6.3 Cross-System Prediction: Wide Binary

Ini Section 5.7 yang jadi extension natural dari 5.6. Lihat di bawah.

#### Framing klaim SPARC yang benar

| ❌ JANGAN | ✅ HARUS |
|---|---|
| "ADCD discovers a novel interpolating function" | "ADCD autonomously converges to a 2-parameter member of the Simple MOND algebraic family (Famaey & Binney 2005), with dramatically different transition parameter (c≈0.27 vs canonical c=4.0). The discovered form achieves 41% NMSE reduction over zero-parameter canonical forms and reaches statistical parity with 2-parameter domain-expert forms in out-of-sample cross-validation." |

---

### SECTION 5.7 (NEW): Wide Binary Cross-Validation

#### Tujuan section ini

Test apakah form ADCD dari SPARC generalize ke sistem independen.

#### Struktur

##### 5.7.1 Setup

```
- System: Wide binary stars dari Gaia DR3
- Classical baseline: Newtonian two-body gravity
- ADCD form dari SPARC: θ₀(√(1+θ₁/x)−1)+1 (NO REFITTING)
- Separation range: 7–30 kAU (Hernandez anomaly zone)
- Data reference: Pittordis & Sutherland 2023, Chae 2023
```

##### 5.7.2 Predicted velocity boost comparison

**Tabel:**

| s (kAU) | x | ADCD | Simple | Standard | Observation (Chae) | σ-distance |
|---|---|---|---|---|---|---|
| 7 | 1.51 | −2.1σ | +0.1σ | −1.8σ | 1.20±0.06 | |
| 10 | 0.74 | −1.0σ | +2.1σ | −0.9σ | ... | |
| 15 | 0.33 | +1.2σ | +5.3σ | +0.5σ | ... | |
| 20 | 0.19 | +3.6σ | +8.3σ | +1.7σ | ... | |
| 30 | 0.08 | +8.4σ | +13.4σ | +3.6σ | ... | |

##### 5.7.3 Interpretasi — harus sangat hati-hati

> *Three findings emerge from this cross-validation:*
>
> 1. *Simple MOND and RAR dramatically overshoot observed velocity boost at s > 15 kAU (5σ–13σ), inconsistent with Chae 2023 measurements.*
>
> 2. *ADCD form (c=0.27) is more consistent with observations at 10–15 kAU than Simple/RAR, but less consistent than Standard MOND at the same separations.*
>
> 3. *ADCD beats Standard MOND at SPARC but loses to it at wide binary — an inconsistency that suggests no single interpolating function simultaneously optimizes both datasets, or that the SPARC-optimized transition parameter (c=0.27) does not generalize.*
>
> *Importantly, the wide binary anomaly itself remains controversial: Banik et al. 2024 attribute the observed boost to selection effects rather than modified gravity. We therefore treat this cross-validation as exploratory rather than definitive.*

**Yang TIDAK boleh diklaim:** Bahwa ADCD "wins" di wide binary.

---

### SECTION 5.8 (NEW): Cosmological Probes

#### 5.8.1 Setup

**Dataset A: Growth rate fσ₈(z)**

```
- 63 titik dari Alestas et al. 2022 compilation
- z ∈ [0.001, 1.944]
- Classical baseline: f(z) = Ωm(z)^0.55 (GR growth index)
- Asymptotic constraint: ∆ → 0 saat z → 0
```

**Dataset B: Cosmic Chronometers H(z)**

```
- 34 titik dari Stern/Zhang/Moresco/Jimenez-Loeb
- z ∈ [0.07, 1.965]
- Classical baseline: H²(z) = H₀²[Ωm(1+z)³ + ΩΛ]
```

#### 5.8.2 Hasil — Tabel CONSTANT_WINS

| Test | Dataset | ΔBIC vs null | Verdict |
|---|---|---|---|
| Full | 63 fσ₈ points | +0.00 | CONSTANT_WINS |
| BOSS | 6 BOSS DR12 pts | −9.63 | CONSTANT_WINS |
| Top-20 | 20 precision pts | +0.00 | CONSTANT_WINS |
| Mid-z | 31 pts z∈[.35,.75] | −0.00 | CONSTANT_WINS |
| CC H(z) | 34 H(z) points | +0.00 | CONSTANT_WINS |

**Key finding:**

- σ₈₀ best-fit = 0.7815 vs Planck 0.811 (−3.7%)
- H₀ best-fit = 68.48 km/s/Mpc, Ωm = 0.296

#### 5.8.3 Interpretasi — caveat WAJIB ada

> *Across five independent tests spanning two observables and three survey subsets, ADCD finds no z-dependent functional correction beyond a constant amplitude rescaling.*
>
> *Two interpretations are consistent with this result:*
>
> *(i) Physics interpretation: S8 tension originates from amplitude renormalization (e.g., σ₈ normalization from early-universe physics) rather than modified growth dynamics — in which case no functional z-correction exists to be discovered.*
>
> *(ii) Sensitivity interpretation: A functional correction exists but falls below ADCD's detection threshold given current survey precision (~20% point errors in fσ₈) and heterogeneous systematics across 15+ survey instruments.*
>
> *The current data cannot distinguish these interpretations. Future surveys (DESI full, Euclid, Rubin LSST) with 3–5× improvement in fσ₈ precision will provide a definitive test.*

---

### SECTION 5.9 (NEW): Structural Dichotomy

Ini section pendek (~400 kata) yang merangkum pattern keseluruhan.

#### Tabel rangkuman

| System | Scale | Result | Anomaly Type |
|---|---|---|---|
| Textbook physics (9 scen) | micro–macro | ✅ Func. | Functional |
| SPARC rotation curves | galactic | ✅ Func. | Functional |
| Wide binary stars | stellar | ~ | Ambiguous |
| Growth rate fσ₈(z) | cosmo | ❌ Const | Amplitude |
| H(z) chronometers | cosmo | ❌ Const | Amplitude |

#### Interpretasi dengan full honesty

> *These results are consistent with — but do not prove — a structural dichotomy between two classes of physical anomaly:*
>
> - **Class I (Functional):** *Anomalies in individual gravitational systems (orbital mechanics, galactic rotation) that require z- or r-dependent corrections to the force law. ADCD reliably discovers these.*
>
> - **Class II (Amplitude):** *Cosmological tensions (S8, H0) that manifest as amplitude offsets with no detected functional structure. ADCD correctly identifies the absence of functional corrections.*
>
> *Whether this reflects genuine physics (different origins for micro- vs macro-scale anomalies) or is a methodological artifact of data quality differences between datasets is an open and testable question. We report this pattern as an observation requiring further investigation, not as a physics conclusion.*

---

### SECTION 7: Limitations — Tambah 3 Subsection

#### 7.x Amplitude vs Functional Distinguishability

> *ADCD cannot detect functional corrections whose amplitude falls below the noise floor of available data. CONSTANT_WINS results are therefore upper bounds on correction detectability, not proofs of correction absence.*

#### 7.x Heterogeneous Survey Compilations

> *fσ₈ compilations mix surveys with different estimators, scale cuts, and bias models. Systematic offsets between surveys could appear as — or mask — functional z-corrections.*

#### 7.x Cross-System Generalization

> *The SPARC-discovered form (c=0.27) does not unambiguously generalize to wide binary kinematics, suggesting transition parameters may be system-dependent. Universal interpolating function discovery from a single system class should be treated as provisional.*

---

### SECTION 8: Conclusion — Rewrite

**Struktur:**

1. **Paragraph 1** — Methodology contribution (tidak berubah)
2. **Paragraph 2** — Synthetic benchmark (tidak berubah)
3. **Paragraph 3** — SPARC: honest reframing
4. **Paragraph 4** — NEW: Wide binary cross-validation finding
5. **Paragraph 5** — NEW: Cosmological probes finding
6. **Paragraph 6** — NEW: Structural dichotomy (dengan caveat)
7. **Paragraph 7** — Future work

---

## 4. Figures Checklist

### EXISTING (keep)

- [x] Figure 1: ADCD architecture diagram
- [x] Figure 2: Noise robustness ADCD vs PySR
- [x] Figure 3: NMSE heatmap
- [x] Figure 4: Performance by tier
- [x] Figure 5: Ablation study
- [x] Figure 6: SPARC stacked relation

### NEW (perlu dibuat)

- [ ] **Figure 7:** Wide binary velocity boost comparison (ADCD vs Simple MOND vs Standard MOND vs Chae data)
- [ ] **Figure 8:** fσ₈ ADCD fit vs GR baseline + constant correction (tunjukkan bahwa constant = winner)
- [ ] **Figure 9:** Structural dichotomy summary figure (2×2 grid: galactic functional vs cosmological amplitude)

---

## 5. Tables Checklist

### EXISTING (keep, sudah solid)

- [x] Table 1–3: Tier benchmark results
- [x] Table 4: PySR configuration
- [x] Table 5: ADCD vs PySR comparison
- [x] Table 6: Search space funnel
- [x] Table 7: Compute budget
- [x] Table 8: Real-world recovery
- [x] Table 9–11: Template leakage, binary pulsar, misspecification
- [x] Table 12–17: SPARC statistical tables

### NEW

- [ ] **Table X:** Wide binary σ-distance table (s, x, ADCD, Simple, Standard, Obs)
- [ ] **Table Y:** CONSTANT_WINS across 5 cosmological tests
- [ ] **Table Z:** Structural dichotomy summary

---

## 6. Klaim yang BOLEH dan TIDAK BOLEH Dibuat

### ✅ BOLEH (didukung bukti)

- "ADCD achieves 82.8% (±7.7%) mean structural recovery"
- "Outperforms PySR by 77.8 pp at 5% noise"
- "Discovers 2-parameter member of Simple MOND family from SPARC"
- "41% NMSE reduction over zero-parameter canonical MOND forms"
- "Statistical parity with 2-parameter domain-expert forms"
- "No z-dependent functional correction detected in fσ₈/H(z)"
- "Results consistent with structural dichotomy between functional and amplitude anomaly classes"

### ❌ TIDAK BOLEH

- "Novel functional form discovered from SPARC"
- "ADCD resolves dark matter problem"
- "S8 tension proved to originate from early universe"
- "ADCD wins at wide binary"
- "Structural dichotomy is proven" (hanya "consistent with")

---

## 7. Timeline Realistis

### Week 1 (sekarang)

```
├── Hari 1–2: Tulis Section 5.7 (wide binary)
├── Hari 3–4: Tulis Section 5.8 (cosmological probes)
├── Hari 5:   Tulis Section 5.9 (structural dichotomy)
└── Hari 6–7: Rewrite abstract + conclusion
```

### Week 2

```
├── Hari 1–3: Buat Figure 7, 8, 9
├── Hari 4–5: Buat Table X, Y, Z
└── Hari 6–7: Revisi Section 7 (limitations) + polish
```

### Week 3

```
├── Hari 1–3: Full paper read-through
├── Hari 4:   Internal consistency check (semua klaim dicek vs bukti)
├── Hari 5:   Cek referensi lengkap
└── Hari 6–7: Format untuk arXiv
```

### Week 4

```
└── Submit ke arXiv → announce di Twitter/X/social media
    → Pilih venue: TMLR (rolling) atau target conference
```

---

## 8. Antisipasi Reviewer Attacks

| # | Attack | Defense |
|---|---|---|
| 1 | "SPARC form is just Simple MOND" | "We explicitly identify and disclose the algebraic family equivalence in Section 5.6.1. The distinction is the transition parameter c=0.27 vs c=4.0. We claim autonomous rediscovery of this family, not a novel family." |
| 2 | "CONSTANT_WINS might just mean data is too noisy" | "We explicitly acknowledge this in Sections 5.8.3 and 7.x. Two interpretations are consistent with CONSTANT_WINS; we report both and do not adjudicate." |
| 3 | "Wide binary test is inconclusive" | "Agreed. We present it as exploratory cross-validation, not definitive validation. The Chae vs Banik controversy is cited explicitly." |
| 4 | "Seed=42 is cherry-picked" | "Disclosed explicitly: seed=42 is highest-performing in our 5-seed set. Primary claim is mean 82.8%, not peak." |
| 5 | "Why compare to PySR not PhySO?" | "Addressed in Section 7: PhySO is tabula rasa, ADCD is correction-first. Different problem formulation, not directly comparable. Left for future work." |

---

## 9. Keputusan yang Perlu Ditentukan

Sebelum mulai nulis, ada satu keputusan yang harus ditentukan sendiri:

### Apakah wide binary section masuk sebagai "full result" atau "preliminary exploration"?

| Option | Pro | Con |
|---|---|---|
| **A — Full result** | Menambah kekuatan empiris paper | Data Chae 2023 masih controversial (Banik 2024 challenge) → reviewer bisa attack langsung |
| **B — Appendix/supplementary** *(rekomendasi)* | Melindungi main claims dari controversy | Kehilangan satu kontribusi yang menarik |

**Rekomendasi: Option B** — taruh di appendix sebagai "Extended Cross-Validation Study" dengan disclaimer eksplisit tentang ongoing controversy di wide binary data. Main paper fokus pada yang solid: synthetic, SPARC, cosmological probes.

> Keputusan ini akan mempengaruhi struktur keseluruhan. Setelah diputuskan, mulai dari Section 5.8 (cosmological probes) karena itu yang paling clean dan paling kuat sebagai added contribution.

---

## Notes

> *Dokumen ini adalah living document yang akan diupdate seiring progres penulisan. Setiap section yang sudah selesai ditulis akan ditandai dengan tanda ✅ di checklist.*
