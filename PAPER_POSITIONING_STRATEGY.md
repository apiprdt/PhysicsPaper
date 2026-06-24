# ADCD — Paper Positioning & Multi-Venue Strategy

> Target: **D (combination)** — arXiv → NeurIPS workshop (Sept 2026) → extended journal (2027).
> This document distills a deep read of `paper/main.tex` (1536 lines) into actionable positioning.

---

## 1. The one-sentence pitch (memorize this)

**ADCD is the first symbolic-regression framework that discovers *corrections* to known physical laws rather than laws from scratch — screening candidates with physics gates before any numerical fitting, and the first to systematically classify anomalies as *functional* vs *amplitude*.**

If a reviewer remembers nothing else, they remember this.

---

## 2. Core contributions — in reviewer-proof language

| # | Contribution | Defensible? | Evidence in paper |
|---|---|---|---|
| **C1** | **Correction-first paradigm** — search restricted subspace $\mathcal{D} \subset \mathcal{F}$ of valid corrections, not full function space | ✅ Novel framing | §Intro, §Related Work (vs PhySO/PySR/DSR) |
| **C2** | **Physics-gated cascade before optimization** — AST + dimensional homogeneity + ARC asymptotic gates filter candidates *before* L-BFGS-B | ✅ Empirically validated | §3.3, ablation Fig.4 (dimensional gate = −11.1pp; all-gates-off = −44.5pp) |
| **C3** | **Structural dichotomy finding** — first SR method to distinguish functional vs amplitude anomalies across physical scales | ✅ Genuine novelty, well-hedged | §10 (SPARC functional; $f\sigma_8$/$H(z)$ amplitude) |
| **C4** | **Categorical immunity to trig-overfitting** — physics gates reject nested $\sin(\sin(\dots))$ that dominate PySR's Pareto front at 0% noise | ✅ Mechanistic argument | §Search Space Efficiency |
| **C5** | **EFT-philosophy automation** — formal correspondence between gates and EFT operator enumeration / power counting | ✅ Conceptual (honestly disclaimed as not full EFT) | §3 "Relation to EFT" |

**C1 + C2** = the methodological core (workshop paper).
**C3** = the scientific-finding angle (journal paper).
**C4 + C5** = the theoretical-depth angle (strong reviewer rebuttal material).

---

## 3. Differentiation matrix — your sharpest weapons

This is what reviewers will probe. Know these cold.

| Competitor | What they do | ADCD's sharp difference |
|---|---|---|
| **PySR** (Cranmer) | Tabula rasa genetic SR | Operates on **correction subspace**, not full $\mathcal{F}$. At 5% noise, ADCD's **multi-seed mean (77.1%)** — and even its **worst of 16 seeds (66.7%, 6/9)** — beats PySR fair (11.1%, 1/9). Doubling PySR's budget (`generous`, 55.6%) still does not reach ADCD's worst seed. The 88.9% figure is seed=42 (the peak), **not** the headline. PySR non-monotonic under noise; ADCD stable. |
| **PhySO / Φ-SO** (Tenachi) | Tabula rasa **unit-constrained** SR | PhySO discovers *full* equations; ADCD searches only *corrections*. PhySO = strict unit enforcement; ADCD = **adaptive parameter relaxation** + ARC + complexity gates. Complementary, not competing — this is your email framing to Tenachi. |
| **DSR** (Petersen) | RL over symbolic tree | Same tabula rasa limitation; ADCD uses prior knowledge of classical law to shrink hypothesis space exponentially. |
| **LASR** (Grayeli/NeurIPS'24) | LLM-built concept library for SR | LASR improves *hypothesis generation* but does **no physics filtering** before fit. ADCD subjects LLM templates to the same gate cascade — **directly combinable**. |
| **PINN / HNN / SINDy** | Embed physics in *model/training* | ADCD embeds physics in *search-space constraints*, screening before optimization. Orthogonal axis. |

**The key tactical line for reviewers:** *"ADCD does not compete with PhySO or PySR — it restricts the problem they solve to a physically-motivated subspace where they are known to fail under noise."*

---

## 4. Venue-by-venue angle (your roadmap to Target D)

### Track 1 — NeurIPS 2026 Workshop (Sept deadline, ~4 pages)
**Best fits:** ML4PhysicalSciences (ml4physicalsciences.github.io) or AI4Science (ai4sciencecommunity.github.io)

**Angle to lead with:** **C1 + C2 + C4** (method + the PySR-immunity result).
**Cut:** most of SPARC/cosmology detail — keep only Fig. dichotomy as teaser.
**Why this works:** Workshop reviewers want a crisp methodological contribution + one strong empirical hook. The "PySR collapses to trig at 0% noise, ADCD doesn't" result is *memorable* and visually striking (Fig.1). 4-page format forces you to sharpen the core claim.

**Suggested title (workshop):** *Physics-Gated Correction Discovery: Why Constraining the Search Space Beats Scaling It in Noisy Symbolic Regression*

### Track 2 — Extended journal (2027)
**Best fits:** *Machine Learning: Science and Technology* (IOP), *J. Computational Physics*, or *Entropy* (MDPI, faster).

**Angle to lead with:** **C3** (structural dichotomy) as the *scientific finding*, with C1/C2 as enabling methodology.
**Expand:** full SPARC analysis, full cosmological null result, multivariable extension (currently in Limitations — promote once fixed), EFT correspondence (§3).
**Why this works:** Journals reward breadth + a genuine scientific claim. The dichotomy — "individual gravitational systems admit functional corrections; cosmological tensions are amplitude-only" — is a *finding*, not just a method. That's journal-grade.

**Suggested title (journal):** *Anomaly-Driven Correction Discovery: A Correction-First Framework for Distinguishing Functional from Amplitude Anomalies in Physical Laws*

### Track 3 — arXiv (now)
**Use the current paper as-is.** It's already 38 pages, well-structured, honestly hedged. The abstract is strong. Don't reformat for arXiv — just submit.

---

## 5. Honest risk register — know your weak spots before reviewers do

| Risk | Severity | Mitigation already in paper | What to strengthen |
|---|---|---|---|
| **Independent author, no affiliation** | 🟡 Medium | None possible | Workshop first (lower bar); cite-heavy Related Work to show community embedment |
| **Seed=42 disclosure looks like cherry-picking** | 🟢 Mitigated | Disclosed explicitly; mean 80.4%±7.4% is primary claim | ✅ Full per-seed×per-noise distribution now shipped (`results/seed_distribution.json`); PySR comparison reframed seed-independent (ADCD worst seed 66.7% > PySR fair 11.1% at 5% noise). Remaining gap: PySR itself is single-run unseeded — note this as a limitation, don't overclaim symmetry |
| **Multivariable ADCD is weak (2/4 at 5% noise)** | 🟠 High | Honestly placed in Limitations | Either fix (log Π-coeffs) or drop entirely from workshop version |
| **"ADCD beats PySR" — is the comparison fair?** | 🟡 Medium | `fair` profile, doubled-budget `generous` profile, same residual | Add PhySO to the comparison for the journal version (huge credibility boost — and Tenachi may help if endorsed) |
| **SPARC θ̂₁≈0.27 vs canonical c=4.0 (factor 15)** | 🟠 High | Honest framing: "rediscovery of family, not novel family"; cross-validation + bootstrap | This is the most attackable number. For journal: get a co-author from galaxy dynamics community |
| **Structural dichotomy is "observation, not proof"** | 🟡 Medium | Explicitly hedged | For journal: add a formal statistical test or theoretical argument |
| **Mock Proposer (template bank) feels hand-crafted** | 🟡 Medium | Grammar Proposer + LLM Proposer also reported | Lead with Grammar/LLM proposer in workshop to avoid "template engineering" criticism |

---

## 6. The 3 most likely reviewer attacks — and your pre-built rebuttals

**Attack A: *"This is just feature engineering on the correction term."***
→ Rebuttal: "No. The correction subspace is defined by the *physical structure* of the problem (EFT operator enumeration), not hand-tuned features. The gates are physics priors (dimensional homogeneity, decoupling limit), not ML features. See §3 'Relation to EFT'."

**Attack B: *"The PySR comparison is unfair — PySR isn't designed for corrections / you only beat it on your best seed."***
→ Rebuttal: "We agree PySR is tabula rasa — that's precisely the point. We ran PySR on the *same residual* (its natural target) at fair AND 2× budget. **The comparison is seed-independent:** at 5% noise, ADCD's *worst* of 16 seeds (66.7%, 6/9) still beats PySR fair (11.1%, 1/9), and PySR with *doubled* budget (generous, 55.6%) cannot reach ADCD's worst seed. The seed=42 figure (88.9%) is the peak, not the headline. The bottleneck is structural selection, not search budget. See Table baseline + generous ablation + `results/seed_distribution.json`."

**Attack C: *"The SPARC result is not novel — MOND is 40 years old."***
→ Rebuttal: "Correct, and we claim exactly that: *rediscovery of a known family from data*, not a novel family. The contribution is that an SR method *autonomously* lands on Simple-MOND without being told the answer, and then *honestly reports* it can't be distinguished from expert 2-param forms. The methodology — not the physics — is the contribution."

---

## 7. Immediate next actions (prioritized)

| # | Action | When | Why |
|---|---|---|---|
| 1 | **Wait for Tenachi endorsement** (7–10 days) | Now | Gates everything |
| 2 | **Prepare PhySO head-to-head** for journal version | Aug | Closes biggest reviewer gap |
| 3 | **Draft 4-page workshop paper** from §1–§7 of current paper | Mid-July | Sept deadline; start early |
| 4 | **Promote/fix multivariable extension** | Aug | Decide: fix or cut |
| 5 | **Get full seed distribution** (not just mean±std) | ✅ Done | Pre-empts cherry-picking attack — `results/seed_distribution.json` ships with the bundle; PySR comparison reframed seed-independent (worst ADCD seed 66.7% > PySR fair 11.1%) |
| 6 | **Identify galaxy-dynamics co-author** for SPARC | Sept–Oct | Journal credibility on θ̂₁ |

---

## 8. The single most important strategic decision

**Should the workshop paper lead with the *method* (C1/C2) or the *finding* (C3 dichotomy)?**

- **Lead with method** → safer, fits ML audience, easier to defend in 4 pages. **(My recommendation.)**
- **Lead with finding** → bolder, fits AI4Science "discovery" framing, but the dichotomy needs SPARC + cosmology which won't fit in 4 pages and is harder to defend without full detail.

Recommendation: **workshop = method-led; journal = finding-led.** This also gives you two genuinely different papers from one body of work, maximizing publication count without self-plagiarism (different framing, different audiences).

---

*This document is your strategic anchor. Revisit before each submission.*
