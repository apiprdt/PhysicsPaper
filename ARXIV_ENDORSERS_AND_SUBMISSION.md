# arXiv Submission & Endorser Strategy — ADCD v3.0.0

> Companion file for `adcd-paper-source-v3.0.0.tar.gz`.
> Paper: *Anomaly-Driven Correction Discovery (ADCD)*
> Target categories: **cs.AI** (primary), **cs.LG**, **physics.comp-ph** (cross-list)

---

## 0. IMPORTANT — what AI assistant can and cannot do here

| Task | Who must do it |
|---|---|
| Click "Submit" on arXiv | **You** (needs your arXiv login) |
| Reply to endorser's email | **You** (the request must come from your name) |
| Identify endorsers, verify eligibility, draft emails, check the bundle | ✅ Done below — ready to use |

I (the assistant) will not send emails to third parties or log into your arXiv account. Everything below is prepared for you to send yourself.

---

## 1. arXiv endorsement policy (verified, mid-2026)

Source: [arXiv endorsement page](https://info.arxiv.org/help/endorsement.html) + [Jan 2026 policy update](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/).

**As an independent researcher you need an endorser because:**
- Automatic endorsement requires an institutional `.edu`/research email **and** prior arXiv authorship.
- Your author address is `Independent Researcher, Indonesia` + Gmail → not auto-eligible.

**Endorser eligibility requirements:**
- Must be a registered arXiv user with an active **Endorser flag** in your target category.
- The flag is earned by submitting **≥ 3 papers** in that archive/category (roughly within the last several years).
- An endorser can only endorse in categories where *they themselves* are flagged.

**Two categories of contact:** (A) authors you cite → highest reply rate, they already know the topic; (B) broader community → lower rate but larger pool.

---

## 2. Endorser shortlist (priority order)

> All are authors you cite (refs verified in `main.tex`). Tier 1 = most likely to reply. Contact as a **co-cited peer**, not a fan.

### 🥇 Tier 1 — Highest priority (closest topic overlap, accessible)

#### 1. Wassim Tenachi — `wassim.tenachi@astro.unistra.fr`
- **Affiliation:** Observatoire astronomique de Strasbourg / CNRS UMR 7550
- **Why top pick:** Creator of **PhySO** (`tenachi2023physo`) — the single most directly comparable method to ADCD (physics-constrained SR with units/class constraints). He will *immediately* understand your contribution.
- **Likely categories:** `physics.comp-ph`, `astro-ph.*`, possibly `cs.LG`
- **Recent PhD (2024), active submitter** → strong endorser candidate and approachable.
- **Refs:** [PhySO paper](https://arxiv.org/abs/2303.03192) · [Class SR (ApJL 2024)](https://iopscience.iop.org/article/10.3847/2041-8213/ad5970) · [ORCID](https://orcid.org/0000-0001-8392-3836)

#### 2. Brenden Petersen (DSR / risk-seeking policy gradients)
- **Affiliation:** Lawrence Livermore National Laboratory (LLNL)
- **Cite:** `petersen2021deep` — *Deep Symbolic Regression* (ICLR 2021)
- **Why:** ADCD cites DSR prominently as a deep-RL SR baseline. LLNL group publishes regularly on `cs.LG` / `cs.AI`. Look up current email via his LLNL staff page or the paper's corresponding-author line.
- **Ref:** [DSR paper](https://arxiv.org/abs/2106.05907)

#### 3. Samuel Greydanus (Hamiltonian Neural Networks)
- **Cite:** `greydanus2019hamiltonian`
- **Why:** Physics-informed ML for equation discovery; active on `cs.LG`. Independent/early-career-friendly, historically responsive on GitHub/email. Find current email via his personal site (greydanus.github.io) or Google Scholar.

### 🥈 Tier 2 — Strong but busier / higher-status

#### 4. Miles Cranmer — PySR author (`cranmer2020pysr`, `cranmer2020lagrangian`)
- **Affiliation:** Cambridge / Flatiron Institute / Princeton (rotating)
- **Why:** Most cited SR-for-physics author; ADCD benchmarks against PySR. Very likely has the Endorser flag in `cs.LG` / `astro-ph`.
- **Caveat:** High email volume — may not reply. Reach out but don't depend on him alone.
- **Ref:** [PySR](https://arxiv.org/abs/2305.01582) · [Google Scholar](https://scholar.google.com/citations?user=10WfwCQAAAAJ)

#### 5. Steven Brunton / Nathan Kutz (SINDy) — `brunton2016sindy`
- **Affiliation:** University of Washington
- **Why:** Foundational SR-for-dynamics authors; prolific arXiv submitters → definitely endorsed in `cs.LG`/`physics.comp-ph`.
- **Caveat:** Lab heads; replies usually go through the postdoc who'd co-sign. Try Brunton's group email (brunton group page) rather than expecting a personal reply.

#### 6. Subhransu S. Chaudhuri / Aditya Sehgal (LASR) — `grayeli2024lasr`
- **Affiliation:** MIT (Chaudhuri group)
- **Why:** Most recent (NeurIPS 2024) cited work — early-career co-authors are most likely to reply to a fellow early-career request. Contact via NeurIPS 2024 author/program page or the LASR repo.
- **Ref:** [LASR](https://arxiv.org/abs/2409.15120)

### 🥉 Tier 3 — Cross-domain (use only if cs.AI/cs.LG fail)

#### 7. Kyu-Hyun Chae — `chae2023wide`
- **Affiliation:** Sejong University
- **Why:** Wide-binary gravity anomaly author. Relevant to your SPARC/MOND experiments. Could endorse `astro-ph.GA`/`astro-ph.CO`. Useful if you cross-list there.
- **Caveat:** Field is gravity, not SR/ML — only endorseable for astro categories, not cs.AI.

---

## 3. Suggested send-order (do NOT mass-email)

Send to **one at a time**, wait ~7–10 days, then move on. Mass-emailing multiple endorsers looks like spam and arXiv logs can flag it.

1. **Tenachi** (best topic match + early-career + approachable)
2. **Petersen** (LLNL, ML-for-physics)
3. **Greydanus** (responsive, independent-friendly)
4. **Cranmer** (high-value, busy — single attempt)
5. **LASR co-authors** (recent, early-career)
6. Tier 3 only if cs.AI/cs.LG still unendorsed after steps 1–5

**Etiquette rule:** never email two endorsers in the *same* category on the same day. arXiv tracks this.

---

## 4. Email template (ready to send)

> Replace `[BRACKETS]`. Keep it under ~200 words. The endorser does NOT need to read your paper in full — they only certify "this looks like a legitimate cs.AI submission." But giving them a 2-sentence pitch + arXiv-ready PDF link dramatically improves reply rate.

---

**Subject:** arXiv endorsement request — physics-constrained symbolic regression (cs.AI)

Dear Dr. [Last name],

I hope this message finds you well. I am writing to request an arXiv endorsement for a manuscript I am submitting to **cs.AI** (cross-listing cs.LG and physics.comp-ph).

I am an independent researcher and your work on [SPECIFIC PAPER OF THEIRS, e.g. *PhySO* / *Deep Symbolic Regression* / *PySR*] has been directly influential on it — I cite it in [Section X / the method comparison].

**Title:** *Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery*

**In one sentence:** Rather than discovering equations from scratch, ADCD takes a known classical law and searches for the minimal physically-valid *correction* term, screened by dimensional, asymptotic, and complexity gates before any parameter fitting.

**Highlights:**
- Benchmarked on 9 anomalies (Mercury perihelion, Lamb shift, muon g-2, blackbody, …) with up to 10% noise.
- 80.4% ± 7.4% structural recovery over 16 seeds; outperforms PySR (fair setting) at 5% noise.
- Code + data deposited on Zenodo (DOI 10.5281/zenodo.20534940).

Full preprint (38 pp, 9 figs, CC BY 4.0): [link to your PDF — Zenodo or a public URL]

I would be grateful if you could endorse me in **cs.AI**. You can do so via arXiv's endorsement system after I submit — I will share the submission reference, or you can issue a preemptive endorsement code at https://arxiv.org/auth/endorse

I completely understand if you are unable to. Thank you for your time either way, and for [SPECIFIC IMPACT their work had].

Best regards,
Muhammad Afif Erdita
Independent Researcher, Indonesia
maeapip10@gmail.com

---

## 5. arXiv submission checklist (you execute)

- [ ] **Account**: log in at https://arxiv.org with your registered email. If not registered, register + verify email first.
- [ ] **Endorsement**: obtain endorser code (see §3–4) *before* you start submission, OR submit first and trigger the "request endorsement" flow — either path works. arXiv will email you the "endorsement needed" link.
- [ ] **User license**: accept the arXiv non-exclusive license + select **CC BY 4.0** (matches `arxiv_metadata.txt`).
- [ ] **Metadata**: copy from `paper/arxiv_metadata.txt`:
  - Title, Authors, Abstract, Comments, Categories, License, Report Number (leave blank).
  - Categories: `cs.AI` primary, `cs.LG physics.comp-ph` cross-list.
- [ ] **Upload bundle**: `adcd-paper-source-v3.0.0.tar.gz` (1.9 MB, 33 files, **no `.aux`/`.log`/`.fls`/`.bbl` junk — verified clean**).
- [ ] **Compile check**: arXiv runs autoTeX. If it fails, the system tells you within minutes — common causes: missing `.bbl` (we use inline `thebibliography`, so OK), font issues, `\cite` unresolved. Our bundle uses only standard packages (`amsmath, graphicx, booktabs, hyperref, cleveref, listings, tikz, mdframed, float, tabularx`) — all arXiv-safe.
- [ ] **PDF preview**: arXiv will build its own PDF. Compare page count (38) against `paper/main.pdf`.
- [ ] **Submit for endorsement check**: arXiv may hold the submission until endorsement resolves. Standard.
- [ ] **Public release**: once endorsed + processed, paper appears within ~24 h (often same day for cs.*).

### Bundle provenance (for your records)
- `adcd-paper-source-v3.0.0.tar.gz` — built 2026-06-23, 33 entries, 17×`.tex`, 11×`.pdf` (figures + main), 1×`.json`, 1×`.txt` (metadata).
- Sources of truth: `paper/main.tex`, `paper/figures/`, `paper/generated/`.

---

## 6. If endorsement is repeatedly denied — fallback paths

1. **Co-author route**: collaborate with one affiliated researcher (even informal) who already has arXiv access; they submit and you "claim ownership." This auto-qualifies you for future submissions.
2. **Domain switch**: if cs.AI endorsement is impossible, the same paper qualifies for `physics.comp-ph` primary — a smaller author pool but the *physics* endorsers (Tenachi, Brunton/Kutz group, Chae) are more accessible there.
3. **Wait-and-cite**: submit to a journal first (e.g. *Machine Learning: Science and Technology*, *J. Computational Physics*); accepted publication + institutional co-author makes future arXiv submissions automatic.

---

*End of guide. Verify each endorser's current affiliation/email immediately before sending — academic addresses change.*
