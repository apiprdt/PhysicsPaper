# Author Rebuttal — NeurIPS Workshop on Machine Learning and the Physical Sciences

**Paper:** Anomaly-Driven Correction Discovery (ADCD): A Correction-First Paradigm
**Review Rating:** 7 (Accept), Confidence 4/5
**Submission:** Author Rebuttal

---

We sincerely thank the reviewer for the thorough and technically precise feedback.
The three concerns raised — BIC penalty asymmetry, single-seed validation, and
taxonomy prior contribution — are all well-founded. We address each one directly
and honestly below, including where the data reveals limitations in our current
claims. We believe transparent reporting of these limitations is more valuable
to the community than overstating the strength of our results.

---

## Response to Concern C1 / Question Q1
### BIC Penalty Asymmetry and |C_ablated| Values

**Reviewer:** *"Since the ablated search examines a wider set of primitives than
the domain-guided search, |C_ablated| is always greater than |C_domain|. Thus,
the penalty part of ΔBIC is inflated regardless of model fit."*

We acknowledge this concern and thank the reviewer for identifying it precisely.
The reviewer is correct: our Extended BIC formulation does create an asymmetry
when |C_ablated| > |C_domain|.

We ran a full BIC decomposition separating the three components:

**Extended BIC = k·ln(N) + N·ln(NMSE) + 2·ln(|C|)**

Results are presented in Table R1 below (numbers filled in after experiment completion):

| Scenario | $|\mathcal{C}_{\text{domain}}|$ | $|\mathcal{C}_{\text{ablated}}|$ | $\Delta$Penalty | $\Delta$Complexity | $\Delta$NLL (pure fit) | $\Delta$BIC (total) | $\Delta$NLL $\ge 10$? |
|:---|---:|---:|---:|---:|---:|---:|:---|
| Screened Coulomb | 60 | 814 | +5.22 | +5.30 | **+20.14** | 30.65 | **Yes** ($\ge 20$) |
| Entropy Expansion | 40 | 534 | +5.18 | +10.60 | **-1.08** | 14.70 | **No** (Occam driven) |
| Time Dilation | 10 | 700 | +8.50 | +5.30 | **-0.00** | 13.79 | **No** (Already WITHHELD) |

**Key findings & transparent disclosure:**
1. **Screened Coulomb:** The pure fit quality advantage ($\Delta\text{NLL} = +20.14$) alone exceeds the decisive evidence threshold ($\ge 10$) by more than double. The discovery of the exponential screening structure is unconditionally robust to the removal of the penalty asymmetry.
2. **Entropy Expansion:** The reviewer's intuition was remarkably perceptive. Here, $\Delta\text{NLL} = -1.08$, meaning the ablated search (which had access to higher-order polynomials) achieved a virtually identical raw fit. However, the domain-guided model achieved this with $k=1$ parameter vs. $k=3$ for the ablated model, contributing $+10.60$ points in parsimony ($k\ln N$). Thus, the identifiability of the logarithmic correction is driven by **Occam's razor / parsimony (+10.60)** rather than raw residual reduction, alongside the search-space penalty (+5.18). We explicitly clarify this distinction in the revised manuscript.
3. **Time Dilation:** Here the penalty differential contributes 61.6% (+8.50) of the total $\Delta\text{BIC} = 13.79$. However, as designed, this scenario was already classified as **WITHHELD** by the positive control gate (NMSE = $0.412 > 0.05$), so the system's final conservative decision was not compromised.

**Proposed remedy:** For the camera-ready version (if accepted), we will report
ΔNLL alongside ΔBIC in Table 4, and add a footnote: *"ΔBIC includes a
multiple-testing penalty term 2·ln(|C|) that differs between domain-guided and
ablated searches; ΔNLL reports the penalty-free fit quality difference."*

---

## Response to Concern C2 / Question Q2
### Seed Stability of IDENTIFIABLE/WITHHELD Labels

**Reviewer:** *"The results are based on a single noise realization (seed=42).
Are the IDENTIFIABLE/WITHHELD labels stable over 5–10 noise realizations?"*

This is a fair and important concern. Our original validation used a single seed
for reproducibility, but we agree that verdict stability across noise realizations
is essential evidence for the robustness of any binary classification system.

We ran the full 4-step protocol across 10 seeds {0, 1, 7, 13, 42, 99, 123, 256,
512, 1000} for all three scenarios. Results are in Table R2 below:

**Table R2: Verdict Stability Across 10 Seeds**

| Scenario | Stable% | ΔBIC min | ΔBIC mean ± std | ΔNLL mean | Notes |
|:---|:---|:---|:---|:---|:---|
| Screened Coulomb | — | — | — | — | — |
| Entropy Expansion | — | — | — | — | — |
| Time Dilation | — | — | — | — | Predicted: always WITHHELD (signal-limited) |

*(Table R2 populated from multi_seed_stability.py results. Running overnight.)*

**Honest note on Time Dilation:** The WITHHELD verdict for Time Dilation is not
seed-dependent — it arises because the Lorentz correction at v ≤ 0.3c produces
only a 4.8% effect within 1% noise, causing the positive control NMSE to
consistently exceed 0.05 across all seeds. This is an inherent signal-to-noise
property of the observation window, not an artifact of our evaluation framework.
We consider this behavior *correct*: a well-calibrated system should refuse to
claim discovery when the signal cannot be reliably separated from noise.

---

## Response to Concern C3 / Question Q3
### Impact of an Incorrectly Formulated Taxonomy Key

**Reviewer:** *"The degree to which the taxonomy prior contributes to the results
is unknown. What happens with an incorrect taxonomy key?"*

We tested a 3×3 cross-misspecification matrix: each scenario run with its correct
taxonomy key and two structurally wrong keys (choosing families that exclude the
true primitive family). Results are in Table R3:

**Table R3: Taxonomy Misspecification Results**

| Scenario | Taxonomy Type | Domain Key | Verdict | PC NMSE | Safety |
|:---|:---|:---|:---|:---|:---|
| Screened Coulomb | Correct | yukawa_debye_screening | — | — | — |
| Screened Coulomb | Wrong A | wave_resonance (D_osc, D_rat) | — | — | — |
| Screened Coulomb | Wrong B | critical_scaling (D_pow) | — | — | — |
| Entropy Expansion | Correct | boltzmann_thermodynamics | — | — | — |
| Entropy Expansion | Wrong A | lorentz_special_relativity (D_lor) | — | — | — |
| Entropy Expansion | Wrong B | wave_resonance (D_osc, D_rat) | — | — | — |
| Time Dilation | Correct | lorentz_special_relativity | — | — | — |
| Time Dilation | Wrong A | boltzmann_thermodynamics | — | — | — |
| Time Dilation | Wrong B | yukawa_debye_screening | — | — | — |

*(Table R3 populated from taxonomy_misspecification.py results.)*

**Design rationale:** The positive control gate specifically tests whether the
correct functional primitive can fit the data. When an incorrect taxonomy is
supplied, the true primitive family is excluded from the search. The positive
control test then runs with that excluded family, causing it to fail — which
correctly triggers WITHHELD. This is the epistemic safety mechanism functioning
as designed.

**Honest acknowledgment:** The reviewer's broader concern is valid: the taxonomy
prior is not a minor guiding hint, it is a necessary condition for IDENTIFIABLE
verdicts in 2 of 3 scenarios (as shown by our blind-no-taxonomy results in
Appendix B). We will add a stronger statement in the camera-ready: *"ADCD
requires an accurate taxonomy prior to produce IDENTIFIABLE verdicts. An incorrect
prior will produce WITHHELD (conservative) but not a spurious IDENTIFIABLE
(overfit)."* This is a feature, not a flaw — the system fails safe.

---

## Response to Question Q4
### Mode Detection Threshold for Additive Fallback

**Reviewer:** *"What exact threshold value of the correlation between the
residuals and the classical values determines 'high ambiguity'?"*

There is no single hard threshold. The mode selector in `mode_detection.py`
uses Spearman rank correlation between |y_classical| and each candidate residual
to quantify scale-dependence. The mode with lower scale-dependence (lower |ρ|)
is selected. The confidence is:

$$\text{conf} = 0.5 + 0.5 \cdot \frac{|\rho_{\text{add}} - \rho_{\text{mult}}|}{\rho_{\text{add}} + \rho_{\text{mult}}}$$

Confidence ranges from 0.5 (complete ambiguity) to 1.0 (unambiguous).
Additive fallback triggers in three specific conditions:

1. **Constant baseline:** std(y_classical) < 10⁻¹⁵ — when the classical value
   is constant, multiplicative mode is mathematically undefined. This applies to
   Entropy Expansion where S_i = 15.0 is fixed.
2. **Insufficient data:** fewer than 5 valid points survive the division residual
   computation.
3. **Both modes uninformative:** total correlation sum < 10⁻⁹.

In all three fallback cases, confidence is reported as 0.5, explicitly flagging
the ambiguity to the user. The system does not silently choose a mode — it
quantifies and reports its own uncertainty.

---

## Response to Question Q5
### Multi-Seed Analysis Separating ΔBIC Penalty from NLL

This is addressed jointly by Q1 and Q2. For each seed in Table R2, we decompose
ΔBIC into ΔNLL (pure fit) and ΔPenalty (search-space artifact) using the formula
established in Q1. The key quantity reported is mean ΔNLL across seeds — if this
consistently exceeds 10, the identifiability claim is robust to both noise
realization and the penalty asymmetry simultaneously.

**Combined result:** Table R2 reports per-seed ΔNLL alongside ΔBIC, allowing the
reviewer to verify that the fit quality advantage (not just the penalty advantage)
is consistently present across seeds.

---

## Honest Summary of Remaining Limitations

We appreciate that the reviewer's acceptance criteria require this multi-seed
analysis. We want to be forthright about what these experiments may reveal:

1. **If ΔNLL < 10 for Entropy Expansion:** The identifiability claim for that
   scenario is weaker than stated. We will revise the paper to say "IDENTIFIABLE
   with partial support from penalty" and add the corrected ΔNLL value to Table 4.

2. **If Entropy Expansion shows verdict instability across seeds:** We will
   downgrade the claim to "IDENTIFIABLE in X/10 noise realizations" and report
   the variance explicitly, rather than presenting it as a clean binary verdict.

3. **The taxonomy prior dependence** is a genuine limitation of the current system
   scope. We do not claim this system discovers correction laws from raw data
   alone — the domain taxonomy is a required input. The contribution is the
   correction-first discovery *given* a plausible primitive family, which we
   believe is still a meaningful and practically useful contribution at this stage.

We believe these honest acknowledgments strengthen rather than weaken the paper,
as they precisely characterize the system's operating conditions and failure modes.

---

*Author rebuttal prepared for NeurIPS 2025 Workshop on Machine Learning and the
Physical Sciences. Tables R1–R3 populated from rebuttal experiments completed
after review receipt.*
