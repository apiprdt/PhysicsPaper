# Phase 3 Report: Verdict-Change Table (v1 Protocol vs. Blind Equal-Pool Protocol)

**Date:** 2026-11-08  
**Branch:** `ExploreSmth`  
**Execution:** Pinned CPU float64 environment (Python 3.11.9, JAX 0.10.1, NumPy 2.4.3)  
**Scripts:** `scripts/run_phase3_rerun.py`, `scripts/test_imported_phase3.py`

---

## 1. Executive Summary

In ADCD v1, the formal discovery protocol compared domain-restricted search pools ($|\mathcal{C}_{\text{dom}}| \in \{60, 40, 10\}$) against unconstrained ablation pools ($|\mathcal{C}_{\text{abl}}| \in \{814, 534, 700\}$). This introduced an artificial mathematical bonus into $\Delta\mathrm{BIC}$:
$$\Delta\mathrm{BIC}_{\text{v1}} = \Delta\mathrm{BIC}_{\text{raw}} + 2 \ln\left(\frac{|\mathcal{C}_{\text{abl}}|}{|\mathcal{C}_{\text{dom}}|}\right)$$
where the penalty bonus was $+5.22$ for Screened Coulomb, $+5.18$ for Entropy Expansion, and $+8.50$ for Time Dilation.

Phase 3 introduces the **Blind Three-Tier Protocol**:
1. It eliminates all ground-truth references (`scenario.correction_expr`, `true_primitive`).
2. Self-ablation compares the winning candidate $M_1$ against the best rival from an alternative family $M_2$ **from the exact same candidate pool**, so the search-space penalty cancels out identically: $\Delta\mathrm{BIC}(M_2 \to M_1) = \mathrm{BIC}(M_2) - \mathrm{BIC}(M_1)$.
3. If $\Delta\mathrm{BIC}(M_2 \to M_1) \le 10$, the protocol honestly reports an **`EQUIVALENCE CLASS`** rather than claiming an unverified single-formula discovery.

---

## 2. Verdict-Change Table Across Canonical Scenarios

| Scenario | v1 Verdict | v1 $\Delta\mathrm{BIC}$ (with asymmetric bonus) | Penalty Bonus $2\ln\frac{|\mathcal{C}_{\text{abl}}|}{|\mathcal{C}_{\text{dom}}|}$ | Hand-Predicted Equal-Pool $\Delta\mathrm{BIC}$ | Actual Phase 3 $\Delta\mathrm{BIC}$ (Equal Pool) | Phase 3 Verdict | Primary Physical Reason for Verdict Change |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Screened Coulomb** ($r \le 4.0\text{ m}$) | `IDENTIFIABLE` | $30.65$ | $+5.22$ | $25.43$ | **$442.82$** (vs rival `D_rat`) | **`UNIQUELY IDENTIFIABLE`** | **Stays Unique:** The curvature of Yukawa exponential decay $e^{-r/\lambda} - 1$ is profoundly distinct from rational / power-law poles across $r \in [0.2, 4.0]$. Even on an equal pool, alternative families fall behind by $\Delta\mathrm{BIC} \gg 10$. |
| **Entropy Expansion** ($dV/V_i \le 1.0$) | `IDENTIFIABLE` | $14.70$ | $+5.18$ | $9.52$ | **$5.25$** (vs rival `exp` composition) | **`EQUIVALENCE CLASS`** | **Becomes EQUIVALENCE CLASS:** In v1, $\Delta\mathrm{BIC} = 14.70$ crossed the $>10$ threshold solely due to the $+5.18$ pool size penalty bonus. When evaluated on the same pool, $\ln(1+u)$ and $(1-e^{-u})$ fit the $dV/V_i \le 1$ window with $\Delta\mathrm{BIC} = 5.25 \le 10$, sharing the exact leading Taylor expansion $\sim u - \frac{1}{2}u^2$. |
| **Time Dilation** ($v \le 0.3c$) | `WITHHELD` | $13.79$ | $+8.50$ | $5.29$ | **$-511.92$** (vs null baseline) | **`WITHHELD`** | **Stays WITHHELD:** Kinematic dilation at $v \le 0.3c$ is only $\le 4.5\%$, swamped by $1\%$ observational noise ($\mathrm{NMSE} = 0.412$). The uncorrected classical baseline fits adequately ($\Delta\mathrm{BIC} \le 10$ vs null). The auditor correctly abstains. |

---

## 3. Discrepancy & Deviation Analysis

1. **Screened Coulomb Magnitude ($442.82$ vs $25.43$):**
   The hand prediction $25.43$ was computed from v1's ablated pool where multi-term unconstrained compositions were allowed. Within the disclosed domain taxonomy pool, the best available non-exponential candidate is single-primitive `D_rat`:
   $$M_2 = \theta_0 \cdot \frac{r/\theta_4}{1 - r/\theta_4}$$
   Because rational poles cannot mimic an exponential curve over a 20-fold dynamic range ($r \in [0.2, 4.0]$), $M_2$ incurs a massive likelihood penalty ($\mathrm{BIC} = -1176.54$ vs $-1619.36$), yielding an actual equal-pool $\Delta\mathrm{BIC} = 442.82$.

2. **Entropy Expansion Equivalence Class:**
   The equal-pool protocol returns an equivalence class containing 3 models:
   - Rank 1: $\theta_0 \ln(1 + dV/V_i)$ ($\mathrm{BIC} = -815.08$)
   - Rank 2: $\theta_0 \ln(1 + dV/V_i) \cdot [1 + \theta_1 (e^{-dV/V_i} - 1)]$ ($\mathrm{BIC} = -809.84$, $\Delta\mathrm{BIC} = 5.25$)
   - Rank 3: $\theta_0 (e^{-dV/V_i} - 1) + \theta_1 \ln(1 + dV/V_i)$ ($\mathrm{BIC} = -809.82$, $\Delta\mathrm{BIC} = 5.26$)
   All three share the leading asymptotic series $5.755 \cdot u - 2.87 \cdot u^2$, confirming that the data cannot tell them apart.

---

## 4. Verification on Imported Custom Formulas

The protocol was verified using `scripts/test_imported_phase3.py` on an imported Yukawa screening hypothesis with 3 external formulas (exponential, rational, tanh):
- All 3 formulas passed dual symbolic and numerical limit verification: *"limit verified, not enforced by construction"*.
- The auditor correctly identified the exponential candidate with $\Delta\mathrm{BIC} = 1090.00 > 10.0$ over rational rivals.
- The report was rendered with canonical SHA-256 hashes (`card_hash`, `report_hash`) and plain-language semantics.
