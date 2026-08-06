# Asymptotic Dictionary Correction Discovery (ADCD): Methodology and Results

## Methodology

### 1. The ADCD Architecture
Traditional symbolic regression attempts a *tabula-rasa* reconstruction of equations, relying on stochastic exploration (e.g., genetic algorithms) to find expressions that fit observational data. In contrast, the Asymptotic Dictionary Correction Discovery (ADCD) framework assumes a known classical baseline and deterministically enumerates structural *corrections* to that baseline. 

The ADCD pipeline operates in three distinct stages:
1.  **Deterministic Grammar Enumeration:** A deterministic proposer constructs candidate algebraic expressions from a predefined dictionary of physical variables, constants, and basic mathematical operators. 
2.  **Structural and Physical Screening (Stage 1):** Candidates are strictly evaluated against fundamental physical laws.
3.  **Numerical Optimization (Stage 2):** Surviving candidates undergo non-linear parameter estimation using JAX-based L-BFGS optimization, followed by Bayesian Information Criterion (BIC) ranking to balance fit against complexity.

### 2. Structural Screening and Asymptotic Regularization
A core tenet of the ADCD framework is that any physical correction must seamlessly vanish at the classical limit. This physical constraint serves as a powerful regularizer, actively pruning structurally invalid candidates before they reach the computationally expensive optimization stage.

Our Stage 1 pipeline implements the following cascade of filters:

*   **AST Budgeting:** The generated mathematical expression (represented as an Abstract Syntax Tree, AST) must not exceed a predefined complexity (`max_depth = 7`, `max_tokens = 25`). Crucially, we apply this budgeting to the *theta-intact* (fully parameterized) expression. This is a deliberate, strict design choice: by counting every free parameter ($\theta_i$) as a distinct token, we prevent the system from "smuggling" extreme structural complexity under the guise of an empirical constant.
*   **Dimensional Consistency:** We represent physical dimensions as integer vectors in the SI base (e.g., $[M, L, T]$). Candidates are evaluated using vector addition and scalar multiplication. Only candidates whose final dimensionality strictly matches the target physical dimension (or are dimensionless, for multiplicative corrections) are allowed to proceed.
*   **Transcendental Safety:** The arguments of any transcendental function (e.g., $\exp, \sin, \log$) must evaluate to exactly dimensionless ($[0,0,0]$). 
*   **Asymptotic Regularization (ARC):** The candidate correction $\Delta(\theta, \vec{x})$ is evaluated at the defined classical limit (e.g., $v \to 0$, $r \to \infty$). The expression *must* analytically converge to a constant (e.g., $1.0$ for multiplicative factors, $0.0$ for additive residuals). Expressions that diverge ($-\infty, \infty$) or oscillate unboundedly are immediately rejected.

## Results

### 3. Scenario Selection and Lock-in
We evaluate the ADCD architecture on three canonical synthetic physics scenarios. These scenarios represent well-known historical corrections to classical laws and are designed to test the framework's ability to recover the correct mathematical structure from noisy observational data.

1.  **Time Dilation:** (Multiplicative) The correction to classical kinetic energy at relativistic speeds.
    *   Classical: $E = \frac{1}{2} m v^2$
    *   Correction Target: $1 / \sqrt{1 - (v/c)^2}$
2.  **Screened Coulomb:** (Multiplicative) The correction to Coulomb's force law in a plasma environment (Yukawa potential).
    *   Classical: $F = k_e \frac{q_1 q_2}{r^2}$
    *   Correction Target: $\exp(-r / \lambda_D)$
3.  **Entropy Expansion:** (Additive) The correction to entropy for a non-ideal gas (van der Waals).
    *   Classical: $S = n R \ln(V - nb)$
    *   Correction Target: $n R \ln(\frac{V-nb}{V})$ (simplified proxy behavior in residuals)

### 4. Stage 1 Survival Rates
We processed a comprehensive dictionary of candidates through the Stage 1 pipeline to observe the efficacy of our structural and physical regularizers. The following table demonstrates the survival rates of the generated candidates as they pass through the screening cascade.

| Filter Stage | Time Dilation | Screened Coulomb | Entropy Expansion |
| :--- | :--- | :--- | :--- |
| **AST Complexity** | 0.56 | 0.56 | 0.56 |
| **Dimensionality** | 1.00 | 1.00 | 1.00 |
| **Transcendental** | 1.00 | 1.00 | 1.00 |
| **Asymptotic (ARC)** | 1.00 | 1.00 | 1.00 |
| **Coarse NMSE** | 1.00 | 0.93 | 0.93 |
| **Overall Stage 1 Survival** | 0.56 | 0.52 | 0.52 |

**Analysis of Survival Rates:**
The primary filter is the AST complexity limit, which permits approximately $56\%$ of the deterministically generated combinations to proceed. As noted in Section 2, this limit is intentionally strict, assessing the full parameter-laden expression to enforce true parsimony. 

For these three locked scenarios (which are all multiplicative corrections), the dimensional filter exhibits a 100% survival rate among candidates that passed the AST check. This indicates that the grammar generation successfully constrained the variables to form physically meaningful combinations for dimensionless targets. The final coarse evaluation (which rejects numerical overflows and $\text{NaN}$ values resulting from evaluating the candidate against the raw dataset) removes an additional $7\%$ of the remaining candidates in the Screened Coulomb and Entropy Expansion scenarios.
