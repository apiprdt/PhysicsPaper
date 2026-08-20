# ADCD Engine: IdentifiabilityGate
module IdentifiabilityGate

using ..ConstantFitter

export IdentVerdict, IDENTIFIABLE, WITHHELD, POSITIVE_CONTROL_FAILED
export bic_score, hierarchical_bic, identifiability_gate

@enum IdentVerdict IDENTIFIABLE WITHHELD POSITIVE_CONTROL_FAILED
function bic_score(n_points::Int, n_params::Int, log_likelihood::Float64)::Float64
    log(n_points) * n_params - 2.0 * log_likelihood
end



"""
    identifiability_gate(fit_result, y_classical, y_obs;
                         bic_threshold=6.0, nmse_threshold=0.1,
                         groups=nothing) -> IdentVerdict

Compute BIC-based identifiability verdict.

Parameters:
  - fit_result: FitResult from ConstantFitter
  - y_classical: classical prediction (positive control)
  - y_obs: observed data
  - bic_threshold: DELTA_BIC required for IDENTIFIABLE (default=6.0 = "strong evidence")
  - nmse_threshold: max allowed NMSE for IDENTIFIABLE
  - groups: optional vector of group sizes for hierarchical BIC correction
"""
function identifiability_gate(
    fit_result   ::FitResult,
    y_classical  ::Vector{Float64},
    y_obs        ::Vector{Float64};
    bic_threshold::Float64 = 6.0,
    nmse_threshold::Float64 = 0.1,
    groups       ::Union{Vector{Vector{Int}},Nothing} = nothing,
    correction_type::String = "multiplicative",
)::IdentVerdict

    n = length(y_obs)
    
    # Compute null model residuals in delta space to match fit_result.residuals
    if correction_type == "additive"
        resid_classical = y_obs .- y_classical
    else
        resid_classical = (y_obs .- y_classical) ./ (y_classical .+ 1e-15)
    end
    
    sigma2_classical = mean(resid_classical.^2)
    if !isfinite(sigma2_classical) || sigma2_classical <= 0.0
        return POSITIVE_CONTROL_FAILED
    end

    # BIC for null model
    ll_classical = -0.5 * n * log(2π * sigma2_classical) - n/2.0
    bic_null = bic_score(n, 0, ll_classical)

    # BIC for correction model
    !fit_result.converged && return WITHHELD
    !isfinite(fit_result.likelihood) && return WITHHELD

    bic_correction = bic_score(n, fit_result.n_params, fit_result.likelihood)

    delta_bic = bic_null - bic_correction  # positive = correction is better


    # NMSE check
    fit_result.nmse > nmse_threshold && return WITHHELD

    # BIC check
    delta_bic >= bic_threshold && return IDENTIFIABLE

    return WITHHELD
end

# Convenience
mean(x) = sum(x) / length(x)
var(x)  = mean((x .- mean(x)).^2)

end  # module IdentifiabilityGate
