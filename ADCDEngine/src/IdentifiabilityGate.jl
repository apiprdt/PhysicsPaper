# ============================================================================
# Modul: IdentifiabilityGate (Hardened, Tested & Synchronized)
# ============================================================================
module IdentifiabilityGate

using ..ConstantFitter

export IdentVerdict, IDENTIFIABLE, WITHHELD, POSITIVE_CONTROL_FAILED
export bic_score, hierarchical_bic_penalty, identifiability_gate, hierarchical_bic, compute_effective_sample_size

@enum IdentVerdict IDENTIFIABLE WITHHELD POSITIVE_CONTROL_FAILED

"""
    bic_score(n_points, n_params, log_likelihood, n_groups=nothing)
Menghitung BIC standar atau Hierarchical BIC jika n_groups tersedia.
"""
function bic_score(n_points::Int, n_params::Int, log_likelihood::Float64, n_groups::Union{Int,Nothing}=nothing)::Float64
    !isfinite(log_likelihood) && return Inf
    # Safeguard n_eff agar selalu >= 1 untuk mencegah log(0) -> -Inf
    n_eff = (n_groups !== nothing && n_groups >= 1) ? n_groups : n_points
    n_eff = max(n_eff, 1)
    return log(n_eff) * n_params - 2.0 * log_likelihood
end

hierarchical_bic_penalty(n_groups::Int, n_params::Int)::Float64 = log(max(n_groups, 1)) * n_params

hierarchical_bic(n_groups::Int, n_params::Int, log_likelihood::Float64)::Float64 =
    bic_score(0, n_params, log_likelihood, n_groups)

compute_effective_sample_size(y_obs, y_cl, groups::Vector)::Int =
    length(groups)

"""
    identifiability_gate(fit_result, y_classical, y_obs, config) -> (IdentVerdict, Float64)
Mengembalikan status vonis dan nilai delta_bic secara terpadu tanpa duplikasi kode.
"""
function identifiability_gate(
    fit_result       ::FitResult,
    y_classical      ::Vector{Float64},
    y_obs            ::Vector{Float64};
    bic_threshold    ::Float64 = 6.0,
    nmse_threshold   ::Float64 = 0.1,
    groups           ::Union{Vector{Vector{Int}},Nothing} = nothing,
    correction_type  ::String = "multiplicative",
    sigma_y          ::Union{Vector{Float64},Nothing} = nothing,
    use_full_loss    ::Bool = false,
    search_space_size::Int = 1,
)::Tuple{IdentVerdict, Float64}

    n = length(y_obs)
    n_groups = groups !== nothing ? length(groups) : nothing

    # 1. Hitung Likelihood Null Model secara konsisten dengan ruang loss
    ll_null = -Inf
    if sigma_y !== nothing
        safe_sigma = max.(sigma_y, 1e-15)
        w_res = (y_obs .- y_classical) ./ safe_sigma
        chi2_null = sum(w_res.^2)
        ll_null = -0.5 * chi2_null - 0.5 * sum(log.(2 * pi .* (safe_sigma.^2)))
    elseif use_full_loss
        diff_null = y_obs .- y_classical
        sigma2_null = mean(diff_null.^2)
        ll_null = (isfinite(sigma2_null) && sigma2_null > 0.0) ? (-0.5 * n * log(2 * pi * sigma2_null) - 0.5 * n) : -Inf
    else
        # Pelindung pembagian nol simetris skala-invarian
        abs_y = abs.(y_classical)
        pos_y = abs_y[abs_y .> 0.0]
        scale_y = isempty(pos_y) ? 1.0 : maximum(pos_y)
        eps_y = max(scale_y * 1e-15, 1e-300)
        safe_cl = [abs(v) < eps_y ? (v >= 0 ? eps_y : -eps_y) : v for v in y_classical]
        resid_null = correction_type == "additive" ?
            (y_obs .- y_classical) :
            (y_obs .- y_classical) ./ safe_cl
        sigma2_null = mean(resid_null.^2)
        ll_null = (isfinite(sigma2_null) && sigma2_null > 0.0) ? (-0.5 * n * log(2 * pi * sigma2_null) - 0.5 * n) : -Inf
    end

    # Jika null model sudah menjelaskan data secara eksak sempurna
    if !isfinite(ll_null) && isfinite(mean((y_obs .- y_classical).^2)) && mean((y_obs .- y_classical).^2) < 1e-12
        return (WITHHELD, 0.0)
    end

    # 2. Komputasi Extended BIC (Match dengan Paper Section 3.7)
    bic_null = bic_score(n, 0, ll_null, n_groups)
    bic_base_corr = bic_score(n, fit_result.n_params, fit_result.likelihood, n_groups)
    bic_correction = bic_base_corr + 2.0 * log(max(search_space_size, 1))

    if !isfinite(bic_null) || !isfinite(bic_correction)
        return (WITHHELD, -Inf)
    end

    delta_bic = bic_null - bic_correction

    # 3. Gating checks
    if !fit_result.converged || fit_result.nmse > nmse_threshold
        return (WITHHELD, delta_bic)
    end

    if delta_bic >= bic_threshold
        return (IDENTIFIABLE, delta_bic)
    end

    return (WITHHELD, delta_bic)
end

mean(x) = sum(x) / length(x)

end  # module IdentifiabilityGate
