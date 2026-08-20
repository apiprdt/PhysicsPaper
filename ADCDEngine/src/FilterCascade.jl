# ADCD Engine: FilterCascade
module FilterCascade

using ..ADCDDimensions
using ..PrimitiveRegistry
using ..CorrectionProposer
using ..ConstantFitter
using ..IdentifiabilityGate

export RunConfig, GateStats, ADCDResult
export run_filter_cascade, run_cascade_on_proposals

mutable struct GateStats
    n_input      ::Int
    n_pass_gate_a::Int
    n_pass_gate_b::Int
    n_pass_gate_c::Int
    n_pass_gate_d::Int
    n_pass_gate_e::Int
    n_withheld   ::Int
end

GateStats() = GateStats(0,0,0,0,0,0,0)

# Bug #2 FIX: Added correction_type field. "multiplicative" = y_cl*(1+Δ),
# "additive" = y_cl+Δ. Previously hardcoded multiplicative, making 4/5
# real-physics scenarios (Mercury, Muon g-2, Pulsar, etc.) unfittable
# when y_classical≡0 (gradient=0, optimizer sees nothing).
struct RunConfig
    domain          ::String
    target_dim      ::String
    input_vars      ::Vector{String}
    known_constants ::Dict{String,Float64}
    bic_threshold   ::Float64
    nmse_coarse     ::Float64
    nmse_fine       ::Float64
    n_restarts      ::Int
    groups          ::Union{Vector{Vector{Int}},Nothing}
    max_proposals   ::Int
    correction_type ::String   # "multiplicative" or "additive"
    classical_limit_direction ::String # "0" or "oo"
end

# Backward-compat constructor: default to "multiplicative" and "0"
RunConfig(domain, target_dim, input_vars, known_constants, bic_threshold,
          nmse_coarse, nmse_fine, n_restarts, groups, max_proposals) =
    RunConfig(domain, target_dim, input_vars, known_constants, bic_threshold,
              nmse_coarse, nmse_fine, n_restarts, groups, max_proposals,
              "multiplicative", "0")

struct ADCDResult
    proposal  ::CorrectionProposal
    fit       ::FitResult
    verdict   ::IdentVerdict
    delta_bic ::Float64
    gate_stats::GateStats
end

# ---------------------------------------------------------------------------
# Individual gate implementations
# ---------------------------------------------------------------------------

function gate_a_dimensional(proposal::CorrectionProposal, target_dim::String)::Bool
    verify_dimension(proposal.expr, target_dim)
end

function gate_b_asymptotic(
    proposal        ::CorrectionProposal,
    vars_data       ::Dict{String,Vector{Float64}},
    constants       ::Dict{String,Float64},
    limit_direction ::String
)::Bool
    # Bug #1 Fix (Audit): Remove hardcoded check for D_rar.
    # Instead, use the classical_limit_direction provided by the scenario config.
    is_inf = limit_direction == "oo"
    
    test_val = is_inf ? 1e12 : 1e-12
    threshold = is_inf ? 1e-3 : 1e-6

    test_vars = Dict{String,Vector{Float64}}()
    for k in keys(vars_data)
        if haskey(constants, k)
            test_vars[k] = fill(constants[k], length(first(values(vars_data))))
        else
            test_vars[k] = fill(test_val, length(first(values(vars_data))))
        end
    end
    
    try
        y = evaluate_expr(proposal.expr, test_vars, constants, ones(proposal.n_params))
        return all(abs.(y) .< threshold)
    catch
        return false
    end
end

function gate_c_coarse(
    proposal        ::CorrectionProposal,
    y_classical     ::Vector{Float64},
    y_obs           ::Vector{Float64},
    vars_data       ::Dict{String,Vector{Float64}},
    constants       ::Dict{String,Float64},
    nmse_threshold  ::Float64,
    correction_type ::String,
)::Union{FitResult, Nothing}
    result = fit_constants(proposal.expr, y_classical, y_obs, vars_data, constants,
                           proposal.n_params; n_restarts=1, rng_seed=0,
                           correction_type=correction_type)
    result.nmse <= nmse_threshold ? result : nothing
end

function gate_d_fine(
    proposal        ::CorrectionProposal,
    y_classical     ::Vector{Float64},
    y_obs           ::Vector{Float64},
    vars_data       ::Dict{String,Vector{Float64}},
    constants       ::Dict{String,Float64},
    n_restarts      ::Int,
    sigma_y         ::Union{Vector{Float64},Nothing},
    correction_type ::String,
)::Union{FitResult, Nothing}
    result = fit_constants(proposal.expr, y_classical, y_obs, vars_data, constants,
                           proposal.n_params; n_restarts=n_restarts, sigma_y=sigma_y,
                           correction_type=correction_type)
    result.converged ? result : nothing
end

# ---------------------------------------------------------------------------
# Bug #1 FIX: The old local gate_e_identifiability has been removed.
# It used fit.theta[end] as a flat constant for the null model (wrong),
# compared k=1 null vs k=n_params correction (wrong — null must be k=0),
# and approximated ll_corr indirectly via fit.nmse (inconsistent with
# the likelihood actually computed in ConstantFitter).
#
# IdentifiabilityGate.identifiability_gate (the correct implementation) is
# now called directly. delta_bic is computed here separately so it can be
# serialized to the JSON output.
# ---------------------------------------------------------------------------

function _compute_delta_bic(
    fine       ::FitResult,
    y_classical::Vector{Float64},
    y_obs      ::Vector{Float64},
    config     ::RunConfig,
)::Float64
    n = length(y_obs)
    
    # Compute null model residuals in delta space
    if config.correction_type == "additive"
        resid_null = y_obs .- y_classical
    else
        resid_null = (y_obs .- y_classical) ./ (y_classical .+ 1e-300)
    end
    
    sigma2_null = sum(resid_null .^ 2) / n
    ll_null = sigma2_null > 0 ? -0.5 * n * log(2 * pi * sigma2_null) - n / 2.0 : -Inf

    bic_null = if config.groups !== nothing
        n_eff = length(config.groups)
        # Proper hierarchical likelihood using the effective sample size directly
        ll_null_eff = sigma2_null > 0 ? -0.5 * n_eff * log(2 * pi * sigma2_null) - n_eff / 2.0 : -Inf
        hierarchical_bic(n_eff, 0, ll_null_eff)
    else
        bic_score(n, 0, ll_null)
    end

    bic_corr = if config.groups !== nothing
        n_eff = length(config.groups)
        sigma2_corr = sum(fine.residuals .^ 2) / length(fine.residuals)
        ll_corr_eff = sigma2_corr > 0 ? -0.5 * n_eff * log(2 * pi * sigma2_corr) - n_eff / 2.0 : -Inf
        hierarchical_bic(n_eff, fine.n_params, ll_corr_eff)
    else
        bic_score(n, fine.n_params, fine.likelihood)
    end

    return bic_null - bic_corr  # positive = correction is better
end

function run_filter_cascade(
    proposal   ::CorrectionProposal,
    y_classical::Vector{Float64},
    y_obs      ::Vector{Float64},
    vars_data  ::Dict{String,Vector{Float64}},
    config     ::RunConfig;
    sigma_y    ::Union{Vector{Float64},Nothing} = nothing,
)::Tuple{Union{ADCDResult, Nothing}, GateStats}

    ct = config.correction_type
    stats = GateStats()
    stats.n_input = 1

    # Gate A: Dimensional check (microseconds)
    gate_a_dimensional(proposal, config.target_dim) || return (nothing, stats)
    stats.n_pass_gate_a = 1

    # Gate B: Asymptotic safety (microseconds)
    gate_b_asymptotic(proposal, vars_data, config.known_constants, config.classical_limit_direction) || return (nothing, stats)
    stats.n_pass_gate_b = 1

    # Gate C: Coarse (fast, ~10ms)
    coarse = gate_c_coarse(
        proposal, y_classical, y_obs, vars_data, config.known_constants,
        config.nmse_coarse, ct)
    coarse === nothing && return (nothing, stats)
    stats.n_pass_gate_c = 1

    # Gate D: Fine (multi-start, ~100ms)
    fine = gate_d_fine(
        proposal, y_classical, y_obs, vars_data, config.known_constants,
        config.n_restarts, sigma_y, ct)
    fine === nothing && return (nothing, stats)
    stats.n_pass_gate_d = 1

    # Gate E: Identifiability verdict (Bug #1 fix: use IdentifiabilityGate module)
    verdict = IdentifiabilityGate.identifiability_gate(
        fine, y_classical, y_obs;
        bic_threshold   = config.bic_threshold,
        nmse_threshold  = config.nmse_fine,
        groups          = config.groups,
        correction_type = config.correction_type,
    )
    delta_bic = _compute_delta_bic(fine, y_classical, y_obs, config)

    if verdict == IDENTIFIABLE
        stats.n_pass_gate_e = 1
    else
        stats.n_withheld = 1
    end

    return (ADCDResult(proposal, fine, verdict, delta_bic, stats), stats)
end

"""
    run_cascade_on_proposals(proposals, y_classical, y_obs, vars_data, config)
        -> Tuple{Vector{ADCDResult}, GateStats}

Run all proposals through the filter cascade.
Returns all results (IDENTIFIABLE first, then WITHHELD), sorted by delta_bic.
"""
function run_cascade_on_proposals(
    proposals  ::Vector{CorrectionProposal},
    y_classical::Vector{Float64},
    y_obs      ::Vector{Float64},
    vars_data  ::Dict{String,Vector{Float64}},
    config     ::RunConfig;
    sigma_y    ::Union{Vector{Float64},Nothing} = nothing,
    verbose    ::Bool = true,
)::Tuple{Vector{ADCDResult}, GateStats}

    agg = GateStats()
    agg.n_input = length(proposals)
    results = ADCDResult[]
    n_limited = min(length(proposals), config.max_proposals)

    if verbose
        println("[FilterCascade] Running $(n_limited) proposals through 5-gate cascade...")
        println("[FilterCascade] Domain: $(config.domain), target_dim: $(config.target_dim), correction: $(config.correction_type)")
    end

    for (i, proposal) in enumerate(proposals[1:n_limited])
        result, stats = run_filter_cascade(
            proposal, y_classical, y_obs, vars_data, config; sigma_y=sigma_y)
        
        # Accumulate stats from all proposals, even those rejected
        agg.n_pass_gate_a += stats.n_pass_gate_a
        agg.n_pass_gate_b += stats.n_pass_gate_b
        agg.n_pass_gate_c += stats.n_pass_gate_c
        agg.n_pass_gate_d += stats.n_pass_gate_d
        agg.n_pass_gate_e += stats.n_pass_gate_e
        agg.n_withheld += stats.n_withheld
        
        result === nothing && continue
        push!(results, result)
    end

    # Sort: IDENTIFIABLE first, then by delta_bic descending
    sort!(results, by=r -> (r.verdict != IDENTIFIABLE, -r.delta_bic))

    verbose && begin
        println("[FilterCascade] Complete.")
        show(agg)
        println("[FilterCascade] IDENTIFIABLE results: $(agg.n_pass_gate_e)")
    end

    return results, agg
end

mean(x) = sum(x) / length(x)

end  # module FilterCascade
