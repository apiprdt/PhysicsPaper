# ============================================================================
# Modul: FilterCascade (Hardened & Cleaned)
# ============================================================================
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

struct RunConfig
    domain                   ::String
    target_dim               ::String
    input_vars               ::Vector{String}
    known_constants          ::Dict{String,Float64}
    bic_threshold            ::Float64
    nmse_coarse              ::Float64
    nmse_fine                ::Float64
    n_restarts               ::Int
    groups                   ::Union{Vector{Vector{Int}},Nothing}
    max_proposals            ::Int
    correction_type          ::String
    classical_limit_direction::String
    classical_limit_variable::String
end

struct ADCDResult
    proposal  ::CorrectionProposal
    fit       ::FitResult
    verdict   ::IdentVerdict
    delta_bic ::Float64
    gate_stats::GateStats
end

# Deterministic integer hashing (tidak tergantung random secret seed Julia runtime)
function deterministic_hash(s::String)::Int
    h = UInt64(5381)
    for b in codeunits(s)
        h = ((h << 5) + h) + UInt64(b)
    end
    return Int(h & 0x7FFFFFFFFFFFFFFF)
end

function gate_a_dimensional(proposal::CorrectionProposal, target_dim::String)::Bool
    verify_dimension(proposal.expr, target_dim)
end

function gate_b_asymptotic(
    proposal       ::CorrectionProposal,
    vars_data      ::Dict{String,Vector{Float64}},
    constants      ::Dict{String,Float64},
    limit_direction::String,
    limit_variable ::String
)::Bool
    is_inf = limit_direction == "oo"
    test_val = is_inf ? 1e12 : 1e-12
    threshold = is_inf ? 1e-3 : 1e-6
    n_pts = length(first(values(vars_data)))
    
    limit_vars = split(limit_variable, ",")

    test_vars = Dict{String,Vector{Float64}}()
    for k in keys(vars_data)
        if k in limit_vars
            test_vars[k] = fill(test_val, n_pts)
        else
            test_vars[k] = vars_data[k]
        end
    end

    try
        # Uji dengan sign +1 dan -1 untuk menghindari bias penguncian tanda parameter
        y_pos = evaluate_expr(proposal.expr, test_vars, constants, ones(proposal.n_params))
        y_neg = evaluate_expr(proposal.expr, test_vars, constants, -ones(proposal.n_params))
        return all(abs.(y_pos) .< threshold) && all(abs.(y_neg) .< threshold)
    catch
        return false
    end
end

function gate_c_coarse(
    proposal       ::CorrectionProposal,
    y_classical    ::Vector{Float64},
    y_obs          ::Vector{Float64},
    vars_data      ::Dict{String,Vector{Float64}},
    constants      ::Dict{String,Float64},
    nmse_threshold ::Float64,
    correction_type::String,
    sigma_y        ::Union{Vector{Float64},Nothing}
)::Union{FitResult, Nothing}
    # Berikan 3 restarts pada skala dasar agar tidak membuang model dengan skala berbeda
    result = fit_constants(proposal.expr, y_classical, y_obs, vars_data, constants,
                           proposal.n_params; n_restarts=3, rng_seed=42,
                           sigma_y=sigma_y, correction_type=correction_type)
    return (result.converged && isfinite(result.nmse) && result.nmse <= nmse_threshold) ? result : nothing
end

function gate_d_fine(
    proposal       ::CorrectionProposal,
    y_classical    ::Vector{Float64},
    y_obs          ::Vector{Float64},
    vars_data      ::Dict{String,Vector{Float64}},
    constants      ::Dict{String,Float64},
    n_restarts     ::Int,
    sigma_y        ::Union{Vector{Float64},Nothing},
    correction_type::String,
    seed           ::Int
)::Union{FitResult, Nothing}
    result = fit_constants(proposal.expr, y_classical, y_obs, vars_data, constants,
                           proposal.n_params; n_restarts=n_restarts, rng_seed=seed,
                           sigma_y=sigma_y, correction_type=correction_type)
    return result.converged ? result : nothing
end

function run_filter_cascade(
    proposal   ::CorrectionProposal,
    y_classical::Vector{Float64},
    y_obs      ::Vector{Float64},
    vars_data  ::Dict{String,Vector{Float64}},
    config     ::RunConfig;
    sigma_y    ::Union{Vector{Float64},Nothing} = nothing,
)::Tuple{Union{ADCDResult, Nothing}, GateStats}

    stats = GateStats()
    stats.n_input = 1

    # Gate A
    gate_a_dimensional(proposal, config.target_dim) || return (nothing, stats)
    stats.n_pass_gate_a = 1

    # Gate B
    gate_b_asymptotic(proposal, vars_data, config.known_constants, config.classical_limit_direction, config.classical_limit_variable) || return (nothing, stats)
    stats.n_pass_gate_b = 1

    # Gate C
    coarse = gate_c_coarse(proposal, y_classical, y_obs, vars_data, config.known_constants,
                           config.nmse_coarse, config.correction_type, sigma_y)
    coarse === nothing && return (nothing, stats)
    stats.n_pass_gate_c = 1

    # Gate D (Deterministik Seed)
    seed = deterministic_hash(proposal.description)
    fine = gate_d_fine(proposal, y_classical, y_obs, vars_data, config.known_constants,
                       config.n_restarts, sigma_y, config.correction_type, seed)
    fine === nothing && return (nothing, stats)
    stats.n_pass_gate_d = 1

    # Gate E
    abs_y = abs.(y_classical)
    pos_y = abs_y[abs_y .> 0.0]
    dr = isempty(pos_y) ? 0.0 : maximum(abs_y) / minimum(pos_y)
    use_full = dr > 1e4

    verdict, delta_bic = IdentifiabilityGate.identifiability_gate(
        fine, y_classical, y_obs;
        bic_threshold   = config.bic_threshold,
        nmse_threshold  = config.nmse_fine,
        groups          = config.groups,
        correction_type = config.correction_type,
        sigma_y         = sigma_y,
        use_full_loss   = use_full
    )

    if verdict == IDENTIFIABLE
        stats.n_pass_gate_e = 1
    else
        stats.n_withheld = 1
    end

    return (ADCDResult(proposal, fine, verdict, delta_bic, stats), stats)
end

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

    for proposal in proposals[1:n_limited]
        result, stats = run_filter_cascade(proposal, y_classical, y_obs, vars_data, config; sigma_y=sigma_y)
        agg.n_pass_gate_a += stats.n_pass_gate_a
        agg.n_pass_gate_b += stats.n_pass_gate_b
        agg.n_pass_gate_c += stats.n_pass_gate_c
        agg.n_pass_gate_d += stats.n_pass_gate_d
        agg.n_pass_gate_e += stats.n_pass_gate_e
        agg.n_withheld    += stats.n_withheld

        result === nothing && continue
        push!(results, result)
    end

    sort!(results, by = r -> (r.verdict != IDENTIFIABLE, -r.delta_bic))
    return results, agg
end

end  # module FilterCascade
