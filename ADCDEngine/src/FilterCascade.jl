# ============================================================================
# Modul: FilterCascade (Hardened, Multivariable-Safe & Optimized)
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

GateStats() = GateStats(0, 0, 0, 0, 0, 0, 0)

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
    classical_limit_variable ::String
end

RunConfig(domain, target_dim, input_vars, known_constants, bic_threshold, nmse_coarse, nmse_fine, n_restarts, groups, max_proposals) =
    RunConfig(domain, target_dim, input_vars, known_constants, bic_threshold, nmse_coarse, nmse_fine, n_restarts, groups, max_proposals, "multiplicative", "0", "")

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
    isempty(vars_data) && return true

    raw_vars = strip(limit_variable)
    limit_vars = isempty(raw_vars) ? collect(keys(vars_data)) : String.(split(raw_vars, ","))
    raw_dirs = strip(limit_direction)
    limit_dirs = isempty(raw_dirs) ? fill("0", length(limit_vars)) : String.(split(raw_dirs, ","))

    # Petakan arah limit spesifik untuk tiap variabel bebas
    dir_map = Dict{String,String}()
    for (i, v) in enumerate(limit_vars)
        dir_map[v] = (i <= length(limit_dirs)) ? limit_dirs[i] : "0"
    end

    # Evaluasi cepat pada 1 titik uji asimptotik (O(1) memory footprint)
    test_vars = Dict{String,Vector{Float64}}()
    for k in keys(vars_data)
        if haskey(dir_map, k)
            is_inf = dir_map[k] in ("oo", "inf", "+oo")
            test_vars[k] = [is_inf ? 1e12 : 1e-12]
        else
            # Variabel non-limit diuji pada baseline median/positif aman
            test_vars[k] = [1.0]
        end
    end

    try
        # Evaluasi asimptotik dengan parameter positif (aman dari DomainError sqrt/log)
        y_pos = evaluate_expr(proposal.expr, test_vars, constants, ones(proposal.n_params))
        # Koreksi delta wajib menuju nol (< 1e-4) pada rezim batas klasik
        return all(isfinite, y_pos) && all(abs.(y_pos) .< 1e-4)
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
    result = fit_constants(
        proposal.expr, y_classical, y_obs, vars_data, constants,
        proposal.n_params; n_restarts=3, rng_seed=42,
        sigma_y=sigma_y, correction_type=correction_type
    )
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
    result = fit_constants(
        proposal.expr, y_classical, y_obs, vars_data, constants,
        proposal.n_params; n_restarts=n_restarts, rng_seed=seed,
        sigma_y=sigma_y, correction_type=correction_type
    )
    return result.converged ? result : nothing
end

function run_filter_cascade(
    proposal   ::CorrectionProposal,
    y_classical::Vector{Float64},
    y_obs      ::Vector{Float64},
    vars_data  ::Dict{String,Vector{Float64}},
    config     ::RunConfig;
    sigma_y    ::Union{Vector{Float64},Nothing} = nothing,
    actual_space_size::Int = config.max_proposals,
)::Tuple{Union{ADCDResult, Nothing}, GateStats}

    stats = GateStats()
    stats.n_input = 1

    # Gate A: Uji Dimensi
    gate_a_dimensional(proposal, config.target_dim) || return (nothing, stats)
    stats.n_pass_gate_a = 1

    # Gate B: Uji Asimptotik
    gate_b_asymptotic(proposal, vars_data, config.known_constants, config.classical_limit_direction, config.classical_limit_variable) || return (nothing, stats)
    stats.n_pass_gate_b = 1

    # Gate C: Saringan Kasar (Coarse Gate)
    coarse = gate_c_coarse(
        proposal, y_classical, y_obs, vars_data, config.known_constants,
        config.nmse_coarse, config.correction_type, sigma_y
    )
    coarse === nothing && return (nothing, stats)
    stats.n_pass_gate_c = 1

    # Gate D: Optimasi Presisi Halus (Fine Fit - Deterministic Seed)
    seed = deterministic_hash(proposal.description)
    fine = gate_d_fine(
        proposal, y_classical, y_obs, vars_data, config.known_constants,
        config.n_restarts, sigma_y, config.correction_type, seed
    )
    fine === nothing && return (nothing, stats)
    stats.n_pass_gate_d = 1

    # Gate E: Sertifikasi Identifiabilitas Bayesian
    abs_y = abs.(y_classical)
    pos_y = abs_y[abs_y .> 0.0]
    dr = isempty(pos_y) ? 0.0 : maximum(abs_y) / minimum(pos_y)
    use_full = dr > 1e4

    verdict, delta_bic = IdentifiabilityGate.identifiability_gate(
        fine, y_classical, y_obs;
        bic_threshold     = config.bic_threshold,
        nmse_threshold    = config.nmse_fine,
        groups            = config.groups,
        correction_type   = config.correction_type,
        sigma_y           = sigma_y,
        use_full_loss     = use_full,
        search_space_size = actual_space_size
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
    real_space_size = length(proposals)

    for proposal in proposals[1:n_limited]
        result, stats = run_filter_cascade(
            proposal, y_classical, y_obs, vars_data, config;
            sigma_y = sigma_y,
            actual_space_size = real_space_size
        )
        agg.n_pass_gate_a += stats.n_pass_gate_a
        agg.n_pass_gate_b += stats.n_pass_gate_b
        agg.n_pass_gate_c += stats.n_pass_gate_c
        agg.n_pass_gate_d += stats.n_pass_gate_d
        agg.n_pass_gate_e += stats.n_pass_gate_e
        agg.n_withheld    += stats.n_withheld

        result === nothing && continue
        push!(results, result)
    end

    # Pengurutan deterministik: IDENTIFIABLE terlebih dahulu, lalu nilai delta_bic tertinggi
    sort!(results, by = r -> (r.verdict != IDENTIFIABLE, -r.delta_bic))
    return results, agg
end

end  # module FilterCascade
