# ADCDEngine.jl
# Main module for ADCD Engine v2 (Julia backend).
# Architecture: Python (adcd/) <-> juliacall <-> ADCDEngine.jl
# Submodules: ADCDDimensions, PrimitiveRegistry, CorrectionProposer,
#             ConstantFitter, IdentifiabilityGate, FilterCascade
module ADCDEngine

include("ADCDDimensions.jl")
include("PrimitiveRegistry.jl")
include("CorrectionProposer.jl")
include("ConstantFitter.jl")
include("IdentifiabilityGate.jl")
include("FilterCascade.jl")

using .ADCDDimensions
using .PrimitiveRegistry
using .CorrectionProposer
using .ConstantFitter
using .IdentifiabilityGate
using .FilterCascade
using JSON3

export run_adcd, RunConfig, ADCDResult, GateStats
export IDENTIFIABLE, WITHHELD, POSITIVE_CONTROL_FAILED
export ProposalConfig, propose_corrections
export verify_dimension, list_primitives

function run_adcd(config_json::String, data_json::String)::String
    config_dict = JSON3.read(config_json, Dict{String,Any})
    data_dict   = JSON3.read(data_json,   Dict{String,Any})

    config = RunConfig(
        config_dict["domain"],
        config_dict["target_dim"],
        Vector{String}(config_dict["input_vars"]),
        Dict{String,Float64}(k=>Float64(v) for (k,v) in get(config_dict,"known_constants",Dict())),
        Float64(get(config_dict, "bic_threshold", 6.0)),
        Float64(get(config_dict, "nmse_coarse",   1.0)),
        Float64(get(config_dict, "nmse_fine",     0.1)),
        Int(get(config_dict, "n_restarts", 15)),
        get(config_dict, "groups", nothing) === nothing ? nothing : [Vector{Int}(g) for g in config_dict["groups"]],
        Int(get(config_dict, "max_proposals", 500)),
        String(get(config_dict, "correction_type", "multiplicative")),  # Bug #2 fix
    )

    # FIX (Deep Audit): parse excluded_primitives so positive_control and
    # ablation_control work correctly in the Julia engine path.
    # Previously this field was silently ignored.
    excluded_raw = get(config_dict, "excluded_primitives", nothing)
    excluded_set = if excluded_raw === nothing || isempty(excluded_raw)
        Set{Symbol}()
    else
        Set{Symbol}(Symbol(s) for s in excluded_raw)
    end

    y_classical = Vector{Float64}(data_dict["y_classical"])
    y_obs       = Vector{Float64}(data_dict["y_obs"])
    vars_data   = Dict{String,Vector{Float64}}(
        k => Vector{Float64}(v) for (k,v) in data_dict["vars"]
    )
    sigma_y_raw = get(data_dict, "sigma_y", nothing)
    sigma_y = sigma_y_raw === nothing ? nothing : Vector{Float64}(sigma_y_raw)

    vars_and_consts = vcat(config.input_vars, collect(keys(config.known_constants)))
    prop_config = ProposalConfig(
        config.domain, vars_and_consts,
        3, true, true,
    )
    proposals_all = propose_corrections(prop_config)

    # Apply primitive exclusion filter (for positive_control / ablation_control)
    proposals = if isempty(excluded_set)
        proposals_all
    else
        filter(p -> isempty(intersect(Set(p.primitives), excluded_set)), proposals_all)
    end

    # Active primitives list (for budget_disclosure reporting in Python)
    active_prims = sort(collect(Set(p for proposal in proposals for p in proposal.primitives)))

    results, stats = run_cascade_on_proposals(
        proposals, y_classical, y_obs, vars_data, config;
        sigma_y=sigma_y, verbose=true
    )

    function node_to_sympy(node::Dict)::String
        if haskey(node, "num")
            return string(node["num"])
        elseif haskey(node, "sym")
            return node["sym"]
        elseif haskey(node, "theta")
            return node["theta"]
        elseif haskey(node, "op")
            op = node["op"]
            args = node["args"]
            if op == "add"
                return "(" * join([node_to_sympy(a) for a in args], " + ") * ")"
            elseif op == "mul"
                return "(" * join([node_to_sympy(a) for a in args], " * ") * ")"
            elseif op == "div"
                return "(" * node_to_sympy(args[1]) * " / " * node_to_sympy(args[2]) * ")"
            elseif op == "pow"
                return "(" * node_to_sympy(args[1]) * "**" * node_to_sympy(args[2]) * ")"
            elseif op == "sqrt"
                return "sqrt(" * node_to_sympy(args[1]) * ")"
            elseif op == "d_lor"
                u = node_to_sympy(args[1])
                return "(1/sqrt(1 - (" * u * ")) - 1)"
            elseif op == "d_exp"
                u = node_to_sympy(args[1])
                return "(1 - exp(-Abs(" * u * ")))"
            elseif op == "d_rat"
                u = node_to_sympy(args[1])
                return "((" * u * ") / (1 + (" * u * ")**2))"
            elseif op == "d_pow"
                u = node_to_sympy(args[1])
                return "(sqrt(Abs(" * u * ")) * (1 - exp(-Abs(" * u * "))))"
            elseif op == "d_log"
                u = node_to_sympy(args[1])
                return "log(1 + Abs(" * u * "))"
            elseif op == "d_sat"
                u = node_to_sympy(args[1])
                return "tanh(" * u * ")"
            elseif op == "d_sqrt_inv"
                u = node_to_sympy(args[1])
                return "(sqrt(Abs(" * u * ")) / (1 + sqrt(Abs(" * u * "))))"
            elseif op == "d_tanh_sq"
                u = node_to_sympy(args[1])
                return "tanh((" * u * ")**2)"
            elseif op == "d_osc"
                u = node_to_sympy(args[1])
                return "(1 - cos(" * u * "))"
            elseif op == "d_nested_mond"
                u = node_to_sympy(args[1])
                return "(exp(-sqrt(Abs(" * u * "))) * (1 - exp(-sqrt(Abs(" * u * ")))))"
            elseif op == "d_rar"
                u = node_to_sympy(args[1])
                return "(exp(-sqrt(Abs(" * u * ") + 1e-15)) / Max(1 - exp(-sqrt(Abs(" * u * ") + 1e-15)), 1e-12))"
            else
                error("Unknown op: $op")
            end
        end
        return ""
    end

    output = Dict(
        "n_proposals_generated" => length(proposals_all),
        "n_proposals_evaluated" => length(proposals),
        "primitives_active"     => [string(p) for p in active_prims],
        "gate_stats" => Dict(
            "n_input"       => stats.n_input,
            "n_pass_gate_a" => stats.n_pass_gate_a,
            "n_pass_gate_b" => stats.n_pass_gate_b,
            "n_pass_gate_c" => stats.n_pass_gate_c,
            "n_pass_gate_d" => stats.n_pass_gate_d,
            "n_identifiable"=> stats.n_pass_gate_e,
            "n_withheld"    => stats.n_withheld,
        ),
        "results" => [
            Dict(
                "description" => r.proposal.description,
                "expr_str"    => node_to_sympy(r.proposal.expr),
                "pattern"     => string(r.proposal.pattern),
                "primitives"  => [string(p) for p in r.proposal.primitives],
                "n_params"    => r.proposal.n_params,
                "theta"       => r.fit.theta,
                "nmse"        => r.fit.nmse,
                "likelihood"  => r.fit.likelihood,
                "converged"   => r.fit.converged,
                "verdict"     => string(r.verdict),
                "delta_bic"   => r.delta_bic,
            )
            for r in results
        ]
    )

    return JSON3.write(output)
end

function run_adcd_from_dicts(config::Dict, data::Dict)
    JSON3.read(run_adcd(JSON3.write(config), JSON3.write(data)), Dict)
end

end  # module ADCDEngine