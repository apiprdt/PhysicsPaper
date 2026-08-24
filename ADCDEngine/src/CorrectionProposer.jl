# ADCD Engine: CorrectionProposer (Hardened)
module CorrectionProposer

using ..ADCDDimensions
using ..PrimitiveRegistry
using JSON3

export CorrectionProposal, ProposalConfig, propose_corrections

struct CorrectionProposal
    expr       ::Dict{String,Any}
    primitives ::Vector{Symbol}
    pattern    ::Symbol
    n_params   ::Int
    description::String
end

struct ProposalConfig
    domain           ::String
    input_vars       ::Vector{String}
    max_params       ::Int
    include_nested   ::Bool
    include_bilateral::Bool
    excluded_primitives::Vector{String}
    data_vars        ::Vector{String}
end

ProposalConfig(domain::String, input_vars::Vector{String}) =
    ProposalConfig(domain, input_vars, 3, true, true, String[], input_vars)

ProposalConfig(domain::String, input_vars::Vector{String}, max_params::Int, include_nested::Bool, include_bilateral::Bool) =
    ProposalConfig(domain, input_vars, max_params, include_nested, include_bilateral, String[], input_vars)

ProposalConfig(domain::String, input_vars::Vector{String}, max_params::Int, include_nested::Bool, include_bilateral::Bool, excluded_primitives::Vector{String}) =
    ProposalConfig(domain, input_vars, max_params, include_nested, include_bilateral, excluded_primitives, input_vars)

theta_node(i::Int)   = Dict{String,Any}("theta" => "theta_$i")
sym_node(s::String)  = Dict{String,Any}("sym" => s)
op_node(op::String, args::Vector) = Dict{String,Any}("op" => op, "args" => args)
num_node(v::Real)    = Dict{String,Any}("num" => Float64(v))

function build_ratio_nodes(vars::Vector{String}, theta_idx::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{Dict{String,Any}}
    results = Dict{String,Any}[]
    
    # 1. Dimensionless Buckingham Pi ratios (already covers degree 1 and degree 2)
    ratios = enumerate_dimensionless_ratios(vars, 2; data_vars=data_vars)
    for ratio_expr in ratios
        push!(results, op_node("mul", [theta_node(theta_idx), ratio_expr]))
    end
    
    # 2. Single variables (scaled by theta, e.g. theta * r or r / theta)
    # Only scale dynamic variables that genuinely vary across data points
    vars_to_scale = data_vars === nothing ? vars : filter(v -> v in data_vars, vars)
    for v in vars_to_scale
        push!(results, op_node("mul", [theta_node(theta_idx), sym_node(v)]))
        push!(results, op_node("div", [sym_node(v), theta_node(theta_idx)]))
    end
    
    return results
end

function build_primitive_application(prim_sym::Symbol, u_node::Dict)::Dict{String,Any}
    op_node(lowercase(string(prim_sym)), [u_node])
end

# Patterns
function pattern_singleton(prim::ADCDPrimitive, vars::Vector{String}, t::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t; data_vars=data_vars)
        delta = op_node("mul", [theta_node(t + 1), build_primitive_application(prim.name, u)])
        push!(proposals, CorrectionProposal(delta, [prim.name], :singleton, 2, "$(prim.name)(theta*u)"))
    end
    return proposals
end

function pattern_additive(p1::ADCDPrimitive, p2::ADCDPrimitive, vars::Vector{String}, t::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t; data_vars=data_vars)
        t1 = op_node("mul", [theta_node(t + 1), build_primitive_application(p1.name, u)])
        t2 = op_node("mul", [theta_node(t + 2), build_primitive_application(p2.name, u)])
        push!(proposals, CorrectionProposal(op_node("add", [t1, t2]), [p1.name, p2.name], :additive, 3, "$(p1.name)+$(p2.name)"))
    end
    return proposals
end

function pattern_multiplicative(p1::ADCDPrimitive, p2::ADCDPrimitive, vars::Vector{String}, t::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t; data_vars=data_vars)
        d1 = build_primitive_application(p1.name, u)
        d2 = build_primitive_application(p2.name, u)
        inner = op_node("add", [num_node(1.0), op_node("mul", [theta_node(t + 2), d2])])
        push!(proposals, CorrectionProposal(op_node("mul", [theta_node(t + 1), d1, inner]), [p1.name, p2.name], :multiplicative, 3, "$(p1.name)*(1+$(p2.name))"))
    end
    return proposals
end

function pattern_nested(outer::ADCDPrimitive, inner::ADCDPrimitive, vars::Vector{String}, t::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t; data_vars=data_vars)
        nested = build_primitive_application(outer.name, build_primitive_application(inner.name, u))
        push!(proposals, CorrectionProposal(op_node("mul", [theta_node(t + 1), nested]), [outer.name, inner.name], :nested, 2, "$(outer.name)($(inner.name)(u))"))
    end
    return proposals
end

function pattern_bilateral(p1::ADCDPrimitive, p2::ADCDPrimitive, vars::Vector{String}, t::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    length(vars) < 3 && return proposals

    for i in 1:(length(vars)-1)
        vars1 = vars[1:i]
        vars2 = vars[(i+1):end]
        u1_list = build_ratio_nodes(vars1, t; data_vars=data_vars)
        u2_list = build_ratio_nodes(vars2, t + 1; data_vars=data_vars)
        
        for u1 in u1_list, u2 in u2_list
            d1 = build_primitive_application(p1.name, u1)
            d2 = build_primitive_application(p2.name, u2)
            delta = op_node("mul", [theta_node(t + 2), d1, d2])
            push!(proposals, CorrectionProposal(delta, [p1.name, p2.name], :bilateral, 3, "$(p1.name)(u1)*$(p2.name)(u2)"))
        end
    end
    return proposals
end

function pattern_ratio_correction(p::ADCDPrimitive, vars::Vector{String}, t::Int; data_vars::Union{Vector{String},Nothing}=nothing)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t; data_vars=data_vars)
        sqrt_u = op_node("sqrt", [u])
        d = build_primitive_application(p.name, sqrt_u)
        delta = op_node("mul", [theta_node(t + 1), op_node("div", [d, op_node("add", [num_node(1.0), d])])])
        push!(proposals, CorrectionProposal(delta, [p.name], :ratio_correction, 2, "$(p.name)(sqrt(u))/(1+$(p.name)(sqrt(u)))"))
    end
    return proposals
end

function propose_corrections(config::ProposalConfig)::Vector{CorrectionProposal}
    all_prims = primitives_for_domain(config.domain)
    prims = filter(p -> !(string(p.name) in config.excluded_primitives), all_prims)
    vars  = config.input_vars
    d_vars = config.data_vars
    max_p = config.max_params
    proposals = CorrectionProposal[]
    t = 0

    for p in prims
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_singleton(p, vars, t; data_vars=d_vars)))
    end

    for i in 1:length(prims), j in 1:length(prims)
        i == j && continue
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_additive(prims[i], prims[j], vars, t; data_vars=d_vars)))
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_multiplicative(prims[i], prims[j], vars, t; data_vars=d_vars)))
    end

    if config.include_nested
        for outer in prims, inner in prims
            outer.name == inner.name && continue
            append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_nested(outer, inner, vars, t; data_vars=d_vars)))
        end
    end

    if config.include_bilateral
        for i in 1:length(prims), j in 1:length(prims)
            append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_bilateral(prims[i], prims[j], vars, t; data_vars=d_vars)))
        end
    end

    for p in prims
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_ratio_correction(p, vars, t; data_vars=d_vars)))
    end

    sort!(proposals, by=p->p.n_params)

    seen = Set{String}()
    unique_proposals = CorrectionProposal[]
    for p in proposals
        expr_str = JSON3.write(p.expr)
        expr_str in seen && continue
        push!(seen, expr_str)
        push!(unique_proposals, p)
    end
    return unique_proposals
end

end  # module CorrectionProposer
