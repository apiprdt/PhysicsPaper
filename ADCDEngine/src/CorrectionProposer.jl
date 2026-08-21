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
    domain          ::String
    input_vars      ::Vector{String}
    max_params      ::Int
    include_nested  ::Bool
    include_bilateral::Bool
end

ProposalConfig(domain::String, input_vars::Vector{String}) =
    ProposalConfig(domain, input_vars, 3, true, true)

theta_node(i::Int)   = Dict{String,Any}("theta" => "theta_$i")
sym_node(s::String)  = Dict{String,Any}("sym" => s)
op_node(op::String, args::Vector) = Dict{String,Any}("op" => op, "args" => args)
num_node(v::Real)    = Dict{String,Any}("num" => Float64(v))

function build_ratio_nodes(vars::Vector{String}, theta_idx::Int, power::Int=1)::Vector{Dict{String,Any}}
    ratios = enumerate_dimensionless_ratios(vars, 2)
    # Jika tidak ada rasio dimensionless murni, kembalikan kosong (jangan fallback ke variabel berdimensi)
    isempty(ratios) && return Dict{String,Any}[]

    results = Dict{String,Any}[]
    for ratio_expr in ratios
        r_node = (power != 1) ? op_node("pow", [ratio_expr, num_node(power)]) : ratio_expr
        push!(results, op_node("mul", [theta_node(theta_idx), r_node]))
    end
    return results
end

function build_primitive_application(prim_sym::Symbol, u_node::Dict)::Dict{String,Any}
    op_node(lowercase(string(prim_sym)), [u_node])
end

# Patterns
function pattern_singleton(prim::ADCDPrimitive, vars::Vector{String}, t::Int)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t)
        delta = op_node("mul", [theta_node(t + 1), build_primitive_application(prim.name, u)])
        push!(proposals, CorrectionProposal(delta, [prim.name], :singleton, 2, "$(prim.name)(theta*u)"))
    end
    return proposals
end

function pattern_additive(p1::ADCDPrimitive, p2::ADCDPrimitive, vars::Vector{String}, t::Int)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t)
        t1 = op_node("mul", [theta_node(t + 1), build_primitive_application(p1.name, u)])
        t2 = op_node("mul", [theta_node(t + 2), build_primitive_application(p2.name, u)])
        push!(proposals, CorrectionProposal(op_node("add", [t1, t2]), [p1.name, p2.name], :additive, 3, "$(p1.name)+$(p2.name)"))
    end
    return proposals
end

function pattern_multiplicative(p1::ADCDPrimitive, p2::ADCDPrimitive, vars::Vector{String}, t::Int)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t)
        d1 = build_primitive_application(p1.name, u)
        d2 = build_primitive_application(p2.name, u)
        inner = op_node("add", [num_node(1.0), op_node("mul", [theta_node(t + 2), d2])])
        push!(proposals, CorrectionProposal(op_node("mul", [theta_node(t + 1), d1, inner]), [p1.name, p2.name], :multiplicative, 3, "$(p1.name)*(1+$(p2.name))"))
    end
    return proposals
end

function pattern_nested(outer::ADCDPrimitive, inner::ADCDPrimitive, vars::Vector{String}, t::Int)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t)
        nested = build_primitive_application(outer.name, build_primitive_application(inner.name, u))
        push!(proposals, CorrectionProposal(op_node("mul", [theta_node(t + 1), nested]), [outer.name, inner.name], :nested, 2, "$(outer.name)($(inner.name)(u))"))
    end
    return proposals
end

function pattern_bilateral(p1::ADCDPrimitive, p2::ADCDPrimitive, vars::Vector{String}, t::Int)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    length(vars) < 3 && return proposals # Butuh minimal 3 variabel fisik agar bisa terbagi menjadi 2 rasio independen

    for i in 1:(length(vars)-1)
        vars1 = vars[1:i]
        vars2 = vars[(i+1):end]
        u1_list = build_ratio_nodes(vars1, t)
        u2_list = build_ratio_nodes(vars2, t + 1)
        
        for u1 in u1_list, u2 in u2_list
            d1 = build_primitive_application(p1.name, u1)
            d2 = build_primitive_application(p2.name, u2)
            delta = op_node("mul", [theta_node(t + 2), d1, d2])
            push!(proposals, CorrectionProposal(delta, [p1.name, p2.name], :bilateral, 3, "$(p1.name)(u1)*$(p2.name)(u2)"))
        end
    end
    return proposals
end

function pattern_ratio_correction(p::ADCDPrimitive, vars::Vector{String}, t::Int)::Vector{CorrectionProposal}
    proposals = CorrectionProposal[]
    for u in build_ratio_nodes(vars, t)
        sqrt_u = op_node("sqrt", [u])
        d = build_primitive_application(p.name, sqrt_u)
        delta = op_node("mul", [theta_node(t + 1), op_node("div", [d, op_node("add", [num_node(1.0), d])])])
        push!(proposals, CorrectionProposal(delta, [p.name], :ratio_correction, 2, "$(p.name)(sqrt(u))/(1+$(p.name)(sqrt(u)))"))
    end
    return proposals
end

function propose_corrections(config::ProposalConfig)::Vector{CorrectionProposal}
    prims = primitives_for_domain(config.domain)
    vars  = config.input_vars
    max_p = config.max_params
    proposals = CorrectionProposal[]
    t = 0

    for p in prims
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_singleton(p, vars, t)))
    end

    for i in 1:length(prims), j in 1:length(prims)
        i == j && continue
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_additive(prims[i], prims[j], vars, t)))
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_multiplicative(prims[i], prims[j], vars, t)))
    end

    if config.include_nested
        for outer in prims, inner in prims
            outer.name == inner.name && continue
            append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_nested(outer, inner, vars, t)))
        end
    end

    if config.include_bilateral
        for i in 1:length(prims), j in 1:length(prims)
            append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_bilateral(prims[i], prims[j], vars, t)))
        end
    end

    for p in prims
        append!(proposals, filter(prop -> prop.n_params <= max_p, pattern_ratio_correction(p, vars, t)))
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
