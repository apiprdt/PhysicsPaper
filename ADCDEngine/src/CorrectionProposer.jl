# ADCD Engine: CorrectionProposer
module CorrectionProposer

using ..ADCDDimensions
using ..PrimitiveRegistry
using JSON3

export CorrectionProposal, ProposalConfig, propose_corrections
struct CorrectionProposal
    expr       ::Dict{String,Any}   # ADCD canonical expression node
    primitives ::Vector{Symbol}     # which primitives are used
    pattern    ::Symbol             # composition pattern used
    n_params   ::Int                # number of free theta parameters
    description::String
end

"""
    ProposalConfig

Controls the grammar expansion.
"""
struct ProposalConfig
    domain         ::String
    input_vars     ::Vector{String}  # physical variable names (e.g. ["v","c"])
    max_params     ::Int             # max theta_i parameters
    include_nested ::Bool            # enable nested D_i(D_j(u)) patterns
    include_bilateral::Bool          # enable two-variable D_i(u1)*D_j(u2) patterns
end

ProposalConfig(domain::String, input_vars::Vector{String}) =
    ProposalConfig(domain, input_vars, 3, true, true)

# ---------------------------------------------------------------------------
# Expression node builders (ADCD canonical format)
# ---------------------------------------------------------------------------

function theta_node(i::Int)::Dict{String,Any}
    Dict("theta" => "theta_$i")
end

function sym_node(s::String)::Dict{String,Any}
    Dict("sym" => s)
end

function op_node(op::String, args::Vector)::Dict{String,Any}
    Dict("op" => op, "args" => args)
end

function num_node(v::Real)::Dict{String,Any}
    Dict("num" => Float64(v))
end

"""
    build_ratio_node(vars, theta_idx_start) -> (node, n_params_used)

Build the dimensionless ratio u = product(var^exp) scaled by a theta.
For single variable: u = theta * var / ref_var
For Buckingham pi: use the null-space ratio directly.
"""
function build_ratio_nodes(
    vars::Vector{String},
    theta_idx::Int,
    power::Int=1
)::Vector{Dict{String,Any}}
    
    ratios = enumerate_dimensionless_ratios(vars, 2)
    if isempty(ratios)
        # Fallback
        var_node = power == 1 ? sym_node(vars[1]) :
                   op_node("pow", [sym_node(vars[1]), num_node(power)])
        u = op_node("mul", [theta_node(theta_idx), var_node])
        return [u]
    end

    results = Dict{String,Any}[]
    for ratio_expr in ratios
        if power != 1
            ratio_expr = op_node("pow", [ratio_expr, num_node(power)])
        end
        u = op_node("mul", [theta_node(theta_idx), ratio_expr])
        push!(results, u)
    end
    
    return results
end

"""
    build_primitive_application(prim_sym, u_node) -> Dict

Apply primitive D_prim to argument u as an expression node.
Primitives are represented as function call nodes.
"""
function build_primitive_application(prim_sym::Symbol, u_node::Dict)::Dict{String,Any}
    op_name = lowercase(string(prim_sym))  # :D_lor -> "d_lor"
    op_node(op_name, [u_node])
end

# ---------------------------------------------------------------------------
# Grammar patterns (6 patterns, expanded from Python's 3)
# ---------------------------------------------------------------------------

"""
    pattern_singleton(prim, vars, theta_start) -> CorrectionProposal

Pattern 1 (exists in Python): Δ = θ₀ · D_i(θ₁·u)
Simple single-primitive correction.
"""
function pattern_singleton(
    prim::ADCDPrimitive,
    vars::Vector{String},
    theta_start::Int
)::Vector{CorrectionProposal}
    u_nodes = build_ratio_nodes(vars, theta_start)
    proposals = CorrectionProposal[]
    for u_node in u_nodes
        d_node = build_primitive_application(prim.name, u_node)
        theta_scale = theta_node(theta_start + 1)
        delta = op_node("mul", [theta_scale, d_node])
        push!(proposals, CorrectionProposal(
            delta,
            [prim.name],
            :singleton,
            2,
            "$(prim.name)(theta*u)"
        ))
    end
    return proposals
end

"""
    pattern_additive(p1, p2, vars, theta_start) -> CorrectionProposal

Pattern 2 (exists in Python): Δ = θ₀·D_a(u) + θ₁·D_b(u)
"""
function pattern_additive(
    p1::ADCDPrimitive,
    p2::ADCDPrimitive,
    vars::Vector{String},
    theta_start::Int
)::Vector{CorrectionProposal}
    u_nodes = build_ratio_nodes(vars, theta_start)
    proposals = CorrectionProposal[]
    for u_node in u_nodes
        d1 = build_primitive_application(p1.name, u_node)
        d2 = build_primitive_application(p2.name, u_node)
        term1 = op_node("mul", [theta_node(theta_start+1),   d1])
        term2 = op_node("mul", [theta_node(theta_start+2), d2])
        delta = op_node("add", [term1, term2])
        push!(proposals, CorrectionProposal(delta, [p1.name, p2.name], :additive, 3, "$(p1.name)+$(p2.name)"))
    end
    return proposals
end

"""
    pattern_multiplicative(p1, p2, vars, theta_start) -> CorrectionProposal

Pattern 3 (exists in Python): Δ = θ₀·D_a(u)·(1 + θ₁·D_b(u))
"""
function pattern_multiplicative(
    p1::ADCDPrimitive,
    p2::ADCDPrimitive,
    vars::Vector{String},
    theta_start::Int
)::Vector{CorrectionProposal}
    u_nodes = build_ratio_nodes(vars, theta_start)
    proposals = CorrectionProposal[]
    for u_node in u_nodes
        d1 = build_primitive_application(p1.name, u_node)
        d2 = build_primitive_application(p2.name, u_node)
        inner = op_node("add", [num_node(1.0),
                                 op_node("mul",[theta_node(theta_start+2), d2])])
        delta = op_node("mul", [theta_node(theta_start+1), d1, inner])
        push!(proposals, CorrectionProposal(delta, [p1.name, p2.name], :multiplicative, 3, "$(p1.name)*(1+$(p2.name))"))
    end
    return proposals
end

"""
    pattern_nested(outer, inner_prim, vars, theta_start) -> CorrectionProposal

Pattern 4 (NEW): Δ = θ₀·D_outer(D_inner(θ₁·u))

This is the CRITICAL new pattern that enables MOND/RAR reconstruction.
D_exp(D_sqrt_inv(u)) ≈ exp(-sqrt(u)) which is exactly the McGaugh RAR form.
The Python grammar_proposer_v3.py CANNOT generate this pattern.
"""
function pattern_nested(
    outer::ADCDPrimitive,
    inner::ADCDPrimitive,
    vars::Vector{String},
    theta_start::Int
)::Vector{CorrectionProposal}
    u_nodes = build_ratio_nodes(vars, theta_start)
    proposals = CorrectionProposal[]
    for u_node in u_nodes
        # inner applied first: D_inner(u)
        inner_applied = build_primitive_application(inner.name, u_node)
        # outer applied to result: D_outer(D_inner(u))
        nested = build_primitive_application(outer.name, inner_applied)
        delta = op_node("mul", [theta_node(theta_start+1), nested])
        push!(proposals, CorrectionProposal(delta, [outer.name, inner.name], :nested, 2, "$(outer.name)($(inner.name)(u))"))
    end
    return proposals
end

"""
    pattern_bilateral(p1, p2, vars1, vars2, theta_start) -> CorrectionProposal

Pattern 5 (NEW): Δ = θ₀·D_a(u₁)·D_b(u₂) where u₁ and u₂ use different vars.

Enables corrections that depend on two independent physical ratios simultaneously.
"""
function pattern_bilateral(
    p1::ADCDPrimitive,
    p2::ADCDPrimitive,
    vars1::Vector{String},
    vars2::Vector{String},
    theta_start::Int
)::Vector{CorrectionProposal}
    u1_nodes = build_ratio_nodes(vars1, theta_start)
    u2_nodes = build_ratio_nodes(vars2, theta_start+1)
    proposals = CorrectionProposal[]
    for u1_node in u1_nodes, u2_node in u2_nodes
        d1 = build_primitive_application(p1.name, u1_node)
        d2 = build_primitive_application(p2.name, u2_node)
        delta = op_node("mul", [theta_node(theta_start+2), d1, d2])
        push!(proposals, CorrectionProposal(delta, [p1.name, p2.name], :bilateral, 3, "$(p1.name)(u1)*$(p2.name)(u2)"))
    end
    return proposals
end

"""
    pattern_ratio_correction(p, vars, theta_start) -> CorrectionProposal

Pattern 6 (NEW): Δ = D_i(sqrt(u))/(1 + D_i(sqrt(u)))
Modular ratio form, useful when correction saturates. Especially useful
for MOND where mu(x) = x/(1+x) is the standard interpolating function.
"""
function pattern_ratio_correction(
    p::ADCDPrimitive,
    vars::Vector{String},
    theta_start::Int
)::Vector{CorrectionProposal}
    u_nodes = build_ratio_nodes(vars, theta_start)
    proposals = CorrectionProposal[]
    for u_node in u_nodes
        sqrt_u = op_node("sqrt", [u_node])
        d = build_primitive_application(p.name, sqrt_u)
        denom = op_node("add", [num_node(1.0), d])
        delta = op_node("mul", [theta_node(theta_start+1),
                                 op_node("div", [d, denom])])
        push!(proposals, CorrectionProposal(delta, [p.name], :ratio_correction, 2, "$(p.name)(sqrt(u))/(1+$(p.name)(sqrt(u)))"))
    end
    return proposals
end

# ---------------------------------------------------------------------------
# Main proposal generator
# ---------------------------------------------------------------------------

"""
    propose_corrections(config) -> Vector{CorrectionProposal}

Generate all candidate correction terms for a given domain and variable set.
Returns proposals sorted by increasing number of free parameters.
"""
function propose_corrections(config::ProposalConfig)::Vector{CorrectionProposal}
    prims = primitives_for_domain(config.domain)
    vars  = config.input_vars
    max_p = config.max_params
    proposals = CorrectionProposal[]
    t = 0  # theta counter start

    # Pattern 1: singleton
    for p in prims
        props = pattern_singleton(p, vars, t)
        append!(proposals, filter(prop -> prop.n_params <= max_p, props))
    end

    # Patterns 2 & 3: pairs
    for i in 1:length(prims), j in 1:length(prims)
        i==j && continue
        props2 = pattern_additive(prims[i], prims[j], vars, t)
        append!(proposals, filter(prop -> prop.n_params <= max_p, props2))
        props3 = pattern_multiplicative(prims[i], prims[j], vars, t)
        append!(proposals, filter(prop -> prop.n_params <= max_p, props3))
    end

    # Pattern 4: nested (NEW) — only if enabled
    if config.include_nested
        for outer in prims, inner in prims
            outer.name == inner.name && continue
            props = pattern_nested(outer, inner, vars, t)
            append!(proposals, filter(prop -> prop.n_params <= max_p, props))
        end
    end

    # Pattern 5: bilateral (NEW) — only if we have >=2 variables and enabled
    if config.include_bilateral && length(vars) >= 2
        vars1 = vars[1:1]; vars2 = vars[2:end]
        for i in 1:length(prims), j in 1:length(prims)
            props = pattern_bilateral(prims[i], prims[j], vars1, vars2, t)
            append!(proposals, filter(prop -> prop.n_params <= max_p, props))
        end
    end

    # Pattern 6: ratio (NEW) — MOND reconstruction
    for p in prims
        props = pattern_ratio_correction(p, vars, t)
        append!(proposals, filter(prop -> prop.n_params <= max_p, props))
    end

    # Sort by n_params (prefer parsimonious)
    sort!(proposals, by=p->p.n_params)

    # Deduplicate by canonical JSON string of expression
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
