# ADCD Engine: ADCDDimensions (Hardened & Unified)
module ADCDDimensions

using LinearAlgebra
using JSON3

export PhysicalDimension, DIMENSION_REGISTRY, TARGET_DIMENSION_MAP
export DimResult, DIM_MISMATCH, DIM_UNKNOWN_SYMBOL, DIM_UNSUPPORTED_OP, DIM_TRANSCENDENTAL_ARG
export is_dimensionless, infer_dim, verify_dimension, enumerate_dimensionless_ratios

struct PhysicalDimension
    M::Int8; L::Int8; T::Int8; Th::Int8; Q::Int8
end

Base.:(==)(a::PhysicalDimension, b::PhysicalDimension) =
    a.M==b.M && a.L==b.L && a.T==b.T && a.Th==b.Th && a.Q==b.Q
Base.zero(::Type{PhysicalDimension}) = PhysicalDimension(0,0,0,0,0)
Base.:(+)(a::PhysicalDimension, b::PhysicalDimension) =
    PhysicalDimension(a.M+b.M, a.L+b.L, a.T+b.T, a.Th+b.Th, a.Q+b.Q)
Base.:(-)(a::PhysicalDimension, b::PhysicalDimension) =
    PhysicalDimension(a.M-b.M, a.L-b.L, a.T-b.T, a.Th-b.Th, a.Q-b.Q)
Base.:(*)(n::Int, d::PhysicalDimension) =
    PhysicalDimension(Int8(n*d.M), Int8(n*d.L), Int8(n*d.T), Int8(n*d.Th), Int8(n*d.Q))

is_dimensionless(d::PhysicalDimension) = d == zero(PhysicalDimension)

@enum DimResult DIM_MISMATCH DIM_UNKNOWN_SYMBOL DIM_UNSUPPORTED_OP DIM_TRANSCENDENTAL_ARG

# Physical dimension registry (5D SI: M=mass, L=length, T=time, Th=temperature, Q=charge)
const DIMENSION_REGISTRY = Dict{Symbol,PhysicalDimension}(
    # Konstanta Elektromagnetik Tambahan
    :k_e => PhysicalDimension(1, 3, -2, 0, -2), # Coulomb constant [N m^2 / C^2]
    
    # Konstanta Termodinamika Tambahan
    :nR  => PhysicalDimension(1, 2, -2, -1, 0), # Moles * Gas constant [J / K]

    # Base SI
    :M  => PhysicalDimension(1,0,0,0,0),   # mass [kg]
    :L  => PhysicalDimension(0,1,0,0,0),   # length [m]
    :T_time => PhysicalDimension(0,0,1,0,0), # time [s] (distinct symbol)
    :Th => PhysicalDimension(0,0,0,1,0),   # temperature [K]
    :Q  => PhysicalDimension(0,0,0,0,1),   # charge [C]
    # Kinematics
    :v  => PhysicalDimension(0,1,-1,0,0),  # velocity [m/s]
    :c  => PhysicalDimension(0,1,-1,0,0),  # speed of light [m/s]
    :a  => PhysicalDimension(0,1,-2,0,0),  # acceleration [m/s^2]
    :r  => PhysicalDimension(0,1,0,0,0),   # radius [m]
    :x  => PhysicalDimension(0,1,0,0,0),   # position [m]
    :d  => PhysicalDimension(0,1,0,0,0),   # distance [m]
    # Mechanics
    :m  => PhysicalDimension(1,0,0,0,0),   # mass [kg]
    :F  => PhysicalDimension(1,1,-2,0,0),  # force [N]
    :E  => PhysicalDimension(1,2,-2,0,0),  # energy [J]
    :p  => PhysicalDimension(1,1,-1,0,0),  # momentum [kg m/s]
    :k  => PhysicalDimension(1,0,-2,0,0),  # spring constant
    :b  => PhysicalDimension(1,-1,0,0,0),  # drag b (F = b v^2) -> M/L
    :rho => PhysicalDimension(1,-3,0,0,0), # density
    # Gravity
    :G  => PhysicalDimension(-1,3,-2,0,0), # gravitational constant
    :g  => PhysicalDimension(0,1,-2,0,0),  # surface gravity
    :g_bar => PhysicalDimension(0,1,-2,0,0), # observed acceleration (RAR)
    :g_bar_newton => PhysicalDimension(0,1,-2,0,0), # Newtonian accel (RAR)
    :a0 => PhysicalDimension(0,1,-2,0,0),  # MOND acceleration scale
    # E&M
    :q  => PhysicalDimension(0,0,0,0,1),   # charge [C]
    :q1 => PhysicalDimension(0,0,0,0,1),
    :q2 => PhysicalDimension(0,0,0,0,1),
    :epsilon => PhysicalDimension(0,1,0,0,0), # screening length [m]
    # Thermodynamics
    :S  => PhysicalDimension(1,2,-2,-1,0), # entropy [J/K]
    :S_i => PhysicalDimension(1,2,-2,-1,0),
    :k_B => PhysicalDimension(1,2,-2,-1,0),# Boltzmann constant
    :T  => PhysicalDimension(0,0,0,1,0),   # Temperature [K]
    :T_temp => PhysicalDimension(0,0,0,1,0),
    :n  => PhysicalDimension(0,0,0,0,0),   # moles / dimensionless
    :sigma => PhysicalDimension(1,0,-3,-4,0), # Stefan-Boltzmann
    # Lengths and Areas
    :l  => PhysicalDimension(0,1,0,0,0),
    :A  => PhysicalDimension(0,2,0,0,0),
    :V_i => PhysicalDimension(0,3,0,0,0),
    :dV  => PhysicalDimension(0,3,0,0,0),
    :V   => PhysicalDimension(0,3,0,0,0),
    # Time symbols
    :tau => PhysicalDimension(0,0,1,0,0),
    :t  => PhysicalDimension(0,0,1,0,0),
    :t_0 => PhysicalDimension(0,0,1,0,0),
    # Cosmological & Kinematics
    :H  => PhysicalDimension(0,0,-1,0,0),
    :z  => PhysicalDimension(0,0,0,0,0),
    :V_inf => PhysicalDimension(0,1,-1,0,0),
    :w  => PhysicalDimension(0,0,-1,0,0),
    :theta => PhysicalDimension(0,0,0,0,0),
)

const TARGET_DIMENSION_MAP = Dict{String,PhysicalDimension}(
    "dimensionless" => PhysicalDimension(0,0,0,0,0),
    "velocity"      => PhysicalDimension(0,1,-1,0,0),
    "acceleration"  => PhysicalDimension(0,1,-2,0,0),
    "length"        => PhysicalDimension(0,1,0,0,0),
    "mass"          => PhysicalDimension(1,0,0,0,0),
    "time"          => PhysicalDimension(0,0,1,0,0),
    "energy"        => PhysicalDimension(1,2,-2,0,0),
    "force"         => PhysicalDimension(1,1,-2,0,0),
    "entropy"       => PhysicalDimension(1,2,-2,-1,0),
)

function infer_dim(
    node::AbstractDict,
    registry::Dict{Symbol,PhysicalDimension}=DIMENSION_REGISTRY
)::Union{PhysicalDimension, DimResult}

    haskey(node, "sym")   && return get(registry, Symbol(node["sym"]), DIM_UNKNOWN_SYMBOL)
    haskey(node, "theta") && return zero(PhysicalDimension)
    haskey(node, "num")   && return zero(PhysicalDimension)

    op   = get(node, "op", "")
    args = get(node, "args", [])

    if op in ("add", "sub")
        length(args) < 2 && return DIM_UNSUPPORTED_OP
        d1 = infer_dim(args[1], registry)
        d1 isa DimResult && return d1
        d2 = infer_dim(args[2], registry)
        d2 isa DimResult && return d2
        d1 == d2 || return DIM_MISMATCH
        return d1
    elseif op == "mul"
        length(args) < 2 && return DIM_UNSUPPORTED_OP
        d1 = infer_dim(args[1], registry)
        d1 isa DimResult && return d1
        d2 = infer_dim(args[2], registry)
        d2 isa DimResult && return d2
        return d1 + d2
    elseif op == "div"
        length(args) < 2 && return DIM_UNSUPPORTED_OP
        d1 = infer_dim(args[1], registry)
        d1 isa DimResult && return d1
        d2 = infer_dim(args[2], registry)
        d2 isa DimResult && return d2
        return d1 - d2
    elseif op == "pow"
        length(args) < 2 && return DIM_UNSUPPORTED_OP
        d_base = infer_dim(args[1], registry)
        d_base isa DimResult && return d_base
        
        # Pangkat dari bilangan tak berdimensi selalu tak berdimensi
        if is_dimensionless(d_base)
            return zero(PhysicalDimension)
        end
        
        # Jika basis memiliki dimensi, eksponen wajib berupa integer
        if haskey(args[2], "num")
            num_val = args[2]["num"]
            if isinteger(num_val)
                n = round(Int, num_val)
                return n * d_base
            end
        end
        return DIM_UNSUPPORTED_OP
    elseif op == "sqrt"
        length(args) < 1 && return DIM_UNSUPPORTED_OP
        d_arg = infer_dim(args[1], registry)
        d_arg isa DimResult && return d_arg
        all(x -> x % 2 == 0, (d_arg.M, d_arg.L, d_arg.T, d_arg.Th, d_arg.Q)) || return DIM_UNSUPPORTED_OP
        return PhysicalDimension(div(d_arg.M, 2), div(d_arg.L, 2), div(d_arg.T, 2), div(d_arg.Th, 2), div(d_arg.Q, 2))
    elseif op == "neg" || lowercase(op) == "abs"
        length(args) < 1 && return DIM_UNSUPPORTED_OP
        return infer_dim(args[1], registry)
    elseif lowercase(op) == "max" || lowercase(op) == "min"
        length(args) < 2 && return DIM_UNSUPPORTED_OP
        d1 = infer_dim(args[1], registry)
        d1 isa DimResult && return d1
        d2 = infer_dim(args[2], registry)
        d2 isa DimResult && return d2
        d1 == d2 || return DIM_MISMATCH
        return d1
    elseif op in ("exp","log","sin","cos","tan","tanh","sinh","cosh",
                  "d_lor","d_exp","d_rat","d_pow","d_log","d_sat",
                  "d_sqrt_inv","d_tanh_sq","d_osc","d_nested_mond","d_rar")
        length(args) < 1 && return DIM_UNSUPPORTED_OP
        arg_node = args[1]
        d_arg = infer_dim(arg_node, registry)
        d_arg isa DimResult && return d_arg
        
        # Jika argumennya sudah dimensionless, maka aman
        is_dimensionless(d_arg) && return zero(PhysicalDimension)
        
        # PENTING: Jika argumennya dikalikan/dibagi dengan theta, 
        # theta bertindak sebagai penghapus dimensi (free-scale parameter).
        function has_theta(n::AbstractDict)
            haskey(n, "theta") && return true
            if haskey(n, "args")
                return any(has_theta(a) for a in n["args"])
            end
            return false
        end
        
        has_theta(arg_node) && return zero(PhysicalDimension)
        
        return DIM_TRANSCENDENTAL_ARG
    else
        return DIM_UNSUPPORTED_OP
    end
end

function verify_dimension(
    expr_node::AbstractDict,
    target_dim_name::String,
    registry::Dict{Symbol,PhysicalDimension}=DIMENSION_REGISTRY
)::Bool
    if haskey(TARGET_DIMENSION_MAP, target_dim_name)
        target = TARGET_DIMENSION_MAP[target_dim_name]
    elseif haskey(registry, Symbol(target_dim_name))
        target = registry[Symbol(target_dim_name)]
    else
        return false  # Fail-closed: tolak target dimensi tak dikenal
    end
    result = infer_dim(expr_node, registry)
    result isa DimResult && return false
    return result == target
end

_to_dict(x) = x
_to_dict(x::AbstractDict) = Dict{String,Any}(string(k) => _to_dict(v) for (k, v) in x)
_to_dict(x::AbstractVector) = [_to_dict(v) for v in x]

function verify_dimension(json_str::String, target_dim_name::String)::Bool
    node = _to_dict(JSON3.read(json_str))
    return verify_dimension(node, target_dim_name)
end

function enumerate_dimensionless_ratios(
    symbols::Vector{String},
    max_degree::Int=2,
    registry::Dict{Symbol,PhysicalDimension}=DIMENSION_REGISTRY
)::Vector{Dict{String,Any}}

    valid = unique(filter(s -> haskey(registry, Symbol(s)), symbols))
    isempty(valid) && return Dict{String,Any}[]
    n = length(valid)

    A = Matrix{Float64}(undef, 5, n)
    for (j, s) in enumerate(valid)
        d = registry[Symbol(s)]
        A[:, j] = Float64[d.M, d.L, d.T, d.Th, d.Q]
    end
    NS = nullspace(A, atol=1e-8)
    size(NS, 2) == 0 && return Dict{String,Any}[]

    basis_vectors = Vector{Int}[]
    for col in 1:size(NS, 2)
        v = NS[:, col]
        nonzeros = filter(x -> abs(x) > 1e-6, v)
        isempty(nonzeros) && continue
        min_val = minimum(abs.(nonzeros))
        v_scaled = v ./ min_val
        rats = [abs(x) < 1e-6 ? 0//1 : rationalize(x, tol=1e-4) for x in v_scaled]
        denoms = [denominator(r) for r in rats]
        lcm_denom = isempty(denoms) ? 1 : foldl(lcm, denoms)
        int_vec = Int[round(Int, (numerator(r) * (lcm_denom  div  denominator(r)))) for r in rats]
        g = foldl(gcd, int_vec)
        if g > 1
            int_vec = int_vec . div  g
        end
        push!(basis_vectors, int_vec)
    end

    isempty(basis_vectors) && return Dict{String,Any}[]

    results = Set{Vector{Int}}()
    k = length(basis_vectors)
    coef_range = -max_degree:max_degree
    for coeffs in Iterators.product(fill(coef_range, k)...)
        all(c == 0 for c in coeffs) && continue
        int_exps = sum(collect(coeffs)[i] .* basis_vectors[i] for i in 1:k)
        any(abs(e) > max_degree for e in int_exps) && continue
        all(e == 0 for e in int_exps) && continue
        
        push!(results, int_exps)
    end

    exprs = Dict{String,Any}[]
    for exps in sort(collect(results))
        factors = Dict{String,Any}[]
        for (s, e) in zip(valid, exps)
            e == 0 && continue
            e == 1 ? push!(factors, Dict("sym" => s)) :
                     push!(factors, Dict("op" => "pow", "args" => [Dict("sym" => s), Dict("num" => e)]))
        end
        isempty(factors) && continue
        expr = length(factors) == 1 ? factors[1] : Dict("op" => "mul", "args" => factors)
        push!(exprs, expr)
    end
    return exprs
end

end  # module ADCDDimensions
