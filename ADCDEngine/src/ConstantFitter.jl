# ADCD Engine: ConstantFitter
module ConstantFitter

using Optim
using Random

using ..ADCDDimensions
using ..CorrectionProposer

export FitResult, fit_constants, evaluate_expr
struct FitResult
    theta     ::Vector{Float64}   # fitted parameters
    nmse      ::Float64           # normalized mean squared error
    likelihood::Float64           # Gaussian log-likelihood
    converged ::Bool
    n_restarts::Int               # how many restarts were run
    n_params  ::Int
    error_msg ::Union{String,Nothing}
    residuals ::Vector{Float64}   # residuals in delta space (for BIC)
end

"""
    evaluate_expr(expr_node, vars_data, constants, theta) -> Vector{Float64}

Numerically evaluate an ADCD expression dict given:
  - vars_data: Dict of variable name -> Float64 vector
  - constants: Dict of known constants (e.g. "c" -> 3e8)
  - theta: vector of free parameter values (theta_0, theta_1, ...)

Returns the evaluated vector or throws on error.
"""
function evaluate_expr(
    expr_node::AbstractDict,
    vars_data ::Dict{String,Vector{Float64}},
    constants ::Dict{String,Float64},
    theta     ::Vector{Float64}
)::Vector{Float64}
    _eval(node::AbstractDict) = begin
        if haskey(node, "sym")
            s = node["sym"]
            haskey(vars_data, s)  && return vars_data[s]
            haskey(constants, s)  && return fill(constants[s], length(first(values(vars_data))))
            error("Unknown symbol: $s")
        end
        if haskey(node, "theta")
            idx = parse(Int, split(node["theta"], "_")[end]) + 1  # theta_0 -> index 1
            return fill(theta[idx], length(first(values(vars_data))))
        end
        if haskey(node, "num")
            return fill(Float64(node["num"]), length(first(values(vars_data))))
        end

        op   = node["op"]
        args = node["args"]

        if op == "add";  return sum(_eval(a) for a in args); end
        if op == "sub";  return _eval(args[1]) .- _eval(args[2]); end
        if op == "mul"
            result = _eval(args[1])
            for a in args[2:end]; result = result .* _eval(a); end
            return result
        end
        if op == "div";  return _eval(args[1]) ./ (_eval(args[2]) .+ 1e-15); end
        if op == "pow"
            base = _eval(args[1])
            exp_v = _eval(args[2])
            return base .^ exp_v
        end
        if op == "neg";  return -_eval(args[1]); end
        if op == "sqrt"; return sqrt.(max.(_eval(args[1]), 0.0)); end
        if op == "exp";  return exp.(clamp.(_eval(args[1]), -700.0, 700.0)); end
        if op == "log";  return log.(max.(_eval(args[1]), 1e-15)); end
        if op == "sin";  return sin.(_eval(args[1])); end
        if op == "cos";  return cos.(_eval(args[1])); end
        if op == "tan";  return tan.(_eval(args[1])); end
        if op == "tanh"; return tanh.(_eval(args[1])); end
        if op == "sinh"; return sinh.(clamp.(_eval(args[1]),-500.0,500.0)); end
        if op == "cosh"; return cosh.(clamp.(_eval(args[1]),-500.0,500.0)); end

        # ADCD primitive ops (lowercase primitive names as ops)
        if op == "d_lor";         u = _eval(args[1]); uc = clamp.(u, 0.0, 1.0 - 1e-9); s = sqrt.(1.0 .- uc); return uc ./ (s .* (1.0 .+ s)); end
        if op == "d_exp";         u = _eval(args[1]); return 1.0 .- exp.(-abs.(u)); end
        if op == "d_rat";         u = _eval(args[1]); return u ./ (1.0 .+ u.^2); end
        if op == "d_pow";         u = _eval(args[1]); return sqrt.(abs.(u)) .* (1.0 .- exp.(-abs.(u))); end
        if op == "d_log";         u = _eval(args[1]); return log.(1.0 .+ abs.(u)); end
        if op == "d_sat";         u = _eval(args[1]); return tanh.(u); end
        if op == "d_sqrt_inv";    u = _eval(args[1]); s=sqrt.(abs.(u)); return s ./ (1.0 .+ s); end
        if op == "d_tanh_sq";     u = _eval(args[1]); return tanh.(u.^2); end
        if op == "d_osc";         u = _eval(args[1]); return 1.0 .- cos.(u); end
        if op == "d_nested_mond"; u = _eval(args[1]); s=sqrt.(abs.(u)); return exp.(-s) .* (1.0 .- exp.(-s)); end
        if op == "d_rar";         u = _eval(args[1]); s=sqrt.(abs.(u) .+ 1e-15); e=exp.(-s); return e ./ max.(1.0 .- e, 1e-12); end

        error("Unknown op: $op")
    end
    return _eval(expr_node)
end

"""
    fit_constants(proposal_expr, y_classical, y_obs, vars_data, constants;
                  n_params, n_restarts=15, rng_seed=42,
                  correction_type="multiplicative") -> FitResult

Fit theta parameters to y_obs.

correction_type controls the model structure:
  "multiplicative": y_pred = y_classical * (1 + Delta(u; theta))
  "additive":       y_pred = y_classical + Delta(u; theta)

Additive mode is required when y_classical ≡ 0 (Mercury perihelion,
Muon g-2, Binary Pulsar) — in multiplicative mode the gradient is
identically zero and the optimizer learns nothing.

Uses multi-start L-BFGS via Optim.jl.
"""
function fit_constants(
    proposal_expr  ::Dict{String,Any},
    y_classical    ::Vector{Float64},
    y_obs          ::Vector{Float64},
    vars_data      ::Dict{String,Vector{Float64}},
    constants      ::Dict{String,Float64},
    n_params       ::Int;
    n_restarts     ::Int = 15,
    rng_seed       ::Int = 42,
    sigma_y        ::Union{Vector{Float64},Nothing} = nothing,
    correction_type::String = "multiplicative",
)::FitResult

    # Helper: build y_pred from delta given correction type
    make_pred(delta::Vector{Float64}) = correction_type == "additive" ?
        y_classical .+ delta :
        y_classical .* (1.0 .+ delta)

    n_params == 0 && begin
        # Helper: compute residuals in delta space to match Python and avoid
        # high-magnitude y_classical points completely dominating the fit.
        function compute_residuals_and_nmse(delta::Vector{Float64})
            if correction_type == "additive"
                resid_obs = y_obs .- y_classical
            else
                resid_obs = (y_obs .- y_classical) ./ (y_classical .+ 1e-15)
            end
            delta_residuals = resid_obs .- delta
            nmse_val = mean(delta_residuals.^2) / (var(resid_obs) + 1e-15)
            return delta_residuals, nmse_val
        end

        try
            delta = evaluate_expr(proposal_expr, vars_data, constants, Float64[])
            delta_residuals, nmse = compute_residuals_and_nmse(delta)
            n = length(y_obs)
            sigma2 = mean(delta_residuals.^2)
            ll = sigma2 > 0 ? (-0.5 * n * log(2π * sigma2) - n / 2.0) : -Inf
            # Store delta_residuals as FitResult residuals
            return FitResult(Float64[], nmse, ll, true, 0, 0, nothing, delta_residuals)
        catch e
            return FitResult(Float64[], Inf, -Inf, false, 0, 0, string(e), Float64[])
        end
    end

    rng = MersenneTwister(rng_seed)

    # Loss function: NMSE in delta space
    function loss(theta::Vector{Float64})::Float64
        try
            delta = evaluate_expr(proposal_expr, vars_data, constants, theta)
            
            if correction_type == "additive"
                resid_obs = y_obs .- y_classical
            else
                resid_obs = (y_obs .- y_classical) ./ (y_classical .+ 1e-15)
            end
            delta_residuals = resid_obs .- delta
            
            if !all(isfinite.(delta_residuals))
                return 1e10
            end
            if sigma_y !== nothing
                # If sigma_y is provided, we should ideally map it to delta space,
                # but for now we fallback to standard weighting.
                y_pred = make_pred(delta)
                return mean(((y_obs .- y_pred) ./ sigma_y).^2)
            else
                return mean(delta_residuals.^2) / (var(resid_obs) + 1e-15)
            end
        catch
            return 1e10
        end
    end

    best_nmse  = Inf
    best_theta = zeros(n_params)
    converged  = false

    scales = [1.0, 1e-9, 1e9, 1e-5, 1e5]

    for i in 1:n_restarts
        scale = scales[(i - 1) % length(scales) + 1]
        # Random initial point: log-uniform around 1, then scaled
        theta0 = exp.(randn(rng, n_params) .* 2.0) .* scale
        theta0 .*= rand(rng, [-1.0, 1.0], n_params)

        try
            result = optimize(
                loss,
                theta0,
                LBFGS(),
                Optim.Options(
                    iterations       = 500,
                    g_tol            = 1e-6,
                    show_trace       = false,
                    allow_f_increases= false,
                )
            )
            if Optim.converged(result) && Optim.minimum(result) < best_nmse
                best_nmse  = Optim.minimum(result)
                best_theta = Optim.minimizer(result)
                converged  = true
            end
        catch
            continue
        end
    end

    # Compute log-likelihood at best theta
    ll = -Inf
    delta_residuals = Float64[]
    try
        delta = evaluate_expr(proposal_expr, vars_data, constants, best_theta)
        
        if correction_type == "additive"
            resid_obs = y_obs .- y_classical
        else
            resid_obs = (y_obs .- y_classical) ./ (y_classical .+ 1e-15)
        end
        delta_residuals = resid_obs .- delta
        
        n = length(y_obs)
        sigma2 = mean(delta_residuals.^2)
        ll = sigma2 > 0 ? -0.5 * n * log(2 * pi * sigma2) - 0.5 * n : -Inf
    catch
    end

    return FitResult(best_theta, best_nmse, ll, converged, n_restarts, n_params, nothing, delta_residuals)
end

# Convenience helpers (avoid importing Statistics separately)
mean(x) = sum(x) / length(x)
var(x)  = mean((x .- mean(x)).^2)

end  # module ConstantFitter
