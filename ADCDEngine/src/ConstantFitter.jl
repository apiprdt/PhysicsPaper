# ADCD Engine: ConstantFitter (Hardened & Unified)
module ConstantFitter

using Optim
using Random
using LinearAlgebra

using ..ADCDDimensions
using ..CorrectionProposer

export FitResult, fit_constants, evaluate_expr

struct FitResult
    theta     ::Vector{Float64}   # fitted parameters
    nmse      ::Float64           # normalized mean squared error (in fit space)
    likelihood::Float64           # Gaussian log-likelihood (consistent with loss space)
    converged ::Bool
    n_restarts::Int               # total restarts executed
    n_params  ::Int
    error_msg ::Union{String,Nothing}
    residuals ::Vector{Float64}   # residuals evaluated in delta space
end

"""
    evaluate_expr(expr_node, vars_data, constants, theta) -> Vector{Float64}

Evaluates an ADCD symbolic expression tree into a Float64 vector.
"""
function evaluate_expr(
    expr_node::AbstractDict,
    vars_data ::Dict{String,Vector{Float64}},
    constants ::Dict{String,Float64},
    theta     ::Vector{Float64}
)::Vector{Float64}
    n_pts = length(first(values(vars_data)))

    _eval(node::AbstractDict) = begin
        if haskey(node, "sym")
            s = node["sym"]
            haskey(vars_data, s) && return vars_data[s]
            haskey(constants, s) && return fill(constants[s], n_pts)
            error("Unknown symbol: $s")
        end
        if haskey(node, "theta")
            idx = parse(Int, split(node["theta"], "_")[end]) + 1
            return fill(theta[idx], n_pts)
        end
        if haskey(node, "num")
            return fill(Float64(node["num"]), n_pts)
        end

        op   = node["op"]
        args = node["args"]

        if op == "add";  return sum(_eval(a) for a in args); end
        if op == "sub";  return _eval(args[1]) .- _eval(args[2]); end
        if op == "mul"
            res = _eval(args[1])
            for a in args[2:end]; res = res .* _eval(a); end
            return res
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
        if op == "sinh"; return sinh.(clamp.(_eval(args[1]), -500.0, 500.0)); end
        if op == "cosh"; return cosh.(clamp.(_eval(args[1]), -500.0, 500.0)); end

        # Primitives
        if op == "d_lor";         u = _eval(args[1]); uc = clamp.(u, 0.0, 1.0 - 1e-9); s = sqrt.(1.0 .- uc); return uc ./ (s .* (1.0 .+ s)); end
        if op == "d_exp";         u = _eval(args[1]); return 1.0 .- exp.(-abs.(u)); end
        if op == "d_rat";         u = _eval(args[1]); return u ./ (1.0 .+ u.^2); end
        if op == "d_pow";         u = _eval(args[1]); return sqrt.(abs.(u)) .* (1.0 .- exp.(-abs.(u))); end
        if op == "d_log";         u = _eval(args[1]); return log.(1.0 .+ abs.(u)); end
        if op == "d_sat";         u = _eval(args[1]); return tanh.(u); end
        if op == "d_sqrt_inv";    u = _eval(args[1]); s = sqrt.(abs.(u)); return s ./ (1.0 .+ s); end
        if op == "d_tanh_sq";     u = _eval(args[1]); return tanh.(u.^2); end
        if op == "d_osc";         u = _eval(args[1]); return 1.0 .- cos.(u); end
        if op == "d_nested_mond"; u = _eval(args[1]); s = sqrt.(abs.(u)); return exp.(-s) .* (1.0 .- exp.(-s)); end
        if op == "d_rar";         u = _eval(args[1]); s = sqrt.(abs.(u) .+ 1e-15); e = exp.(-s); return e ./ max.(1.0 .- e, 1e-12); end

        error("Unknown op: $op")
    end
    return _eval(expr_node)
end

"""
    fit_constants(proposal_expr, y_classical, y_obs, vars_data, constants;
                  n_params, n_restarts=15, rng_seed=42, sigma_y=nothing,
                  correction_type="multiplicative") -> FitResult
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

    n = length(y_obs)

    # 1. Mode dan Denominator Normalisasi
    abs_y = abs.(y_classical)
    pos_y = abs_y[abs_y .> 0.0]
    dynamic_range = isempty(pos_y) ? 0.0 : maximum(abs_y) / minimum(pos_y)
    use_full_loss = dynamic_range > 1e4

    # Baseline residuals (Delta space)
    resid_obs = correction_type == "additive" ?
        (y_obs .- y_classical) :
        (y_obs .- y_classical) ./ (y_classical .+ 1e-15)

    var_resid = max(var(resid_obs), 1e-15)
    var_full  = max(var(y_obs), 1e-15)

    # Helper evaluasi y_pred
    function get_y_pred(delta::Vector{Float64})
        return correction_type == "additive" ? (y_classical .+ delta) : (y_classical .* (1.0 .+ delta))
    end

    # 2. Penanganan Kasus Nol Parameter (n_params == 0)
    if n_params == 0
        try
            delta = evaluate_expr(proposal_expr, vars_data, constants, Float64[])
            delta_residuals = resid_obs .- delta

            if sigma_y !== nothing
                y_pred = get_y_pred(delta)
                w_res = (y_obs .- y_pred) ./ sigma_y
                chi2 = sum(w_res.^2)
                nmse_val = chi2 / n
                ll = -0.5 * chi2 - 0.5 * sum(log.(2π .* (sigma_y.^2)))
            elseif use_full_loss
                y_pred = get_y_pred(delta)
                diff = y_obs .- y_pred
                sigma2 = mean(diff.^2)
                nmse_val = sigma2 / var_full
                ll = sigma2 > 0 ? (-0.5 * n * log(2π * sigma2) - 0.5 * n) : -Inf
            else
                sigma2 = mean(delta_residuals.^2)
                nmse_val = sigma2 / var_resid
                ll = sigma2 > 0 ? (-0.5 * n * log(2π * sigma2) - 0.5 * n) : -Inf
            end

            return FitResult(Float64[], nmse_val, ll, true, 0, 0, nothing, delta_residuals)
        catch e
            return FitResult(Float64[], Inf, -Inf, false, 0, 0, string(e), Float64[])
        end
    end

    # 3. Objective Loss Function
    function loss(theta::Vector{Float64})::Float64
        try
            delta = evaluate_expr(proposal_expr, vars_data, constants, theta)
            if !all(isfinite, delta)
                return 1e10
            end

            if sigma_y !== nothing
                y_pred = get_y_pred(delta)
                return mean(((y_obs .- y_pred) ./ sigma_y).^2)
            elseif use_full_loss
                y_pred = get_y_pred(delta)
                return mean((y_pred .- y_obs).^2) / var_full
            else
                delta_residuals = resid_obs .- delta
                return mean(delta_residuals.^2) / var_resid
            end
        catch
            return 1e10
        end
    end

    rng = MersenneTwister(rng_seed)
    best_loss  = Inf
    best_theta = zeros(n_params)
    converged  = false

    # 4. Skala Inisialisasi Multivariabel Simetris
    scales = [1.0, 1e-3, 1e3, 1e-6, 1e6]

    for i in 1:n_restarts
        scale = scales[(i - 1) % length(scales) + 1]

        # Inisialisasi tanda: Pastikan restart awal mencakup variasi tanda
        init_signs = if i == 1
            ones(n_params)
        elseif i == 2
            -ones(n_params)
        elseif i == 3 && n_params >= 2
            # Pola selang-seling [+1, -1, +1, ...]
            Float64[(-1.0)^j for j in 1:n_params]
        else
            rand(rng, [-1.0, 1.0], n_params)
        end

        # Inisialisasi magnitudo parameter
        init_theta = if i <= 2
            fill(1.0 * scale, n_params)
        elseif i <= 4
            fill(0.5 * scale, n_params)
        else
            use_wide = rand(rng) > 0.5
            exponents = use_wide ? (rand(rng, n_params) .* 20.0 .- 10.0) : (rand(rng, n_params) .* 6.0 .- 3.0)
            (10.0 .^ exponents) .* scale
        end

        # Optimasi pada ruang logaritmik bernilai bertanda: theta = sign * exp(u)
        loss_log(u_log::Vector{Float64}) = loss(init_signs .* exp.(u_log))
        u0 = log.(max.(abs.(init_theta), 1e-12))

        try
            result_log = optimize(
                loss_log,
                u0,
                LBFGS(),
                Optim.Options(
                    iterations        = 500,
                    g_tol             = 1e-6,
                    show_trace        = false,
                    allow_f_increases = false,
                )
            )

            curr_loss = Optim.minimum(result_log)
            if isfinite(curr_loss) && curr_loss < best_loss
                best_loss  = curr_loss
                best_theta = init_signs .* exp.(Optim.minimizer(result_log))
                converged  = true
            end
        catch
            continue
        end
    end

    # 5. Sinkronisasi Likelihood & Residuals Pasca-Optimasi
    ll = -Inf
    delta_residuals = Float64[]

    try
        delta = evaluate_expr(proposal_expr, vars_data, constants, best_theta)
        delta_residuals = resid_obs .- delta

        if sigma_y !== nothing
            y_pred = get_y_pred(delta)
            w_res = (y_obs .- y_pred) ./ sigma_y
            chi2 = sum(w_res.^2)
            ll = -0.5 * chi2 - 0.5 * sum(log.(2π .* (sigma_y.^2)))
        elseif use_full_loss
            y_pred = get_y_pred(delta)
            diff = y_obs .- y_pred
            sigma2 = mean(diff.^2)
            ll = sigma2 > 0 ? (-0.5 * n * log(2π * sigma2) - 0.5 * n) : -Inf
        else
            sigma2 = mean(delta_residuals.^2)
            ll = sigma2 > 0 ? (-0.5 * n * log(2π * sigma2) - 0.5 * n) : -Inf
        end
    catch e
        return FitResult(best_theta, Inf, -Inf, false, n_restarts, n_params, string(e), delta_residuals)
    end

    return FitResult(best_theta, best_loss, ll, converged, n_restarts, n_params, nothing, delta_residuals)
end

# Helpers
mean(x) = sum(x) / length(x)
var(x)  = mean((x .- mean(x)).^2)

end  # module ConstantFitter
