using Pkg
Pkg.activate(raw"e:\ADCD\ADCDEngine")
include(raw"e:\ADCD\ADCDEngine\src\ADCDDimensions.jl")
include(raw"e:\ADCD\ADCDEngine\src\PrimitiveRegistry.jl")
include(raw"e:\ADCD\ADCDEngine\src\CorrectionProposer.jl")
include(raw"e:\ADCD\ADCDEngine\src\ConstantFitter.jl")
include(raw"e:\ADCD\ADCDEngine\src\IdentifiabilityGate.jl")
include(raw"e:\ADCD\ADCDEngine\src\FilterCascade.jl")

using .ADCDDimensions, .PrimitiveRegistry, .CorrectionProposer, .ConstantFitter, .IdentifiabilityGate, .FilterCascade

n = 80
r_vals = collect(range(0.1, 10.0, n))
v_vals = collect(range(1e6, 0.9e8, n))
c_val  = 3e8
k_e    = 8.99e9; q1 = 1.6e-19; q2 = 1.6e-19

y_cl  = k_e .* q1 .* q2 ./ r_vals.^2
true_theta = 0.15
beta  = (v_vals ./ c_val).^2
y_obs = y_cl .* (1.0 .+ true_theta .* beta)

vars_data = Dict("r" => r_vals, "v" => v_vals, "c" => fill(c_val, n), "q" => fill(q1, n), "Q" => fill(q2, n))
config = RunConfig("lorentz_special_relativity", "dimensionless", ["v", "c"], Dict("c" => c_val, "k_e" => k_e), 6.0, 1.5, 0.05, 10, nothing, 200)

prop_config = ProposalConfig("lorentz_special_relativity", ["v","c"])
proposals = propose_corrections(prop_config)
println("Proposals: ", length(proposals))
for p in proposals
    println("  - $(p.description)")
end

for p in proposals
    println("--- Testing proposal: $(p.description) ---")
    dim_ok = FilterCascade.gate_a_dimensional(p, config.target_dim)
    println("  Gate A (dim): ", dim_ok)
    asymp_ok = FilterCascade.gate_b_asymptotic(p, vars_data, config.known_constants)
    println("  Gate B (asymp): ", asymp_ok)
    coarse = FilterCascade.gate_c_coarse(p, y_cl, y_obs, vars_data, config.known_constants, config.nmse_coarse)
    println("  Gate C (coarse): ", coarse !== nothing ? "NMSE=$(coarse.nmse)" : "FAIL")
    fine = FilterCascade.gate_d_fine(p, y_cl, y_obs, vars_data, config.known_constants, config.n_restarts)
    if fine !== nothing
        verdict, dbic = FilterCascade.gate_e_identifiability(fine, y_cl, y_obs, config)
        println("  Gate D (fine): NMSE=$(fine.nmse), theta=$(fine.theta)")
        println("  Gate E (verdict): $(verdict), delta_bic=$(dbic)")
    else
        println("  Gate D (fine): FAIL/NOT_CONVERGED")
    end
end
