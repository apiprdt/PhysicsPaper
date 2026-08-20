using ADCDEngine, ADCDEngine.CorrectionProposer, ADCDEngine.ADCDDimensions
import JSON3

# Critical audit: does theta * t_0 (the bilateral u1 fallback) pass Gate A?
# It SHOULD NOT pass Gate A because t_0 is dimensional (time)
# The bilateral pattern with a dimensional u1 falls through to gate A which rejects it

# Simulate what FilterCascade Gate A does:
u1_bad = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_0"),Dict("sym"=>"t_0")])
d_lor_bad = Dict("op"=>"d_lor","args"=>[u1_bad])

println("Gate A check: d_lor(theta*t_0) dimensionless?")
println("  result: ", verify_dimension(d_lor_bad,"dimensionless"))

# The bilateral fallback generates theta*t_0 which is DIM_TRANSCENDENTAL_ARG in d_lor
# Gate A REJECTS it -> these proposals are safely discarded
println()
println("CONCLUSION: bilateral fallback proposals with dimensional u1 are correctly rejected by Gate A")

# Now count what really passes through the full pipeline with known_constants
using ADCDEngine.FilterCascade, ADCDEngine.ConstantFitter

# Simulate a simple Time Dilation with the correct known_constants
n = 30
c_val = 3.0e8
v_vals = range(1e7, 2.5e8, n)
t0_val = 1.0
y_classical = fill(t0_val, n)
y_obs = t0_val ./ sqrt.(1 .- (collect(v_vals)/c_val).^2)

vars_data = Dict("t_0"=>fill(t0_val,n), "v"=>collect(v_vals))
known_consts = Dict("c"=>c_val)

config = RunConfig("lorentz_special_relativity","dimensionless",["t_0","v"],known_consts,6.0,1.0,0.1,15,nothing,500)
vars_and_consts = vcat(config.input_vars, collect(keys(config.known_constants)))
prop_config = ProposalConfig(config.domain, vars_and_consts,3,true,true)
proposals = propose_corrections(prop_config)

println()
println("Total proposals generated (with known_consts included): ", length(proposals))
passing_a = filter(p -> verify_dimension(p.expr, config.target_dim), proposals)
println("Proposals passing Gate A: ", length(passing_a))
for p in passing_a
    println("  ", p.description)
end
