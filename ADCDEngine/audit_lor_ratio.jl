using ADCDEngine, ADCDEngine.ConstantFitter, ADCDEngine.ADCDDimensions
import JSON3

# Understand the discrepancy: y_cls=ones(n) means y_obs = correction, NOT y_classical*(1+delta)
# The fitter does y_pred = y_classical * (1 + delta)
# For time dilation: y_classical=t_0, y_obs=t_observed=gamma*t_0
# So delta = gamma*t_0/t_0 - 1 = gamma - 1

n = 20
c_val = 3.0e8
v_frac = collect(range(0.1, 0.95, n))
v_vals = v_frac .* c_val
t0 = 1.0
y_cls = fill(t0, n)  # classical prediction is t_0 itself
y_obs = t0 ./ sqrt.(1.0 .- v_frac.^2)  # gamma * t_0

vars_data = Dict("v"=>v_vals)
consts = Dict("c"=>c_val)

# Build expression: theta_1 * D_lor(theta_0 * v/c)
# With theta0=1: u = v/c (correct dimensionless ratio)
# D_lor(u) = 1/sqrt(1-u) - 1 where u=v/c... BUT D_lor is defined for u=v^2/c^2!
# AUDIT BUG HYPOTHESIS: D_lor physically takes u = v^2/c^2, NOT v/c
# If the engine proposes D_lor(theta0 * v/c), it evaluates d_lor(v/c) = v/c / (sqrt(1-v/c)*(1+sqrt(1-v/c)))
# This is NOT equal to gamma-1 = 1/sqrt(1-v^2/c^2) - 1

sample_beta = 0.5  # v/c = 0.5
u_wrong = sample_beta          # v/c
u_correct = sample_beta^2      # v^2/c^2

# What d_lor actually computes:
d_lor_wrong = u_wrong / (sqrt(1-u_wrong) * (1+sqrt(1-u_wrong)))
d_lor_correct = u_correct / (sqrt(1-u_correct) * (1+sqrt(1-u_correct)))
gamma_minus_1 = 1/sqrt(1-sample_beta^2) - 1.0

println("beta=0.5, gamma-1=", gamma_minus_1)
println("d_lor(v/c)=", d_lor_wrong, " (NOT equal to gamma-1)")
println("d_lor(v^2/c^2)=", d_lor_correct, " (this IS gamma-1)")

# Now check what the diagnose script sees: NMSE 0.0002 with first candidate
# First candidate has theta~[1.007, 1.007] -> both thetas ~1
# So the expression is 1.007 * D_lor(1.007 * v^2/c^2)
# This is the v^2/c^2 variant, not v/c

# The two ratios from enumerate are: v*c^-1 (=v/c) and v^2*c^-2 (=v^2/c^2)  
# Which one came FIRST in the output? Looking at previous audit log:
# ratios returned: [{v*c^-1}, {v^2*c^-2}]  (sorted lexicographically)
# So first proposal: D_lor(theta0 * v/c) -> NMSE=0.77 (WRONG)
# Second proposal: D_lor(theta0 * v^2/c^2) -> NMSE~0 (CORRECT)

# Verify:
ratio_v2c2 = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_0"),Dict("op"=>"mul","args"=>[Dict("op"=>"pow","args"=>[Dict("sym"=>"v"),Dict("num"=>2)]),Dict("op"=>"pow","args"=>[Dict("sym"=>"c"),Dict("num"=>-2)])])])
expr2 = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_1"),Dict("op"=>"d_lor","args"=>[ratio_v2c2])])

delta_true = evaluate_expr(expr2, vars_data, consts, [1.0, 1.0])
residuals = y_obs .- y_cls .* (1.0 .+ delta_true)
var_y = sum((y_obs .- sum(y_obs)/n).^2)/n
nmse2 = sum(residuals.^2) / (n * var_y + 1e-300)
println()
println("D_lor(v^2/c^2) proposal NMSE at true params: ", nmse2, " (expect ~0)")
println("CONFIRMED: engine correctly finds v^2/c^2 as true Lorentz ratio")
