using ADCDEngine, ADCDEngine.ConstantFitter, ADCDEngine.ADCDDimensions
import JSON3

n = 20
c_val = 3.0e8
v_frac = range(0.1, 0.95, n)
v_vals = collect(v_frac) .* c_val
y_cls = ones(n)
y_obs = 1.0 ./ sqrt.(1 .- collect(v_frac).^2) .- 1

vars_data = Dict("v"=>v_vals)
consts = Dict("c"=>c_val)

ratio = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_0"),Dict("op"=>"mul","args"=>[Dict("sym"=>"v"),Dict("op"=>"pow","args"=>[Dict("sym"=>"c"),Dict("num"=>-1)])])])
d_lor_node = Dict("op"=>"d_lor","args"=>[ratio])
expr = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_1"),d_lor_node])

# true solution: theta0=1 (so u=v/c), theta1=1
theta_true = [1.0, 1.0]
delta = evaluate_expr(expr, vars_data, consts, theta_true)
res2 = sum((y_obs .- delta).^2)
var_y = sum((y_obs .- sum(y_obs)/n).^2) / n
nmse = res2 / (n * var_y + 1e-300)
println("AUDIT theta_index: nmse at true params = ", nmse, " (expect <1e-8)")

sample_v = 0.866 * c_val
delta_s = evaluate_expr(expr, Dict("v"=>[sample_v]), consts, [1.0, 1.0])
expected = 1.0/sqrt(1.0 - 0.866^2) - 1.0
println("At v=0.866c: delta=", delta_s[1], " expected=", expected, " err=", abs(delta_s[1]-expected))
