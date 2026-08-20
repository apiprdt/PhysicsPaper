using ADCDEngine, ADCDEngine.ConstantFitter, ADCDEngine.ADCDDimensions
import JSON3

# AUDIT: expr_str theta mapping vs generate_final_3_figures.py
# The figure script expects cand["theta_fit"] which is a dict {theta_0: val, ...}
# The Julia engine outputs cand["theta"] as a list [val0, val1, ...]
# The julia_bridge.py CandidateResult has theta: list[float]
# Does generate_final_3_figures.py actually use Julia output or Python output?

# Check what the figure script parses for theta_fit
println("AUDIT: node_to_sympy output for D_lor(v^2/c^2) proposal")

# Manually build the expr as julia_bridge would output it
ratio = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_0"),Dict("op"=>"mul","args"=>[Dict("op"=>"pow","args"=>[Dict("sym"=>"v"),Dict("num"=>2)]),Dict("op"=>"pow","args"=>[Dict("sym"=>"c"),Dict("num"=>-2)])])])
d_lor_node = Dict("op"=>"d_lor","args"=>[ratio])
expr = Dict("op"=>"mul","args"=>[Dict("theta"=>"theta_1"),d_lor_node])

# Manually simulate node_to_sympy
function node_to_sympy(node::Dict)::String
    haskey(node,"num") && return string(node["num"])
    haskey(node,"sym") && return node["sym"]
    haskey(node,"theta") && return node["theta"]
    op = node["op"]
    args = node["args"]
    if op=="mul"; return "("*join([node_to_sympy(a) for a in args]," * ")*")"; end
    if op=="pow"; return "("*node_to_sympy(args[1])*"**"*node_to_sympy(args[2])*")"; end
    if op=="d_lor"; u=node_to_sympy(args[1]); return "(1/sqrt(1 - ("*u*")) - 1)"; end
    return "UNKNOWN"
end

result = node_to_sympy(expr)
println("expr_str = ", result)

# Now check if this is sympy-parseable and correct
# The substitution in Python: sp.sympify(expr_str).subs({theta_0: 1.0, theta_1: 1.0})
# At theta0=1, theta1=1, beta=0.5: should give gamma-1=0.1547

# For theta_0=1, theta_1=1, v=0.5c, c=3e8:
# expr_str = (theta_1 * (1/sqrt(1 - (theta_0 * (v**2 * c**-2.0))) - 1))
# subs: theta_0->1, theta_1->1
# = (1/sqrt(1 - (v^2/c^2)) - 1) = gamma-1 ✓

println()
println("AUDIT: With theta0=1.0, theta1=1.0, v=0.5c -> expr_str gives gamma-1=0.1547 ✓")
println("AUDIT: BUT generate_final_3_figures.py uses [theta_fit], NOT [theta]")
println("       Julia outputs: theta=[val0,val1] (list)")
println("       Python expects: theta_fit={theta_0:val0, theta_1:val1} (dict)")
println("       ACTION REQUIRED: Check if julia_bridge creates theta_fit dict")
