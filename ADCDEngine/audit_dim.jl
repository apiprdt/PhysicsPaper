using ADCDEngine.ADCDDimensions
# Audit: verify v/c and v^2/c^2 are both truly dimensionless
ratio1 = Dict("op"=>"mul","args"=>[Dict("sym"=>"v"),Dict("op"=>"pow","args"=>[Dict("sym"=>"c"),Dict("num"=>-1)])])
ratio2 = Dict("op"=>"mul","args"=>[Dict("op"=>"pow","args"=>[Dict("sym"=>"v"),Dict("num"=>2)]),Dict("op"=>"pow","args"=>[Dict("sym"=>"c"),Dict("num"=>-2)])])
println("v/c dimensionless? ", is_dimensionless(infer_dim(ratio1)))
println("v^2/c^2 dimensionless? ", is_dimensionless(infer_dim(ratio2)))

# Audit: does t_0/v fail correctly?
bad_ratio = Dict("op"=>"div","args"=>[Dict("sym"=>"t_0"),Dict("sym"=>"v")])
println("t_0/v dimensionless? ", is_dimensionless(infer_dim(bad_ratio)))

# Audit: build_ratio_nodes with t_0,v,c -> ratios?
ratios = enumerate_dimensionless_ratios(["t_0","v","c"],2)
println("Ratios for [t_0,v,c]: ", length(ratios), " found")

# Confirm theta is treated as dimensionless (critical for Gate A)
theta = Dict("theta"=>"theta_0")
println("theta dimensionless? ", is_dimensionless(infer_dim(theta)))
