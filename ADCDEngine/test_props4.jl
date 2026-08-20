using JSON3
using ADCDEngine.ADCDDimensions
r = enumerate_dimensionless_ratios(["t_0", "v", "c"], 2)
println(JSON3.write(r))
