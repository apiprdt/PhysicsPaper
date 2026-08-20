using JSON3
using ADCDEngine.ADCDDimensions
using ADCDEngine.CorrectionProposer

config = ProposalConfig("dimensionless", ["t_0", "v", "c"])
props = propose_corrections(config)
lor_props = filter(p -> :D_lor in p.primitives, props)

println("Found ", length(lor_props), " D_lor proposals")
for p in lor_props[1:min(5, length(lor_props))]
    println("Pattern: ", p.description)
    println("Expr: ", JSON3.write(p.expr))
    dim_res = infer_dim(p.expr)
    println("Infer dim: ", dim_res)
    println("Verify dim: ", verify_dimension(p.expr, "dimensionless"))
end