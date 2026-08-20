using JSON3
using ADCDEngine
using ADCDEngine.CorrectionProposer
using ADCDEngine.ADCDDimensions

prop_config = ProposalConfig("mechanics", ["t_0", "v", "c"], 3, true, true)
proposals_all = propose_corrections(prop_config)

# Keep only those with just D_lor
lor_only = filter(p -> all(prim == :D_lor for prim in p.primitives) && length(p.primitives) > 0, proposals_all)
println("Found ", length(lor_only), " D_lor ONLY proposals")

for p in lor_only
    println(p.description)
    println(JSON3.write(p.expr))
    println("Dim ok? ", verify_dimension(p.expr, "dimensionless"))
end
