using ADCDEngine, ADCDEngine.CorrectionProposer, ADCDEngine.ADCDDimensions
import JSON3

# Audit bilateral: does bilateral generate valid (dimensionless) proposals when vars1=[t_0], vars2=[v,c]?
# This is the suspicious case: vars2 = vars[2:end] = all but first var
vars = ["t_0","v","c"]
vars1 = vars[1:1]
vars2 = vars[2:end]
println("vars1=", vars1, " vars2=", vars2)

r1 = enumerate_dimensionless_ratios(vars1,2)
r2 = enumerate_dimensionless_ratios(vars2,2)
println("Ratios for vars1 [t_0]: ", length(r1), " (expect 0 — t_0 alone has no dim ratio)")
println("Ratios for vars2 [v,c]: ", length(r2), " (expect 2 — v/c and v^2/c^2)")

# Audit: bilateral fallback - when vars1 has no ratio (isempty), what happens?
# build_ratio_nodes falls back to theta*t_0, which is NOT dimensionless
# Gate A should reject this automatically
println()
println("=== Bilateral dimensional integrity check ===")
cfg = ProposalConfig("lorentz_special_relativity",["t_0","v","c"],3,false,true)
proposals = propose_corrections(cfg)
bilateral_props = filter(p->p.pattern==:bilateral, proposals)
println("Bilateral proposals: ", length(bilateral_props))
for p in bilateral_props[1:min(3,end)]
    ok = verify_dimension(p.expr,"dimensionless")
    println("  dim ok=", ok, " desc=", p.description, " expr=", JSON3.write(p.expr)[1:min(100,end)])
end
