using Pkg
Pkg.activate(raw"e:\ADCD\ADCDEngine")
Pkg.add(["JSON3", "LinearAlgebra", "Optim", "Symbolics", "StructTypes"])
Pkg.precompile()
println("Pkg add & precompile OK")