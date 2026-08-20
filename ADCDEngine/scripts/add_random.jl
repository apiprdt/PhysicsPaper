using Pkg
Pkg.activate(raw"e:\ADCD\ADCDEngine")
Pkg.add("Random")
Pkg.precompile()
println("Random added & precompiled successfully")