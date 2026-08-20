using Pkg
Pkg.activate(raw"e:\ADCD\ADCDEngine")
Pkg.precompile()
using ADCDEngine
println("ADCDEngine fully precompiled and ready!")