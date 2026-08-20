# adcd_cli.jl ? CLI runner for ADCD Julia Engine
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using ADCDEngine

if length(ARGS) >= 3 && ARGS[1] == "--verify-dim"
    expr_json = isfile(ARGS[2]) ? read(ARGS[2], String) : ARGS[2]
    target_dim = ARGS[3]
    ok = ADCDEngine.ADCDDimensions.verify_dimension(expr_json, target_dim)
    println(ok ? "true" : "false")
    exit(0)
elseif length(ARGS) >= 1 && ARGS[1] == "--list-primitives"
    ADCDEngine.list_primitives()
    exit(0)
elseif length(ARGS) >= 2
    config_file = ARGS[1]
    data_file   = ARGS[2]
    config_json = read(config_file, String)
    data_json   = read(data_file, String)
    result_json = ADCDEngine.run_adcd(config_json, data_json)
    if length(ARGS) >= 3
        write(ARGS[3], result_json)
    else
        println(result_json)
    end
else
    println(stderr, "Usage: julia adcd_cli.jl <config.json> <data.json> [output.json]")
    exit(1)
end
