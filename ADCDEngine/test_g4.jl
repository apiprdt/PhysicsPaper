
using ADCDEngine.PrimitiveRegistry
# Evaluate D_nested_mond(D_sqrt_inv(u)) for u in [1e-4, 1, 100]
for u in [1e-4, 1.0, 100.0]
    inner = evaluate_primitive(PRIMITIVE_REGISTRY[:D_sqrt_inv], u)
    outer = evaluate_primitive(PRIMITIVE_REGISTRY[:D_nested_mond], inner)
    println("u=", u, " -> ", outer)
end
