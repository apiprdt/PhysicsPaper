# ADCD Engine: PrimitiveRegistry (Hardened)
module PrimitiveRegistry

using ..ADCDDimensions

export ADCDPrimitive, PRIMITIVE_REGISTRY, DOMAIN_TAXONOMY
export primitives_for_domain, evaluate_primitive, list_primitives

# Aman dari overflow pada u > 10^154 dan aman dari zero-gradient di u=0
@inline safe_abs(u::Float64) = hypot(u, 1e-20)

struct ADCDPrimitive
    name           ::Symbol
    func           ::Function
    description    ::String
    latex          ::String
    classical_limit::String
    domain_note    ::String
    divergent_safe ::Bool
end

ADCDPrimitive(name, func, description, latex, classical_limit, domain_note) =
    ADCDPrimitive(name, func, description, latex, classical_limit, domain_note, false)

evaluate_primitive(p::ADCDPrimitive, u::Real) = p.func(Float64(u))

# Primitives
_d_lor(u)         = (uc = min(u, 1.0 - 1e-9); s = sqrt(max(1.0 - uc, 0.0)); uc / (s * (1.0 + s)))
_d_exp(u)         = 1.0 - exp(-safe_abs(u))
_d_rat(u)         = u / (1.0 + u^2)
_d_pow(u)         = sqrt(safe_abs(u)) * (1.0 - exp(-safe_abs(u)))
_d_log(u)         = log(1.0 + safe_abs(u))
_d_sat(u)         = tanh(u)
_d_sqrt_inv(u)    = (s = sqrt(safe_abs(u)); s / (1.0 + s))
_d_tanh_sq(u)     = tanh(u^2)
_d_osc(u)         = 1.0 - cos(u)
_d_nested_mond(u) = (s = sqrt(safe_abs(u)); exp(-s) * (1.0 - exp(-s)))
_d_rar(u)         = (s = sqrt(safe_abs(u)); e = exp(-s); e / max(1.0 - e, 1e-12))

const PRIMITIVE_REGISTRY = Dict{Symbol, ADCDPrimitive}(
    :D_lor         => ADCDPrimitive(:D_lor, _d_lor, "Lorentz correction", raw"\frac{u}{\sqrt{1-u}(1+\sqrt{1-u})}", "D(0)=0", "u = v^2/c^2"),
    :D_exp         => ADCDPrimitive(:D_exp, _d_exp, "Exponential screening", raw"1 - e^{-u}", "D(0)=0", "Screening/damping"),
    :D_rat         => ADCDPrimitive(:D_rat, _d_rat, "Rational pole", raw"\frac{u}{1+u^2}", "D(0)=0", "Rational regularized"),
    :D_pow         => ADCDPrimitive(:D_pow, _d_pow, "Power-law anomalous", raw"\sqrt{|u|}(1-e^{-|u|})", "D(0)=0", "Phase transitions"),
    :D_log         => ADCDPrimitive(:D_log, _d_log, "Logarithmic correction", raw"\log(1+|u|)", "D(0)=0", "Running coupling"),
    :D_sat         => ADCDPrimitive(:D_sat, _d_sat, "Saturation", raw"\tanh(u)", "D(0)=0", "Langevin saturation"),
    :D_sqrt_inv    => ADCDPrimitive(:D_sqrt_inv, _d_sqrt_inv, "Square-root inverse", raw"\frac{\sqrt{|u|}}{1+\sqrt{|u|}}", "D(0)=0", "Deep-MOND interpolation"),
    :D_tanh_sq     => ADCDPrimitive(:D_tanh_sq, _d_tanh_sq, "Even tanh", raw"\tanh(u^2)", "D(0)=0", "Symmetric transitions"),
    :D_osc         => ADCDPrimitive(:D_osc, _d_osc, "Oscillatory", raw"1 - \cos(u)", "D(0)=0", "Interference/waves"),
    :D_nested_mond => ADCDPrimitive(:D_nested_mond, _d_nested_mond, "Asymmetric nested", raw"e^{-\sqrt{|u|}}(1-e^{-\sqrt{|u|}})", "D(0)=0", "Bell-shaped anomaly"),
    :D_rar         => ADCDPrimitive(:D_rar, _d_rar, "McGaugh RAR", raw"\frac{e^{-\sqrt{u}}}{1 - e^{-\sqrt{u}}}", "D(u->0)->Inf, D(Inf)->0", "Quotient RAR form", true),
)

const DOMAIN_TAXONOMY = Dict{String, Vector{Symbol}}(
    "yukawa_debye_screening"     => [:D_exp, :D_rat],
    "lorentz_special_relativity" => [:D_lor],
    "boltzmann_thermodynamics"   => [:D_exp, :D_log],
    "mond_radial_acceleration"   => [:D_rar, :D_sqrt_inv, :D_nested_mond, :D_rat],
    "gr_orbital_corrections"     => [:D_lor, :D_rat],
    "ising_mean_field"           => [:D_sat, :D_rat],
    "critical_scaling"           => [:D_pow],
    "turbulent_transport"        => [:D_pow, :D_log],
    "quantum_corrections"        => [:D_exp, :D_log, :D_rat],
    "generic"                    => [k for (k,v) in PRIMITIVE_REGISTRY if !v.divergent_safe],
)

function primitives_for_domain(domain::String)::Vector{ADCDPrimitive}
    syms = get(DOMAIN_TAXONOMY, domain, [k for (k,v) in PRIMITIVE_REGISTRY if !v.divergent_safe])
    return [PRIMITIVE_REGISTRY[s] for s in syms if haskey(PRIMITIVE_REGISTRY, s)]
end

function __init__()
    for (name, prim) in PRIMITIVE_REGISTRY
        prim.divergent_safe && continue
        val_zero = prim.func(0.0)
        if abs(val_zero) > 1e-8
            error("[PrimitiveRegistry] Asymptotic safety violation on $(name): D(0) = $(val_zero)")
        end
    end
end

end  # module PrimitiveRegistry
