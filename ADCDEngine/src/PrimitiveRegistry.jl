# ADCD Engine: PrimitiveRegistry
module PrimitiveRegistry

using ..ADCDDimensions

export ADCDPrimitive, PRIMITIVE_REGISTRY, DOMAIN_TAXONOMY
export primitives_for_domain, evaluate_primitive, list_primitives
struct ADCDPrimitive
    name           ::Symbol
    func           ::Function   # u::Float64 -> Float64
    description    ::String
    latex          ::String     # LaTeX representation
    classical_limit::String     # what D(u) -> as u->0
    domain_note    ::String     # physical interpretation
    divergent_safe ::Bool       # true = D(0)→∞ by design (e.g. RAR); Gate B skipped
end

# Default outer constructor for non-divergent primitives (backward compat)
ADCDPrimitive(name, func, description, latex, classical_limit, domain_note) =
    ADCDPrimitive(name, func, description, latex, classical_limit, domain_note, false)

function evaluate_primitive(p::ADCDPrimitive, u::Real)::Float64
    p.func(Float64(u))
end

function Base.show(io::IO, p::ADCDPrimitive)
    print(io, "ADCDPrimitive(:$(p.name)) — $(p.description)")
end

# ---------------------------------------------------------------------------
# Primitive definitions
# All satisfy D(0) = 0 by construction. Verified by __init__ @asserts below.
# ---------------------------------------------------------------------------

# D_lor: Lorentz correction (Special Relativity)
# D_lor(u) = 1/√(1-u) - 1   where u = v²/c²
# Rationalized form: u / (√(1-u) · (1 + √(1-u)))
# This avoids catastrophic cancellation as u→0 (numerator is the small quantity).
# Domain: u ∈ [0, 1). Hard-clipped to [0, 1-ε] to prevent NaN from sqrt(negative).
# D_lor(0) = 0, D_lor(u→1⁻) → +∞  (Lorentz divergence)
_d_lor(u) = begin
    uc = clamp(u, 0.0, 1.0 - 1e-9)
    s = sqrt(1.0 - uc)
    uc / (s * (1.0 + s))
end

# D_exp: Exponential screening (Yukawa, Debye, Boltzmann)
# D_exp(u) = 1 - exp(-u)   where u = r/lambda or E/kT
# D_exp(0) = 0, D_exp(inf) -> 1
# Physically: deviation from unscreened potential
_d_exp(u) = 1.0 - exp(-abs(u))

# D_rat: Rational pole correction (Coulomb, 1/r^n potentials)
# D_rat(u) = u / (1 + u^2)
# D_rat(0) = 0, bounded
# Physically: regularised rational form without singularity at u=0
_d_rat(u) = u / (1.0 + u^2)

# D_pow: Power-law anomalous correction (critical phenomena, anomalous diffusion)
# D_pow(u; alpha) = u^alpha - 1  -- but alpha is a theta param, so we use
# D_pow(u) = u * (u - 1)  as the template form where alpha is fit
# Actually for the registry we store the family: D_pow(u) = u^theta - u
# At u=1 -> 0, but we need D(0) = 0, so use:
# D_pow(u) = u - u^2 = u(1-u)
# Better: D_pow(u) = u/(1+u) * ln(1+u)  but this is complex
# Simplest asymptotically-safe power form: D_pow(u) = u^(1/2) * (1 - exp(-u))
# D_pow(0) = 0, bounded
_d_pow(u) = sqrt(abs(u)) * (1.0 - exp(-abs(u)))

# D_log: Logarithmic correction (thermodynamics, QCD running coupling)
# D_log(u) = log(1 + u)   where u = T/T_c or similar
# D_log(0) = 0
_d_log(u) = log(1.0 + abs(u))

# D_sat: Saturation / Langevin (magnetic, spin systems)
# D_sat(u) = tanh(u)   (Brillouin/Langevin approximation for S=1/2)
# D_sat(0) = 0, D_sat(inf) -> 1
_d_sat(u) = tanh(u)

# D_sqrt_inv: MOND-type correction (deep-MOND radial acceleration)
# D_sqrt_inv(u) = sqrt(u) / (1 + sqrt(u))
# D_sqrt_inv(0) = 0, D_sqrt_inv(inf) -> 1
# Physically: interpolates between MOND (u<<1) and Newtonian (u>>1) regimes
# This is the key primitive missing from Python ADCD that enables SPARC/RAR
_d_sqrt_inv(u) = sqrt(abs(u)) / (1.0 + sqrt(abs(u)))

# D_tanh: Hyperbolic tangent correction (phase transitions, neural analogy)
# D_tanh(u) = tanh(u^2)  (even function, zero at u=0)
# D_tanh(0) = 0
_d_tanh_sq(u) = tanh(u^2)

# D_osc: Oscillatory correction (interference, diffraction)
# D_osc(u) = 1 - cos(u)
# D_osc(0) = 0, bounded in [0, 2]
_d_osc(u) = 1.0 - cos(u)

# D_nested_mond: Asymmetric nested exponential (smooth bell-shape)
# D_nested(u) = exp(-sqrt(u)) * (1 - exp(-sqrt(u)))
# D_nested(0) = 0 (verified: exp(0)*(1-exp(0)) = 1*0 = 0), D_nested(inf)->0
# NOTE: This is NOT the McGaugh RAR form. It is a smooth bell-shaped function
# useful for anomalies that peak at intermediate u and vanish at both limits.
# For the exact McGaugh RAR interpolation, use D_rar below.
_d_nested_mond(u) = begin
    s = sqrt(abs(u))
    exp(-s) * (1.0 - exp(-s))
end

# D_rar: Exact McGaugh-Lelli-Schombert (2016) RAR interpolating function
# Derivation: g_obs = g_bar / (1 - e^{-√u}),  u = g_bar/a0
#   Δ = g_obs/g_bar - 1 = e^{-√u} / (1 - e^{-√u})   ← quotient, NOT product
# Physics: D(u→0) → +∞ (deep-MOND boost, gravitational enhancement is large)
#          D(u→∞) →  0  (Newtonian limit, correction vanishes)
# DIVERGENT_SAFE: D(0)→∞ by design. Gate B (D(0)=0 check) is intentionally
# skipped for this primitive. D(∞)=0 is the relevant safety requirement here.
_d_rar(u) = begin
    s = sqrt(abs(u) + 1e-15)   # avoid sqrt(0) → 0/0
    e = exp(-s)
    e / max(1.0 - e, 1e-12)    # clip denominator; at u→0, this → e/e = 1 (finite)
end

const PRIMITIVE_REGISTRY = Dict{Symbol, ADCDPrimitive}(
    :D_lor => ADCDPrimitive(
        :D_lor, _d_lor,
        "Lorentz correction (Special Relativity)",
        raw"\frac{u}{\sqrt{1-u}(1+\sqrt{1-u})}",
        "D(0)=0, D(u→1⁻)→+∞, domain u∈[0,1)",
        "u = v²/c²; rationalized form of 1/√(1-u)-1 (avoids catastrophic cancellation)"
    ),
    :D_exp => ADCDPrimitive(
        :D_exp, _d_exp,
        "Exponential screening (Yukawa, Debye, Boltzmann)",
        raw"1 - e^{-u}",
        "D(0)=0, D(inf)->1",
        "u = r/lambda or E/kT; models screening/damping"
    ),
    :D_rat => ADCDPrimitive(
        :D_rat, _d_rat,
        "Rational correction (Coulomb, power law)",
        raw"\frac{u}{1+u^2}",
        "D(0)=0, bounded",
        "u = r/r0 or similar; regularised rational without singularity"
    ),
    :D_pow => ADCDPrimitive(
        :D_pow, _d_pow,
        "Power-law anomalous correction (critical phenomena)",
        raw"\sqrt{|u|}(1-e^{-|u|})",
        "D(0)=0",
        "u = (T-Tc)/Tc; sub-power-law correction near phase transition"
    ),
    :D_log => ADCDPrimitive(
        :D_log, _d_log,
        "Logarithmic correction (thermodynamics, running coupling)",
        raw"\log(1+|u|)",
        "D(0)=0",
        "u = T/T_c or alpha_s(Q); logarithmic deviation"
    ),
    :D_sat => ADCDPrimitive(
        :D_sat, _d_sat,
        "Saturation correction (magnetic, Langevin/Brillouin)",
        raw"\tanh(u)",
        "D(0)=0, D(inf)->1",
        "u = muB/kT; models magnetic saturation"
    ),
    :D_sqrt_inv => ADCDPrimitive(
        :D_sqrt_inv, _d_sqrt_inv,
        "MOND interpolation correction (radial acceleration relation)",
        raw"\frac{\sqrt{|u|}}{1+\sqrt{|u|}}",
        "D(0)=0, D(inf)->1",
        "u = g_bar/a0; enables deep-MOND to Newtonian interpolation (SPARC/RAR)"
    ),
    :D_tanh_sq => ADCDPrimitive(
        :D_tanh_sq, _d_tanh_sq,
        "Even hyperbolic tangent (symmetric phase corrections)",
        raw"\tanh(u^2)",
        "D(0)=0",
        "u = order parameter; symmetric around u=0"
    ),
    :D_osc => ADCDPrimitive(
        :D_osc, _d_osc,
        "Oscillatory correction (interference, diffraction)",
        raw"1 - \cos(u)",
        "D(0)=0, bounded [0,2]",
        "u = k*r or similar; oscillatory deviation from linear"
    ),
    :D_nested_mond => ADCDPrimitive(
        :D_nested_mond, _d_nested_mond,
        "Asymmetric nested exponential (smooth bell-shape, vanishes at both limits)",
        raw"e^{-\sqrt{|u|}}(1-e^{-\sqrt{|u|}})",
        "D(0)=0 (verified), D(inf)->0",
        "u = g_bar/a0; bell-shaped anomaly peaking at intermediate u. NOT McGaugh RAR (use D_rar for that)"
    ),
    # Bug #3 fix: D_rar is the CORRECT McGaugh-Lelli-Schombert (2016) RAR form.
    # D_nested_mond was incorrectly claimed to be this — it is a product, not a quotient.
    # D_rar diverges at u→0 (deep-MOND boost) by physical design; divergent_safe=true.
    :D_rar => ADCDPrimitive(
        :D_rar, _d_rar,
        "Exact McGaugh-Lelli-Schombert (2016) RAR interpolating function",
        raw"\frac{e^{-\sqrt{u}}}{1 - e^{-\sqrt{u}}}",
        "D(u→0)→+∞ (deep-MOND divergence, by design), D(∞)→0 (Newtonian)",
        "u = g_bar/a0; quotient form of RAR. Gate B skipped — divergence at u=0 is physically correct",
        true,  # divergent_safe = true: skip D(0)=0 check in __init__
    ),
)

# ---------------------------------------------------------------------------
# Domain taxonomy (mirrors Python DOMAIN_TAXONOMY in quickfit.py, expanded)
# ---------------------------------------------------------------------------

const DOMAIN_TAXONOMY = Dict{String, Vector{Symbol}}(
    # Yukawa / Debye screening
    "yukawa_debye_screening"    => [:D_exp, :D_rat],
    # Special Relativity
    "lorentz_special_relativity"=> [:D_lor],
    # Boltzmann thermodynamics
    "boltzmann_thermodynamics"  => [:D_exp, :D_log],
    # MOND / Radial Acceleration Relation — D_rar is the physically correct form
    # D_sqrt_inv and D_nested_mond kept for shape-search flexibility
    "mond_radial_acceleration"  => [:D_rar, :D_sqrt_inv, :D_nested_mond, :D_rat],
    # GR orbital corrections
    "gr_orbital_corrections"    => [:D_lor, :D_rat],
    # Ising / mean-field magnetism
    "ising_mean_field"          => [:D_sat, :D_rat],
    # Critical phenomena / scaling
    "critical_scaling"          => [:D_pow],
    # Turbulent transport
    "turbulent_transport"       => [:D_pow, :D_log],
    # Quantum corrections
    "quantum_corrections"       => [:D_exp, :D_log, :D_rat],
    # Generic / unknown domain — all safe primitives (D_rar excluded: divergent)
    "generic"                   => [k for (k,v) in PRIMITIVE_REGISTRY if !v.divergent_safe],
)

function primitives_for_domain(domain::String)::Vector{ADCDPrimitive}
    syms = get(DOMAIN_TAXONOMY, domain, [k for (k,v) in PRIMITIVE_REGISTRY if !v.divergent_safe])
    return [PRIMITIVE_REGISTRY[s] for s in syms if haskey(PRIMITIVE_REGISTRY, s)]
end

function list_primitives()
    for (k,v) in PRIMITIVE_REGISTRY
        println("  :$(k) — $(v.description)")
        println("         LaTeX: $(v.latex)")
        println("         Classical limit: $(v.classical_limit)")
        v.divergent_safe && println("         [DIVERGENT-SAFE: D(0)→∞, Gate B skipped]")
    end
end

# ---------------------------------------------------------------------------
# Module __init__: verify asymptotic safety at load time
# For standard primitives: D(0)=0 (Patent Claim #3)
# For divergent_safe primitives (e.g. D_rar): D(∞)→0 verified separately.
# The D(0)=0 check is intentionally SKIPPED for divergent_safe primitives —
# deep-MOND divergence at u=0 is physically correct for the RAR function.
# ---------------------------------------------------------------------------

function __init__()
    n_standard  = count(p -> !p.divergent_safe, values(PRIMITIVE_REGISTRY))
    n_divergent = count(p ->  p.divergent_safe, values(PRIMITIVE_REGISTRY))
    println("[PrimitiveRegistry] Verifying asymptotic safety: D(0)=0 for $n_standard standard primitives...")
    println("[PrimitiveRegistry] Divergent-safe primitives (D(0)→∞ allowed): $n_divergent")
    failed = Symbol[]
    for (name, prim) in PRIMITIVE_REGISTRY
        prim.divergent_safe && continue  # D(0)=0 not required; skip
        val_at_zero = prim.func(0.0)
        if abs(val_at_zero) > 1e-10
            push!(failed, name)
            @error "Primitive $(name) FAILS D(0)=0 check: D(0) = $(val_at_zero)"
        end
    end
    if isempty(failed)
        println("[PrimitiveRegistry] All $n_standard standard primitives pass D(0)=0. Engine is asymptotically safe.")
    else
        error("[PrimitiveRegistry] FATAL: Primitives $(failed) violate D(0)=0. ADCD engine cannot start.")
    end
end

end  # module PrimitiveRegistry
