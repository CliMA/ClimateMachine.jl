export PhasePartition
# Thermodynamic states
export ThermodynamicState,
    PhaseDry,
    PhaseDry_ρT,
    PhaseDry_pT,
    PhaseDry_pθ,
    PhaseEquil,
    PhaseEquil_ρTq,
    PhaseEquil_pTq,
    PhaseEquil_ρθq,
    PhaseEquil_pθq,
    PhaseNonEquil,
    PhaseNonEquil_ρTq,
    PhaseNonEquil_ρθq,
    PhaseNonEquil_pθq

"""
    ThermodynamicState{FT}

A thermodynamic state, which can be initialized for
various thermodynamic formulations (via its sub-types).
All `ThermodynamicState`'s have access to functions to
compute all other thermodynamic properties.
"""
abstract type ThermodynamicState{FT} end

"""
    PhasePartition

Represents the mass fractions of the moist air mixture.

# Constructors

    PhasePartition(q_tot::Real[, q_liq::Real[, q_ice::Real]])
    PhasePartition(ts::ThermodynamicState)

See also [`PhasePartition_equil`](@ref)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhasePartition{FT <: Real}
    "total specific humidity"
    tot::FT
    "liquid water specific humidity (default: `0`)"
    liq::FT
    "ice specific humidity (default: `0`)"
    ice::FT
    function PhasePartition(tot::FT, liq::FT, ice::FT) where {FT}
        q_tot_safe = max(tot, 0)
        q_liq_safe = max(liq, 0)
        q_ice_safe = max(ice, 0)
        return new{FT}(q_tot_safe, q_liq_safe, q_ice_safe)
    end
end

PhasePartition(q_tot::FT, q_liq::FT) where {FT <: Real} =
    PhasePartition(q_tot, q_liq, zero(FT))
PhasePartition(q_tot::FT) where {FT <: Real} =
    PhasePartition(q_tot, zero(FT), zero(FT))

#####
##### Dry states
#####

"""
    PhaseDry{FT} <: ThermodynamicState

A dry thermodynamic state (`q_tot = 0`).

# Constructors

    PhaseDry(param_set, e_int, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseDry{FT, PS} <: ThermodynamicState{FT}
    "parameter set, used to dispatch planet parameter function calls"
    param_set::PS
    "internal energy"
    e_int::FT
    "density of dry air"
    ρ::FT
end
PhaseDry(param_set::APS, e_int::FT, ρ::FT) where {FT} =
    PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)

"""
    PhaseDry_pT(param_set, p, T)

Constructs a [`PhaseDry`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `p` pressure
 - `T` temperature
"""
function PhaseDry_pT(param_set::APS, p::FT, T::FT) where {FT <: Real}
    e_int = internal_energy(param_set, T)
    ρ = air_density(param_set, T, p)
    return PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)
end

"""
    PhaseDry_pθ(param_set, p, θ_dry)

Constructs a [`PhaseDry`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `p` pressure
 - `θ_dry` dry potential temperature
"""
function PhaseDry_pθ(param_set::APS, p::FT, θ_dry::FT) where {FT <: Real}
    T = exner_given_pressure(param_set, p) * θ_dry
    e_int = internal_energy(param_set, T)
    ρ = air_density(param_set, T, p)
    return PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)
end

"""
    PhaseDry_ρT(param_set, ρ, T)

Constructs a [`PhaseDry`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `ρ` density
 - `T` temperature
"""
function PhaseDry_ρT(param_set::APS, ρ::FT, T::FT) where {FT <: Real}
    e_int = internal_energy(param_set, T)
    return PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)
end

#####
##### Equilibrium states
#####

"""
    PhaseEquil{FT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic equilibrium (therefore, saturation adjustment
may be needed).

# Constructors

    PhaseEquil(param_set, e_int, ρ, q_tot)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseEquil{FT, PS} <: ThermodynamicState{FT}
    "parameter set, used to dispatch planet parameter function calls"
    param_set::PS
    "internal energy"
    e_int::FT
    "density of air (potentially moist)"
    ρ::FT
    "total specific humidity"
    q_tot::FT
    "temperature: computed via [`saturation_adjustment`](@ref)"
    T::FT
end

"""
    PhaseEquil
Moist thermodynamic phase, given
 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `e_int` internal energy
 - `ρ` density
 - `q_tot` total specific humidity
and, optionally
 - `maxiter` maximum iterations for saturation adjustment
 - `temperature_tol` temperature tolerance for saturation adjustment
 - `sat_adjust` function pointer to particular saturation adjustment method, options include
    - `saturation_adjustment` uses Newtons method with analytic gradients
    - `saturation_adjustment_SecantMethod` uses Secant method
"""
function PhaseEquil(
    param_set::APS,
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int = 8,
    temperature_tol::FT = FT(1e-1),
    sat_adjust::Function = saturation_adjustment,
) where {FT <: Real}
    phase_type = PhaseEquil
    q_tot_safe = clamp(q_tot, FT(0), FT(1))
    T = sat_adjust(
        param_set,
        e_int,
        ρ,
        q_tot_safe,
        phase_type,
        maxiter,
        temperature_tol,
    )
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot_safe, T)
end

"""
    PhaseEquil_ρθq(param_set, ρ, θ_liq_ice, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `ρ` (moist-)air density
 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `temperature_tol` temperature tolerance for saturation adjustment
 - `maxiter` maximum iterations for saturation adjustment
"""
function PhaseEquil_ρθq(
    param_set::APS,
    ρ::FT,
    θ_liq_ice::FT,
    q_tot::FT,
    maxiter::Int = 36,
    temperature_tol::FT = FT(1e-1),
) where {FT <: Real}
    phase_type = PhaseEquil
    tol = ResidualTolerance(temperature_tol)
    T = saturation_adjustment_given_ρθq(
        param_set,
        ρ,
        θ_liq_ice,
        q_tot,
        phase_type,
        maxiter,
        tol,
    )
    q_pt = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
end

"""
    PhaseEquil_pθq(param_set, p, θ_liq_ice, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `p` pressure
 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `temperature_tol` temperature tolerance for saturation adjustment
 - `maxiter` maximum iterations for saturation adjustment
"""
function PhaseEquil_pθq(
    param_set::APS,
    p::FT,
    θ_liq_ice::FT,
    q_tot::FT,
    maxiter::Int = 30,
    temperature_tol::FT = FT(1e-1),
) where {FT <: Real}
    phase_type = PhaseEquil
    tol = ResidualTolerance(temperature_tol)
    T = saturation_adjustment_given_pθq(
        param_set,
        p,
        θ_liq_ice,
        q_tot,
        phase_type,
        maxiter,
        tol,
    )
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end

"""
    PhaseEquil_ρTq(param_set, ρ, T, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `ρ` density
 - `T` temperature
 - `q_tot` total specific humidity
"""
function PhaseEquil_ρTq(
    param_set::APS,
    ρ::FT,
    T::FT,
    q_tot::FT,
) where {FT <: Real}
    phase_type = PhaseEquil
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
end

"""
    PhaseEquil_pTq(param_set, p, T, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `p` pressure
 - `T` temperature
 - `q_tot` total specific humidity
"""
function PhaseEquil_pTq(
    param_set::APS,
    p::FT,
    T::FT,
    q_tot::FT,
) where {FT <: Real}
    phase_type = PhaseEquil
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
end

#####
##### Non-equilibrium states
#####

"""
   	 PhaseNonEquil{FT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic non-equilibrium (therefore, temperature can
be computed directly).

# Constructors

    PhaseNonEquil(param_set, e_int, q::PhasePartition, ρ)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct PhaseNonEquil{FT, PS} <: ThermodynamicState{FT}
    "parameter set, used to dispatch planet parameter function calls"
    param_set::PS
    "internal energy"
    e_int::FT
    "density of air (potentially moist)"
    ρ::FT
    "phase partition"
    q::PhasePartition{FT}
end
function PhaseNonEquil(
    param_set::APS,
    e_int::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT}
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q)
end

"""
    PhaseNonEquil_ρTq(param_set, ρ, T, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `ρ` (moist-)air density
 - `T` temperature
 - `q_pt` phase partition
"""
function PhaseNonEquil_ρTq(
    param_set::APS,
    ρ::FT,
    T::FT,
    q_pt::PhasePartition{FT},
) where {FT <: Real}
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end

"""
    PhaseNonEquil_ρθq(param_set, ρ, θ_liq_ice, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `ρ` (moist-)air density
 - `θ_liq_ice` liquid-ice potential temperature
 - `q_pt` phase partition
and, optionally
 - `potential_temperature_tol` potential temperature for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve
"""
function PhaseNonEquil_ρθq(
    param_set::APS,
    ρ::FT,
    θ_liq_ice::FT,
    q_pt::PhasePartition{FT},
    maxiter::Int = 10,
    potential_temperature_tol::FT = FT(1e-2),
) where {FT <: Real}
    phase_type = PhaseNonEquil
    tol = ResidualTolerance(potential_temperature_tol)
    T = air_temperature_given_θρq_nonlinear(
        param_set,
        θ_liq_ice,
        ρ,
        maxiter,
        tol,
        q_pt,
    )
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end

"""
    PhaseNonEquil_pθq(param_set, p, θ_liq_ice, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `p` pressure
 - `θ_liq_ice` liquid-ice potential temperature
 - `q_pt` phase partition
"""
function PhaseNonEquil_pθq(
    param_set::APS,
    p::FT,
    θ_liq_ice::FT,
    q_pt::PhasePartition{FT},
) where {FT <: Real}
    T = air_temperature_given_θpq(param_set, θ_liq_ice, p, q_pt)
    ρ = air_density(param_set, T, p, q_pt)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end
