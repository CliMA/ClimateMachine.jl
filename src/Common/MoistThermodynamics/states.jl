export PhasePartition
# Thermodynamic states
export ThermodynamicState,
    PhaseDry,
    PhaseDry_given_pT,
    PhaseEquil,
    PhaseNonEquil,
    TemperatureSHumEquil,
    LiquidIcePotTempSHumEquil,
    LiquidIcePotTempSHumEquil_given_pressure,
    LiquidIcePotTempSHumNonEquil,
    LiquidIcePotTempSHumNonEquil_given_pressure

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
end

PhasePartition(q_tot::FT, q_liq::FT) where {FT <: Real} =
    PhasePartition(q_tot, q_liq, zero(FT))
PhasePartition(q_tot::FT) where {FT <: Real} =
    PhasePartition(q_tot, zero(FT), zero(FT))


"""
    ThermodynamicState{FT}

A thermodynamic state, which can be initialized for
various thermodynamic formulations (via its sub-types).
All `ThermodynamicState`'s have access to functions to
compute all other thermodynamic properties.
"""
abstract type ThermodynamicState{FT} end

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
 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
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
    maxiter::Int = 6,
    temperature_tol::FT = FT(1e-1),
    sat_adjust::Function = saturation_adjustment,
) where {FT <: Real}
    _cv_d = FT(cv_d(param_set))
    # Convert temperature tolerance to a convergence criterion on internal energy residuals
    tol = ResidualTolerance(temperature_tol * _cv_d)
    q_tot_safe = clamp(q_tot, FT(0), FT(1))
    T = sat_adjust(param_set, e_int, ρ, q_tot_safe, maxiter, tol)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot_safe, T)
end

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
    PhaseDry_given_pT(param_set, p, T)

Constructs a [`PhaseDry`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `p` pressure
 - `T` temperature
"""
function PhaseDry_given_pT(param_set::APS, p::FT, T::FT) where {FT <: Real}
    e_int = internal_energy(param_set, T)
    ρ = air_density(param_set, T, p)
    return PhaseDry{FT, typeof(param_set)}(param_set, e_int, ρ)
end


"""
    LiquidIcePotTempSHumEquil(param_set, θ_liq_ice, ρ, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `temperature_tol` temperature tolerance for saturation adjustment
 - `maxiter` maximum iterations for saturation adjustment
"""
function LiquidIcePotTempSHumEquil(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int = 36,
    temperature_tol::FT = FT(1e-1),
) where {FT <: Real}
    tol = ResidualTolerance(temperature_tol)
    T = saturation_adjustment_q_tot_θ_liq_ice(
        param_set,
        θ_liq_ice,
        ρ,
        q_tot,
        maxiter,
        tol,
    )
    q_pt = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
end

"""
    LiquidIcePotTempSHumEquil_given_pressure(param_set, θ_liq_ice, p, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
 - `q_tot` total specific humidity
 - `temperature_tol` temperature tolerance for saturation adjustment
 - `maxiter` maximum iterations for saturation adjustment
"""
function LiquidIcePotTempSHumEquil_given_pressure(
    param_set::APS,
    θ_liq_ice::FT,
    p::FT,
    q_tot::FT,
    maxiter::Int = 30,
    temperature_tol::FT = FT(1e-1),
) where {FT <: Real}
    tol = ResidualTolerance(temperature_tol)
    T = saturation_adjustment_q_tot_θ_liq_ice_given_pressure(
        param_set,
        θ_liq_ice,
        p,
        q_tot,
        maxiter,
        tol,
    )
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end

"""
    TemperatureSHumEquil(param_set, T, p, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `T` temperature
 - `p` pressure
 - `q_tot` total specific humidity
"""
function TemperatureSHumEquil(
    param_set::APS,
    T::FT,
    p::FT,
    q_tot::FT,
) where {FT <: Real}
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
end

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
    LiquidIcePotTempSHumNonEquil(param_set, θ_liq_ice, ρ, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
 - `q_pt` phase partition
and, optionally
 - `potential_temperature_tol` potential temperature for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve
"""
function LiquidIcePotTempSHumNonEquil(
    param_set::APS,
    θ_liq_ice::FT,
    ρ::FT,
    q_pt::PhasePartition{FT},
    maxiter::Int = 10,
    potential_temperature_tol::FT = FT(1e-2),
) where {FT <: Real}
    tol = ResidualTolerance(potential_temperature_tol)
    T = air_temperature_from_liquid_ice_pottemp_non_linear(
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
    LiquidIcePotTempSHumNonEquil_given_pressure(param_set, θ_liq_ice, p, q_pt)

Constructs a [`PhaseNonEquil`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`MoistThermodynamics`](@ref) for more details
 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
 - `q_pt` phase partition
"""
function LiquidIcePotTempSHumNonEquil_given_pressure(
    param_set::APS,
    θ_liq_ice::FT,
    p::FT,
    q_pt::PhasePartition{FT},
) where {FT <: Real}
    T = air_temperature_from_liquid_ice_pottemp_given_pressure(
        param_set,
        θ_liq_ice,
        p,
        q_pt,
    )
    ρ = air_density(param_set, T, p, q_pt)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end
