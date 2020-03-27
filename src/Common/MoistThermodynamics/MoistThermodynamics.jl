"""
    MoistThermodynamics

Moist thermodynamic functions, e.g., for air pressure (atmosphere equation
of state), latent heats of phase transitions, saturation vapor pressures, and
saturation specific humidities.
"""
module MoistThermodynamics

using DocStringExtensions

using RootSolvers
using ..Parameters
using ..PlanetParameters

# Atmospheric equation of state
export air_pressure,
    air_temperature, air_density, specific_volume, soundspeed_air
export total_specific_humidity

# Energies
export total_energy, internal_energy, internal_energy_sat

# Specific heats of moist air
export cp_m, cv_m, gas_constant_air, gas_constants

# Latent heats
export latent_heat_vapor,
    latent_heat_sublim, latent_heat_fusion, latent_heat_liq_ice

# Saturation vapor pressures and specific humidities over liquid and ice
export Liquid, Ice
export saturation_vapor_pressure, q_vap_saturation_generic, q_vap_saturation
export saturation_excess

# Functions used in thermodynamic equilibrium among phases (liquid and ice
# determined diagnostically from total water specific humidity)

export liquid_fraction, PhasePartition_equil

# Auxiliary functions, e.g., for diagnostic purposes
export dry_pottemp,
    dry_pottemp_given_pressure, virtual_pottemp, exner, exner_given_pressure
export liquid_ice_pottemp,
    liquid_ice_pottemp_given_pressure, liquid_ice_pottemp_sat, relative_humidity
export air_temperature_from_liquid_ice_pottemp,
    air_temperature_from_liquid_ice_pottemp_given_pressure
export air_temperature_from_liquid_ice_pottemp_non_linear
export vapor_specific_humidity

# The default ParameterSet for Moist Thermodynamics:
const MTPS = EarthParameterSet

include("states.jl")
include("isentropic.jl")

@inline q_pt_0(::Type{FT}) where {FT} = PhasePartition{FT}(FT(0), FT(0), FT(0))

"""
    gas_constant_air([q::PhasePartition])

The specific gas constant of moist air given
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
gas_constant_air(q::PhasePartition{FT}, param_set::PS = MTPS()) where {FT, PS} =
    FT(R_d) *
    (1 + (FT(molmass_ratio) - 1) * q.tot - FT(molmass_ratio) * (q.liq + q.ice))

gas_constant_air(::Type{FT}, param_set::PS = MTPS()) where {FT, PS} =
    gas_constant_air(q_pt_0(FT), param_set)

"""
    gas_constant_air(ts::ThermodynamicState)

The specific gas constant of moist air given
a thermodynamic state `ts`.
"""
gas_constant_air(ts::ThermodynamicState) =
    gas_constant_air(PhasePartition(ts), ts.param_set)
gas_constant_air(ts::PhaseDry{FT}) where {FT <: Real} = FT(R_d)

"""
    vapor_specific_humidity(q::PhasePartition{FT})

The vapor specific humidity, given a `PhasePartition` `q`.
"""
vapor_specific_humidity(q::PhasePartition) = q.tot - q.liq - q.ice
vapor_specific_humidity(ts::ThermodynamicState) =
    vapor_specific_humidity(PhasePartition(ts))

"""
    air_pressure(T, ρ[, q::PhasePartition])

The air pressure from the equation of state
(ideal gas law) where

 - `T` air temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
air_pressure(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = gas_constant_air(q, param_set) * ρ * T
air_pressure(T::FT, ρ::FT, param_set::PS) where {FT <: Real, PS} =
    air_pressure(T, ρ, q_pt_0(FT), param_set)

"""
    air_pressure(ts::ThermodynamicState)

The air pressure from the equation of state
(ideal gas law), given a thermodynamic state `ts`.
"""
air_pressure(ts::ThermodynamicState) = air_pressure(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)


"""
    air_density(T, p[, q::PhasePartition])

The (moist-)air density from the equation of state
(ideal gas law) where

 - `T` air temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
air_density(
    T::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = p / (gas_constant_air(q, param_set) * T)
air_density(T::FT, p::FT, param_set::PS = MTPS()) where {FT, PS} =
    air_density(T, p, q_pt_0(FT), param_set)

"""
    air_density(ts::ThermodynamicState)

The (moist-)air density, given a thermodynamic state `ts`.
"""
air_density(ts::ThermodynamicState) = ts.ρ

"""
    specific_volume(ts::ThermodynamicState)

The (moist-)air specific volume, given a thermodynamic state `ts`.
"""
specific_volume(ts::ThermodynamicState) = 1 / air_density(ts)

"""
    total_specific_humidity(ts::ThermodynamicState)

Total specific humidity, given a thermodynamic state `ts`.
"""
total_specific_humidity(ts::ThermodynamicState) = ts.q_tot
total_specific_humidity(ts::PhaseDry{FT}) where {FT} = FT(0)
total_specific_humidity(ts::PhaseNonEquil) = ts.q.tot

"""
    cp_m([q::PhasePartition])

The isobaric specific heat capacity of moist
air where, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
cp_m(q::PhasePartition{FT}, param_set::PS = MTPS()) where {FT <: Real, PS} =
    FT(cp_d) +
    (FT(cp_v) - FT(cp_d)) * q.tot +
    (FT(cp_l) - FT(cp_v)) * q.liq +
    (FT(cp_i) - FT(cp_v)) * q.ice

cp_m(::Type{FT}, param_set::PS = MTPS()) where {FT <: Real, PS} =
    cp_m(q_pt_0(FT), param_set)

"""
    cp_m(ts::ThermodynamicState)

The isobaric specific heat capacity of moist
air, given a thermodynamic state `ts`.
"""
cp_m(ts::ThermodynamicState) = cp_m(PhasePartition(ts), ts.param_set)
cp_m(ts::PhaseDry{FT}) where {FT <: Real} = FT(cp_d)

"""
    cv_m([q::PhasePartition])

The isochoric specific heat capacity of moist
air where optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
cv_m(q::PhasePartition{FT}, param_set::PS = MTPS()) where {FT <: Real, PS} =
    FT(cv_d) +
    (FT(cv_v) - FT(cv_d)) * q.tot +
    (FT(cv_l) - FT(cv_v)) * q.liq +
    (FT(cv_i) - FT(cv_v)) * q.ice

cv_m(::Type{FT}, param_set::PS = MTPS()) where {FT <: Real, PS} =
    cv_m(q_pt_0(FT), param_set)

"""
    cv_m(ts::ThermodynamicState)

The isochoric specific heat capacity of moist
air given a thermodynamic state `ts`.
"""
cv_m(ts::ThermodynamicState) = cv_m(PhasePartition(ts), ts.param_set)
cv_m(ts::PhaseDry{FT}) where {FT <: Real} = FT(cv_d)


"""
    (R_m, cp_m, cv_m, γ_m) = gas_constants([q::PhasePartition])

Wrapper to compute all gas constants at once, where optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`
"""
function gas_constants(
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    R_gas = gas_constant_air(q, param_set)
    cp = cp_m(q, param_set)
    cv = cv_m(q, param_set)
    γ = cp / cv
    return (R_gas, cp, cv, γ)
end
gas_constants(param_set::PS) where {FT <: Real, PS} =
    gas_constants(q_pt_0(FT), param_set)

"""
    (R_m, cp_m, cv_m, γ_m) = gas_constants(ts::ThermodynamicState)

Wrapper to compute all gas constants at once, given a thermodynamic state `ts`.

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`

"""
gas_constants(ts::ThermodynamicState) =
    gas_constants(PhasePartition(ts), ts.param_set)

"""
    air_temperature(e_int, q::PhasePartition)

The air temperature, where

 - `e_int` internal energy per unit mass
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_temperature(
    e_int::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    T_0 +
    (
        e_int - (q.tot - q.liq) * FT(e_int_v0) +
        q.ice * (FT(e_int_v0) + FT(e_int_i0))
    ) / cv_m(q, param_set)
end
air_temperature(e_int::FT, param_set::PS) where {FT <: Real, PS} =
    air_temperature(e_int, q_pt_0(FT), param_set)

"""
    air_temperature(ts::ThermodynamicState)

The air temperature, given a thermodynamic state `ts`.
"""
air_temperature(ts::ThermodynamicState) =
    air_temperature(internal_energy(ts), PhasePartition(ts), ts.param_set)
air_temperature(ts::PhaseEquil) = ts.T


"""
    internal_energy(T[, q::PhasePartition])

The internal energy per unit mass, given a thermodynamic state `ts` or

 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
internal_energy(
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} =
    cv_m(q, param_set) * (T - FT(T_0)) + (q.tot - q.liq) * FT(e_int_v0) -
    q.ice * (FT(e_int_v0) + FT(e_int_i0))
internal_energy(T::FT, param_set::PS = MTPS()) where {FT, PS} =
    internal_energy(T, q_pt_0(FT), param_set)

"""
    internal_energy(ts::ThermodynamicState)

The internal energy per unit mass, given a thermodynamic state `ts`.
"""
internal_energy(ts::ThermodynamicState) = ts.e_int

"""
    internal_energy(ρ::FT, ρe::FT, ρu::AbstractVector{FT}, e_pot::FT)

The internal energy per unit mass, given
 - `ρ` (moist-)air density
 - `ρe` total energy per unit volume
 - `ρu` momentum vector
 - `e_pot` potential energy (e.g., gravitational) per unit mass
"""
@inline function internal_energy(
    ρ::FT,
    ρe::FT,
    ρu::AbstractVector{FT},
    e_pot::FT,
) where {FT <: Real}
    ρinv = 1 / ρ
    ρe_kin = ρinv * sum(abs2, ρu) / 2
    ρe_pot = ρ * e_pot
    ρe_int = ρe - ρe_kin - ρe_pot
    e_int = ρinv * ρe_int
    return e_int
end

"""
    internal_energy_sat(T, ρ, q_tot)

The internal energy per unit mass in thermodynamic equilibrium at saturation where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
"""
internal_energy_sat(
    T::FT,
    ρ::FT,
    q_tot::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS} =
    internal_energy(T, PhasePartition_equil(T, ρ, q_tot, param_set), param_set)

"""
    internal_energy_sat(ts::ThermodynamicState)

The internal energy per unit mass in
thermodynamic equilibrium at saturation,
given a thermodynamic state `ts`.
"""
internal_energy_sat(ts::ThermodynamicState) = internal_energy_sat(
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
    ts.param_set,
)


"""
    total_energy(e_kin, e_pot, T[, q::PhasePartition])

The total energy per unit mass, given

 - `e_kin` kinetic energy per unit mass
 - `e_pot` potential energy per unit mass
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.

"""
function total_energy(
    e_kin::FT,
    e_pot::FT,
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    return e_kin + e_pot + internal_energy(T, q, param_set)
end
total_energy(
    e_kin::FT,
    e_pot::FT,
    T::FT,
    param_set::PS,
) where {FT <: Real, PS} = total_energy(e_kin, e_pot, T, q_pt_0(FT), param_set)

"""
    total_energy(e_kin, e_pot, ts::ThermodynamicState)

The total energy per unit mass
given a thermodynamic state `ts`.
"""
total_energy(
    e_kin::FT,
    e_pot::FT,
    ts::ThermodynamicState{FT},
) where {FT <: Real} = internal_energy(ts) + FT(e_kin) + FT(e_pot)

"""
    soundspeed_air(T[, q::PhasePartition])

The speed of sound in unstratified air, where
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function soundspeed_air(
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    γ = cp_m(q, param_set) / cv_m(q, param_set)
    R_m = gas_constant_air(q, param_set)
    return sqrt(γ * R_m * T)
end
soundspeed_air(T::FT, param_set::PS = MTPS()) where {FT, PS} =
    soundspeed_air(T, q_pt_0(FT), param_set)

"""
    soundspeed_air(ts::ThermodynamicState)

The speed of sound in unstratified air given a thermodynamic state `ts`.
"""
soundspeed_air(ts::ThermodynamicState) =
    soundspeed_air(air_temperature(ts), PhasePartition(ts), ts.param_set)


"""
    latent_heat_vapor(T::FT) where {FT<:Real}

The specific latent heat of vaporization where
 - `T` temperature
"""
latent_heat_vapor(T::FT, param_set::PS = MTPS()) where {FT <: Real, PS} =
    latent_heat_generic(T, FT(LH_v0), FT(cp_v) - FT(cp_l), param_set)

"""
    latent_heat_vapor(ts::ThermodynamicState)

The specific latent heat of vaporization
given a thermodynamic state `ts`.
"""
latent_heat_vapor(ts::ThermodynamicState) =
    latent_heat_vapor(air_temperature(ts), ts.param_set)

"""
    latent_heat_sublim(T::FT) where {FT<:Real}

The specific latent heat of sublimation where
 - `T` temperature
"""
latent_heat_sublim(T::FT, param_set::PS = MTPS()) where {FT <: Real, PS} =
    latent_heat_generic(T, FT(LH_s0), FT(cp_v) - FT(cp_i), param_set)

"""
    latent_heat_sublim(ts::ThermodynamicState)

The specific latent heat of sublimation
given a thermodynamic state `ts`.
"""
latent_heat_sublim(ts::ThermodynamicState) =
    latent_heat_sublim(air_temperature(ts), ts.param_set)

"""
    latent_heat_fusion(T::FT) where {FT<:Real}

The specific latent heat of fusion where
 - `T` temperature
"""
latent_heat_fusion(T::FT, param_set::PS = MTPS()) where {FT <: Real, PS} =
    latent_heat_generic(T, FT(LH_f0), FT(cp_l) - FT(cp_i), param_set)

"""
    latent_heat_fusion(ts::ThermodynamicState)

The specific latent heat of fusion
given a thermodynamic state `ts`.
"""
latent_heat_fusion(ts::ThermodynamicState{FT}) where {FT <: Real, PS} =
    latent_heat_fusion(air_temperature(ts), ts.param_set)

"""
    latent_heat_generic(T::FT, LH_0::FT, Δcp::FT) where {FT<:Real}

The specific latent heat of a generic phase transition between
two phases, computed using Kirchhoff's relation with constant
isobaric specific heat capacities of the two phases, given

 - `T` temperature
 - `LH_0` latent heat of the phase transition at `T_0`
 - `Δcp` difference between the isobaric specific heat capacities
         (heat capacity in the higher-temperature phase minus that
         in the lower-temperature phase).
"""
latent_heat_generic(
    T::FT,
    LH_0::FT,
    Δcp::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = LH_0 + Δcp * (T - FT(T_0))


"""
    Phase

A condensed phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`q_vap_saturation_generic`](@ref).
"""
abstract type Phase end

"""
    Liquid <: Phase

A liquid phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`q_vap_saturation_generic`](@ref).
"""
struct Liquid <: Phase end

"""
    Ice <: Phase

An ice phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`q_vap_saturation_generic`](@ref).
"""
struct Ice <: Phase end

"""
    saturation_vapor_pressure(T, Liquid())

Return the saturation vapor pressure over a plane liquid surface at
temperature `T`.

    saturation_vapor_pressure(T, Ice())

Return the saturation vapor pressure over a plane ice surface at
temperature `T`.

    saturation_vapor_pressure(T, LH_0, Δcp)

Compute the saturation vapor pressure over a plane surface by integration
of the Clausius-Clapeyron relation.

The Clausius-Clapeyron relation

    dlog(p_v_sat)/dT = [LH_0 + Δcp * (T-T_0)]/(R_v*T^2)

is integrated from the triple point temperature `T_triple`, using
Kirchhoff's relation

    L = LH_0 + Δcp * (T - T_0)

for the specific latent heat `L` with constant isobaric specific
heats of the phases. The linear dependence of the specific latent heat
on temperature `T` allows analytic integration of the Clausius-Clapeyron
relation to obtain the saturation vapor pressure `p_v_sat` as a function of
the triple point pressure `press_triple`.

"""
saturation_vapor_pressure(
    T::FT,
    ::Liquid,
    param_set::PS = MTPS(),
) where {FT <: Real, PS} =
    saturation_vapor_pressure(T, FT(LH_v0), FT(cp_v) - FT(cp_l), param_set)

function saturation_vapor_pressure(
    ts::ThermodynamicState{FT},
    ::Liquid,
) where {FT <: Real}

    return saturation_vapor_pressure(
        air_temperature(ts),
        FT(LH_v0),
        FT(cp_v) - FT(cp_l),
        ts.param_set,
    )

end

saturation_vapor_pressure(
    T::FT,
    ::Ice,
    param_set::PS = MTPS(),
) where {FT <: Real, PS} =
    saturation_vapor_pressure(T, FT(LH_s0), FT(cp_v) - FT(cp_i), param_set)

saturation_vapor_pressure(
    ts::ThermodynamicState{FT},
    ::Ice,
) where {FT <: Real} = saturation_vapor_pressure(
    air_temperature(ts),
    FT(LH_s0),
    FT(cp_v) - FT(cp_i),
    ts.param_set,
)

function saturation_vapor_pressure(
    T::FT,
    LH_0::FT,
    Δcp::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS}

    return FT(press_triple) *
           (T / FT(T_triple))^(Δcp / FT(R_v)) *
           exp(
               (FT(LH_0) - Δcp * FT(T_0)) / FT(R_v) *
               (1 / FT(T_triple) - 1 / T),
           )

end

"""
    q_vap_saturation_generic(T, ρ[; phase=Liquid()])

Compute the saturation specific humidity over a plane surface of condensate, given

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `Liquid()` indicating condensate is liquid
 - `Ice()` indicating condensate is ice
"""
function q_vap_saturation_generic(
    T::FT,
    ρ::FT,
    param_set::PS = MTPS();
    phase::Phase = Liquid(),
) where {FT <: Real, PS}
    p_v_sat = saturation_vapor_pressure(T, phase, param_set)
    return q_vap_saturation_from_pressure(T, ρ, p_v_sat, param_set)
end

"""
    q_vap_saturation(T, ρ[, q::PhasePartition])

Compute the saturation specific humidity, given

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref)

If the `PhasePartition` `q` is given, the saturation specific humidity is that of a
mixture of liquid and ice, computed in a thermodynamically consistent way from the
weighted sum of the latent heats of the respective phase transitions (Pressel et al.,
JAMES, 2015). That is, the saturation vapor pressure and from it the saturation specific
humidity are computed from a weighted mean of the latent heats of vaporization and
sublimation, with the weights given by the fractions of condensates `q.liq/(q.liq + q.ice)`
and `q.ice/(q.liq + q.ice)` that are liquid and ice, respectively.

If the `PhasePartition` `q` is not given, or has zero liquid and ice specific humidities,
the saturation specific humidity is that over a mixture of liquid and ice, with the
fraction of liquid given by temperature dependent `liquid_fraction(T)` and the
fraction of ice by the complement `1 - liquid_fraction(T)`.
"""
function q_vap_saturation(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}

    # get phase partitioning
    _liquid_frac = liquid_fraction(T, q, param_set)
    _ice_frac = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0 = _liquid_frac * FT(LH_v0) + _ice_frac * FT(LH_s0)
    Δcp =
        _liquid_frac * (FT(cp_v) - FT(cp_l)) + _ice_frac * (FT(cp_v) - FT(cp_i))

    # saturation vapor pressure over possible mixture of liquid and ice
    p_v_sat = saturation_vapor_pressure(T, FT(LH_0), Δcp, param_set)

    return q_vap_saturation_from_pressure(T, ρ, p_v_sat, param_set)

end
q_vap_saturation(T::FT, ρ::FT, param_set::PS) where {FT <: Real, PS} =
    q_vap_saturation(T, ρ, q_pt_0(FT), param_set)

"""
    q_vap_saturation(ts::ThermodynamicState)

Compute the saturation specific humidity, given a thermodynamic state `ts`.
"""
q_vap_saturation(ts::ThermodynamicState) = q_vap_saturation(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    q_vap_saturation_from_pressure(T, ρ, p_v_sat)

Compute the saturation specific humidity, given

 - `T` temperature,
 - `ρ` (moist-)air density
 - `p_v_sat` saturation vapor pressure
"""
q_vap_saturation_from_pressure(
    T::FT,
    ρ::FT,
    p_v_sat::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = p_v_sat / (ρ * FT(R_v) * T)

"""
    saturation_excess(T, ρ, q::PhasePartition)

The saturation excess in equilibrium where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q` [`PhasePartition`](@ref)

The saturation excess is the difference between the total specific humidity `q.tot`
and the saturation specific humidity in equilibrium, and it is defined to be
nonzero only if this difference is positive.
"""
saturation_excess(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT},
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = max(0, q.tot - q_vap_saturation(T, ρ, q, param_set))

"""
    saturation_excess(ts::ThermodynamicState)

Compute the saturation excess in equilibrium,
given a thermodynamic state `ts`.
"""
saturation_excess(ts::ThermodynamicState) = saturation_excess(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    liquid_fraction(T[, q::PhasePartition])

The fraction of condensate that is liquid where

 - `T` temperature
 - `q` [`PhasePartition`](@ref)

If `q.liq` or `q.ice` are nonzero, the liquid fraction is computed from
them.

Otherwise, phase equilibrium is assumed so that the fraction of liquid
is a function that is 1 above `T_freeze` and goes to zero below `T_freeze`.
"""
function liquid_fraction(
    T::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    q_c = q.liq + q.ice     # condensate specific humidity
    if q_c > 0
        return q.liq / q_c
    else
        # For now: Heaviside function for partitioning into liquid and ice: all liquid
        # for T > T_freeze; all ice for T <= T_freeze
        return FT(T > FT(T_freeze))
    end
end
liquid_fraction(T::FT, param_set::PS = MTPS()) where {FT, PS} =
    liquid_fraction(T, q_pt_0(FT), param_set)

"""
    liquid_fraction(ts::ThermodynamicState)

The fraction of condensate that is liquid given a thermodynamic state `ts`.
"""
liquid_fraction(ts::ThermodynamicState) =
    liquid_fraction(air_temperature(ts), PhasePartition(ts), ts.param_set)

"""
    PhasePartition_equil(T, ρ, q_tot)

Partition the phases in equilibrium, returning a [`PhasePartition`](@ref) object using the
[`liquid_fraction`](@ref) function where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity

The residual `q.tot - q.liq - q.ice` is the vapor specific humidity.
"""
function PhasePartition_equil(
    T::FT,
    ρ::FT,
    q_tot::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    _liquid_frac = liquid_fraction(T, param_set)                      # fraction of condensate that is liquid
    q_c = saturation_excess(T, ρ, PhasePartition(q_tot), param_set) # condensate specific humidity
    q_liq = _liquid_frac * q_c                             # liquid specific humidity
    q_ice = (1 - _liquid_frac) * q_c                       # ice specific humidity

    return PhasePartition(q_tot, q_liq, q_ice)
end

PhasePartition_equil(ts::PhaseNonEquil) = PhasePartition_equil(
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
    ts.param_set,
)

PhasePartition(ts::PhaseDry{FT}) where {FT <: Real} = q_pt_0(FT)
PhasePartition(ts::PhaseEquil) = PhasePartition_equil(
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
    ts.param_set,
)
PhasePartition(ts::PhaseNonEquil) = ts.q

function ∂e_int_∂T(
    T::FT,
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    param_set::PS,
) where {FT <: Real, PS}
    cvm = cv_m(PhasePartition_equil(T, ρ, q_tot, param_set), param_set)
    q_vap_sat = q_vap_saturation(T, ρ, param_set)
    λ = liquid_fraction(T, param_set)
    L = λ * FT(LH_v0) + (1 - λ) * FT(LH_s0)
    ∂q_vap_sat_∂T = q_vap_sat * L / (FT(R_v) * T^2)
    T0 = FT(T_0)
    dcvm_dq_vap = FT(cv_v) - λ * FT(cv_l) - (1 - λ) * FT(cv_i)
    return cvm +
           (FT(e_int_v0) + (1 - λ) * FT(e_int_i0) + (T - T0) * dcvm_dq_vap) *
           ∂q_vap_sat_∂T
end

"""
    saturation_adjustment(e_int, ρ, q_tot)

Compute the temperature that is consistent with

 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `tol` tolerance for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``e_int - internal_energy_sat(T,ρ,q_tot) = 0``

using Newtons method with analytic gradients.

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment(
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int,
    tol::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    T_1 =
        max(FT(T_min), air_temperature(e_int, PhasePartition(q_tot), param_set)) # Assume all vapor
    q_v_sat = q_vap_saturation(T_1, ρ, param_set)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > FT(T_min)
        return T_1
    else
        sol = find_zero(
            T -> internal_energy_sat(T, ρ, q_tot, param_set) - e_int,
            T_ -> ∂e_int_∂T(T_, e_int, ρ, q_tot, param_set),
            T_1,
            NewtonsMethod(),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment did not converge")
        end
        return sol.root
    end
end

"""
    ΔT_min(::Type{FT})

Minimum interval for saturation adjustment using Secant method
"""
@inline ΔT_min(::Type{FT}) where {FT} = FT(3)

"""
    ΔT_max(::Type{FT})

Maximum interval for saturation adjustment using Secant method
"""
@inline ΔT_max(::Type{FT}) where {FT} = FT(10)

"""
    bound_upper_temperature(T_1::FT, T_2::FT) where {FT<:Real}

Bounds the upper temperature, `T_2`, for
saturation adjustment using Secant method
"""
@inline function bound_upper_temperature(T_1::FT, T_2::FT) where {FT <: Real}
    T_2 = max(T_1 + ΔT_min(FT), T_2)
    return min(T_1 + ΔT_max(FT), T_2)
end

"""
    saturation_adjustment_SecantMethod(e_int, ρ, q_tot)

Compute the temperature `T` that is consistent with

 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
 - `tol` tolerance for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``e_int - internal_energy_sat(T,ρ,q_tot) = 0``

See also [`saturation_adjustment_q_tot_θ_liq_ice`](@ref).
"""
function saturation_adjustment_SecantMethod(
    e_int::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int,
    tol::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    T_1 =
        max(FT(T_min), air_temperature(e_int, PhasePartition(q_tot), param_set)) # Assume all vapor
    q_v_sat = q_vap_saturation(T_1, ρ, param_set)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > FT(T_min)
        return T_1
    else
        # FIXME here: need to revisit bounds for saturation adjustment to guarantee bracketing of zero.
        T_2 = air_temperature(
            e_int,
            PhasePartition(q_tot, FT(0), q_tot),
            param_set,
        ) # Assume all ice
        T_2 = bound_upper_temperature(T_1, T_2)
        sol = find_zero(
            T -> internal_energy_sat(T, ρ, q_tot, param_set) - e_int,
            T_1,
            T_2,
            SecantMethod(),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment_SecantMethod did not converge")
        end
        return sol.root
    end
end

"""
    saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, ρ, q_tot)

Compute the temperature `T` that is consistent with

 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `ρ` (moist-)air density
 - `tol` tolerance for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``
  θ_{liq_ice} - liquid_ice_pottemp_sat(T, ρ, q_tot) = 0
``

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment_q_tot_θ_liq_ice(
    θ_liq_ice::FT,
    ρ::FT,
    q_tot::FT,
    maxiter::Int,
    tol::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    T_1 = max(
        FT(T_min),
        air_temperature_from_liquid_ice_pottemp(
            θ_liq_ice,
            ρ,
            PhasePartition(q_tot),
            param_set,
        ),
    ) # Assume all vapor
    q_v_sat = q_vap_saturation(T_1, ρ, param_set)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > FT(T_min)
        return T_1
    else
        T_2 = air_temperature_from_liquid_ice_pottemp(
            θ_liq_ice,
            ρ,
            PhasePartition(q_tot, FT(0), q_tot),
            param_set,
        ) # Assume all ice
        T_2 = bound_upper_temperature(T_1, T_2)
        sol = find_zero(
            T -> liquid_ice_pottemp_sat(T, ρ, q_tot, param_set) - θ_liq_ice,
            T_1,
            T_2,
            SecantMethod(),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment_q_tot_θ_liq_ice did not converge")
        end
        return sol.root
    end
end

"""
    saturation_adjustment_q_tot_θ_liq_ice_given_pressure(θ_liq_ice, p, q_tot)

Compute the temperature `T` that is consistent with

 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `p` pressure
 - `tol` tolerance for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve

by finding the root of

``
  θ_{liq_ice} - liquid_ice_pottemp_sat(T, air_density(T, p, PhasePartition(q_tot)), q_tot) = 0
``

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment_q_tot_θ_liq_ice_given_pressure(
    θ_liq_ice::FT,
    p::FT,
    q_tot::FT,
    maxiter::Int,
    tol::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    T_1 = air_temperature_from_liquid_ice_pottemp_given_pressure(
        θ_liq_ice,
        p,
        PhasePartition(q_tot),
        param_set,
    ) # Assume all vapor
    ρ = air_density(T_1, p, PhasePartition(q_tot), param_set)
    q_v_sat = q_vap_saturation(T_1, ρ, param_set)
    unsaturated = q_tot <= q_v_sat
    if unsaturated && T_1 > FT(T_min)
        return T_1
    else
        T_2 = air_temperature_from_liquid_ice_pottemp(
            θ_liq_ice,
            p,
            PhasePartition(q_tot, FT(0), q_tot),
            param_set,
        ) # Assume all ice
        T_2 = bound_upper_temperature(T_1, T_2)
        sol = find_zero(
            T ->
                liquid_ice_pottemp_sat(
                    T,
                    air_density(T, p, PhasePartition(q_tot), param_set),
                    q_tot,
                    param_set,
                ) - θ_liq_ice,
            T_1,
            T_2,
            SecantMethod(),
            CompactSolution(),
            tol,
            maxiter,
        )
        if !sol.converged
            error("saturation_adjustment_q_tot_θ_liq_ice_given_pressure did not converge")
        end
        return sol.root
    end
end

"""
    latent_heat_liq_ice(q::PhasePartition{FT})

Effective latent heat of condensate (weighted sum of liquid and ice),
with specific latent heat evaluated at reference temperature `T_0`.
"""
latent_heat_liq_ice(
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = FT(LH_v0) * q.liq + FT(LH_s0) * q.ice


"""
    liquid_ice_pottemp_given_pressure(T, p, q::PhasePartition)

The liquid-ice potential temperature where
 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function liquid_ice_pottemp_given_pressure(
    T::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp_given_pressure(T, p, q, param_set) *
           (1 - latent_heat_liq_ice(q, param_set) / (cp_m(q, param_set) * T))
end


"""
    liquid_ice_pottemp(T, ρ, q::PhasePartition)

The liquid-ice potential temperature where
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
liquid_ice_pottemp(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = liquid_ice_pottemp_given_pressure(
    T,
    air_pressure(T, ρ, q, param_set),
    q,
    param_set,
)
liquid_ice_pottemp(T::FT, ρ::FT, param_set::PS = MTPS()) where {FT, PS} =
    liquid_ice_pottemp(T, ρ, q_pt_0(FT), param_set)

"""
    liquid_ice_pottemp(ts::ThermodynamicState)

The liquid-ice potential temperature,
given a thermodynamic state `ts`.
"""
liquid_ice_pottemp(ts::ThermodynamicState) = liquid_ice_pottemp(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    dry_pottemp(T, ρ[, q::PhasePartition])

The dry potential temperature where

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
 """
dry_pottemp(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = T / exner(T, ρ, q, param_set)
dry_pottemp(T::FT, ρ::FT, param_set::PS = MTPS()) where {FT, PS} =
    dry_pottemp(T, ρ, q_pt_0(FT), param_set)

"""
    dry_pottemp_given_pressure(T, p[, q::PhasePartition])

The dry potential temperature where

 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
 """
dry_pottemp_given_pressure(
    T::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = T / exner_given_pressure(p, q, param_set)
dry_pottemp_given_pressure(
    T::FT,
    p::FT,
    param_set::PS = MTPS(),
) where {FT, PS} = dry_pottemp_given_pressure(T, p, q_pt_0(FT), param_set)

"""
    dry_pottemp(ts::ThermodynamicState)

The dry potential temperature, given a thermodynamic state `ts`.
"""
dry_pottemp(ts::ThermodynamicState) = dry_pottemp(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    air_temperature_from_liquid_ice_pottemp(θ_liq_ice, ρ, q::PhasePartition)

The temperature given
 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_temperature_from_liquid_ice_pottemp(
    θ_liq_ice::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}

    cvm = cv_m(q, param_set)
    cpm = cp_m(q, param_set)
    R_m = gas_constant_air(q, param_set)
    κ = 1 - cvm / cpm
    T_u = (ρ * R_m * θ_liq_ice / FT(MSLP))^(R_m / cvm) * θ_liq_ice
    T_1 = latent_heat_liq_ice(q, param_set) / cvm
    T_2 = -κ / (2 * T_u) * (latent_heat_liq_ice(q, param_set) / cvm)^2
    return T_u + T_1 + T_2
end
air_temperature_from_liquid_ice_pottemp(
    θ_liq_ice::FT,
    ρ::FT,
    param_set::PS = MTPS(),
) where {FT, PS} =
    air_temperature_from_liquid_ice_pottemp(θ_liq_ice, ρ, q_pt_0(FT), param_set)

"""
    air_temperature_from_liquid_ice_pottemp_non_linear(θ_liq_ice, ρ, q::PhasePartition)

Computes temperature `T` given

 - `θ_liq_ice` liquid-ice potential temperature
 - `ρ` (moist-)air density
 - `tol` tolerance for non-linear equation solve
 - `maxiter` maximum iterations for non-linear equation solve
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air,

by finding the root of
``
  T - air_temperature_from_liquid_ice_pottemp_given_pressure(θ_liq_ice, air_pressure(T, ρ, q), q) = 0
``
"""
function air_temperature_from_liquid_ice_pottemp_non_linear(
    θ_liq_ice::FT,
    ρ::FT,
    maxiter::Int,
    tol::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    sol = find_zero(
        T ->
            T - air_temperature_from_liquid_ice_pottemp_given_pressure(
                θ_liq_ice,
                air_pressure(T, ρ, q, param_set),
                q,
                param_set,
            ),
        FT(T_min),
        FT(T_max),
        SecantMethod(),
        CompactSolution(),
        tol,
        maxiter,
    )
    if !sol.converged
        error("air_temperature_from_liquid_ice_pottemp_non_linear did not converge")
    end
    return sol.root
end
air_temperature_from_liquid_ice_pottemp_non_linear(
    θ_liq_ice::FT,
    ρ::FT,
    maxiter::Int,
    tol::FT,
    param_set::PS,
) where {FT <: Real, PS} = air_temperature_from_liquid_ice_pottemp_non_linear(
    θ_liq_ice,
    ρ,
    maxiter,
    tol,
    q_pt_0(FT),
    param_set,
)

"""
    air_temperature_from_liquid_ice_pottemp_given_pressure(θ_liq_ice, p[, q::PhasePartition])

The air temperature where

 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function air_temperature_from_liquid_ice_pottemp_given_pressure(
    θ_liq_ice::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    return θ_liq_ice * exner_given_pressure(p, q, param_set) +
           latent_heat_liq_ice(q, param_set) / cp_m(q, param_set)
end
air_temperature_from_liquid_ice_pottemp_given_pressure(
    θ_liq_ice::FT,
    p::FT,
    param_set::PS,
) where {FT <: Real, PS} =
    air_temperature_from_liquid_ice_pottemp_given_pressure(
        θ_liq_ice,
        p,
        q_pt_0(FT),
        param_set,
    )

"""
    virtual_pottemp(T, ρ[, q::PhasePartition])

The virtual temperature where

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
virtual_pottemp(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} =
    gas_constant_air(q, param_set) / FT(R_d) * dry_pottemp(T, ρ, q, param_set)
virtual_pottemp(T::FT, ρ::FT, param_set::PS = MTPS()) where {FT, PS} =
    virtual_pottemp(T, ρ, q_pt_0(FT), param_set)

"""
    virtual_pottemp(ts::ThermodynamicState)

The virtual potential temperature,
given a thermodynamic state `ts`.
"""
virtual_pottemp(ts::ThermodynamicState) = virtual_pottemp(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    liquid_ice_pottemp_sat(T, ρ[, q::PhasePartition])

The saturated liquid ice potential temperature where

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function liquid_ice_pottemp_sat(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    q_v_sat = q_vap_saturation(T, ρ, q, param_set)
    return liquid_ice_pottemp(T, ρ, PhasePartition(q_v_sat), param_set)
end
liquid_ice_pottemp_sat(T::FT, ρ::FT, param_set::PS = MTPS()) where {FT, PS} =
    liquid_ice_pottemp_sat(T, ρ, q_pt_0(FT), param_set)

"""
    liquid_ice_pottemp_sat(T, ρ, q_tot)

The saturated liquid ice potential temperature where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
"""
liquid_ice_pottemp_sat(
    T::FT,
    ρ::FT,
    q_tot::FT,
    param_set::PS = MTPS(),
) where {FT <: Real, PS} = liquid_ice_pottemp(
    T,
    ρ,
    PhasePartition_equil(T, ρ, q_tot, param_set),
    param_set,
)

"""
    liquid_ice_pottemp_sat(ts::ThermodynamicState)

The liquid potential temperature given a thermodynamic state `ts`.
"""
liquid_ice_pottemp_sat(ts::ThermodynamicState) = liquid_ice_pottemp_sat(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    exner_given_pressure(p[, q::PhasePartition])

The Exner function where
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
function exner_given_pressure(
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    # gas constant and isobaric specific heat of moist air
    _R_m = gas_constant_air(q, param_set)
    _cp_m = cp_m(q, param_set)

    return (p / FT(MSLP))^(_R_m / _cp_m)
end
exner_given_pressure(p::FT, param_set::PS = MTPS()) where {FT, PS} =
    exner_given_pressure(p, q_pt_0(FT), param_set)

"""
    exner(T, ρ[, q::PhasePartition)])

The Exner function where
 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
"""
exner(
    T::FT,
    ρ::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS} =
    exner_given_pressure(air_pressure(T, ρ, q, param_set), q, param_set)
exner(T::FT, ρ::FT, param_set::PS = MTPS()) where {FT, PS} =
    exner(T, ρ, q_pt_0(FT), param_set)

"""
    exner(ts::ThermodynamicState)

The Exner function, given a thermodynamic state `ts`.
"""
exner(ts::ThermodynamicState) = exner(
    air_temperature(ts),
    air_density(ts),
    PhasePartition(ts),
    ts.param_set,
)

"""
    relative_humidity(T, p, e_int, q::PhasePartition)

The relative humidity, given
 - `T` temperature
 - `p` pressure
 - `e_int` internal energy per unit mass
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument, the results are for dry air.
 """
function relative_humidity(
    T::FT,
    p::FT,
    e_int::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT <: Real, PS}
    q_vap = q.tot - q.liq - q.ice
    p_vap =
        q_vap *
        air_density(T, p, q, param_set) *
        FT(R_v) *
        air_temperature(e_int, q, param_set)
    liq_frac = liquid_fraction(T, q, param_set)
    p_vap_sat =
        liq_frac * saturation_vapor_pressure(T, Liquid(), param_set) +
        (1 - liq_frac) * saturation_vapor_pressure(T, Ice(), param_set)
    return p_vap / p_vap_sat
end
relative_humidity(
    T::FT,
    p::FT,
    e_int::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
    param_set::PS = MTPS(),
) where {FT, PS} = relative_humidity(T, p, e_int, q_pt_0(FT), param_set)


"""
    relative_humidity(ts::ThermodynamicState)

The relative humidity, given a thermodynamic state `ts`.
"""
relative_humidity(ts::ThermodynamicState{FT}) where {FT <: Real} =
    relative_humidity(
        air_temperature(ts),
        air_pressure(ts),
        internal_energy(ts),
        PhasePartition(ts),
        ts.param_set,
    )

end #module MoistThermodynamics.jl
