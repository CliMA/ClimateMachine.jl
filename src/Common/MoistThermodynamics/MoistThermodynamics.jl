"""
    MoistThermodynamics

Moist thermodynamic functions, e.g., for air pressure (atmosphere equation
of state), latent heats of phase transitions, saturation vapor pressures, and
saturation specific humidities
"""
module MoistThermodynamics

using DocStringExtensions

using ..RootSolvers
using ..PlanetParameters

# Atmospheric equation of state
export air_pressure, air_temperature, air_density, specific_volume, soundspeed_air
export linearized_air_pressure

# Energies
export total_energy, internal_energy, internal_energy_sat

# Specific heats of moist air
export cp_m, cv_m, gas_constant_air, moist_gas_constants

# Latent heats
export latent_heat_vapor, latent_heat_sublim, latent_heat_fusion

# Saturation vapor pressures and specific humidities over liquid and ice
export Liquid, Ice
export saturation_vapor_pressure, q_vap_saturation_generic, q_vap_saturation
export saturation_excess

# Functions used in thermodynamic equilibrium among phases (liquid and ice
# determined diagnostically from total water specific humidity)
export liquid_fraction_equil, liquid_fraction_nonequil, saturation_adjustment, PhasePartition_equil

# Auxiliary functions, e.g., for diagnostic purposes
export air_temperature_from_liquid_ice_pottemp, dry_pottemp, virtual_pottemp, exner
export liquid_ice_pottemp, liquid_ice_pottemp_sat, relative_humidity

include("states.jl")


"""
    gas_constant_air([q::PhasePartition])

The specific gas constant of moist air given
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
gas_constant_air(q::PhasePartition{FT}) where {FT<:Real} =
  FT(R_d) * ( 1 +  (FT(molmass_ratio) - 1)*q.tot - FT(molmass_ratio)*(q.liq + q.ice) )
gas_constant_air(::Type{FT}) where {FT<:Real} = gas_constant_air(PhasePartition{FT}(FT(0), FT(0), FT(0)))

"""
    gas_constant_air(ts::ThermodynamicState)

The specific gas constant of moist air given
a thermodynamic state `ts`.
"""
gas_constant_air(ts::ThermodynamicState) =
  gas_constant_air(PhasePartition(ts))
gas_constant_air(ts::PhaseDry{FT}) where {FT<:Real} = FT(R_d)


"""
    air_pressure(T, ρ[, q::PhasePartition])

The air pressure from the equation of state
(ideal gas law) where:

 - `T` air temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
air_pressure(T::FT, ρ::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  gas_constant_air(q) * ρ * T

"""
    air_pressure(ts::ThermodynamicState)

The air pressure from the equation of state
(ideal gas law), given a thermodynamic state `ts`.
"""
air_pressure(ts::ThermodynamicState) =
  air_pressure(air_temperature(ts), air_density(ts), PhasePartition(ts))
air_pressure(ts::PhaseDry) = air_density(ts) * gas_constant_air(ts) * air_temperature(ts)

"""
    linearized_air_pressure(ρ, e_tot, e_pot[, q::PhasePartition])

The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:

 - `ρ` (moist-)air density
 - `e_tot` total energy per unit mass
 - `e_pot` potential energy per unit mass
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
linearized_air_pressure(ρ::FT, e_tot::FT, e_pot::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  ρ*FT(R_d)*FT(T_0) + FT(R_d)/FT(cv_d)*(ρ*e_tot - ρ*e_pot - (ρ*q.tot - ρ*q.liq)*FT(e_int_v0) + ρ*q.ice*(FT(e_int_i0) + FT(e_int_v0)))

linearized_air_pressure(e_kin::FT, e_pot::FT, ts::ThermodynamicState{FT}) where {FT<:Real} =
  linearized_air_pressure(air_density(ts), total_energy(e_kin, e_pot, ts), e_pot, PhasePartition(ts))

function linearized_air_pressure(e_kin::FT, e_pot::FT, ts::PhaseDry{FT}) where {FT<:Real}
  ρ = air_density(ts)
  e_tot = total_energy(e_kin, e_pot, ts)
  return ρ*FT(R_d)*FT(T_0) + FT(R_d)/FT(cv_d)*(ρ*e_tot - ρ*e_pot)
end

"""
    air_density(T, p[, q::PhasePartition])

The (moist-)air density from the equation of state
(ideal gas law) where

 - `T` air temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
air_density(T::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  p / (gas_constant_air(q) * T)

"""
    air_density(ts::ThermodynamicState)

The (moist-)air density from the equation of state
(ideal gas law), given a thermodynamic state `ts`.
"""
air_density(ts::ThermodynamicState) = ts.ρ

"""
    specific_volume(T, p[, q::PhasePartition])

The (moist-)air specific volume from the equation of
state (ideal gas law) where

 - `T` air temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
specific_volume(T::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  (gas_constant_air(q) * T) / p

"""
    specific_volume(ts::ThermodynamicState)

The (moist-)air specific volume from the equation of
state (ideal gas law), given a thermodynamic state `ts`.
"""
specific_volume(ts::ThermodynamicState) = specific_volume(air_temperature(ts), air_pressure(ts), PhasePartition(ts))
specific_volume(ts::PhaseDry) = (gas_constant_air(ts) * air_temperature(ts)) / air_pressure(ts)

"""
    cp_m([q::PhasePartition])

The isobaric specific heat capacity of moist
air where, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
cp_m(q::PhasePartition{FT}) where {FT<:Real} =
  FT(cp_d) + (FT(cp_v) - FT(cp_d))*q.tot + (FT(cp_l) - FT(cp_v))*q.liq + (FT(cp_i) - FT(cp_v))*q.ice
cp_m(::Type{FT}) where {FT<:Real} = cp_m(PhasePartition{FT}(FT(0), FT(0), FT(0)))

"""
    cp_m(ts::ThermodynamicState)

The isobaric specific heat capacity of moist
air, given a thermodynamic state `ts`.
"""
cp_m(ts::ThermodynamicState) = cp_m(PhasePartition(ts))
cp_m(ts::PhaseDry{FT}) where {FT<:Real} = FT(cp_d)

"""
    cv_m([q::PhasePartition])

The isochoric specific heat capacity of moist
air where optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
cv_m(q::PhasePartition{FT}) where {FT<:Real} =
  FT(cv_d) + (FT(cv_v) - FT(cv_d))*q.tot + (FT(cv_l) - FT(cv_v))*q.liq + (FT(cv_i) - FT(cv_v))*q.ice
cv_m(::Type{FT}) where {FT<:Real} = cv_m(PhasePartition{FT}(FT(0), FT(0), FT(0)))

"""
    cv_m(ts::ThermodynamicState)

The isochoric specific heat capacity of moist
air given a thermodynamic state `ts`.
"""
cv_m(ts::ThermodynamicState) = cv_m(PhasePartition(ts))
cv_m(ts::PhaseDry{FT}) where {FT<:Real} = FT(cv_d)


"""
    (R_m, cp_m, cv_m, γ_m) = moist_gas_constants([q::PhasePartition])

Wrapper to compute all gas constants at once, where optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`

Without the specific humidity arguments, the results
are that of dry air.
"""
function moist_gas_constants(q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
    R_gas  = gas_constant_air(q)
    cp = cp_m(q)
    cv = cv_m(q)
    γ = cp/cv
    return (R_gas, cp, cv, γ)
end

"""
    (R_m, cp_m, cv_m, γ_m) = moist_gas_constants(ts::ThermodynamicState)

Wrapper to compute all gas constants at once, given a thermodynamic state `ts`.

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`

"""
moist_gas_constants(ts::ThermodynamicState) = moist_gas_constants(PhasePartition(ts))
function moist_gas_constants(ts::PhaseDry)
    R_gas  = gas_constant_air(ts)
    cp = cp_m(ts)
    cv = cv_m(ts)
    γ = cp/cv
    return (R_gas, cp, cv, γ)
end

"""
    air_temperature(e_int, q::PhasePartition)

The air temperature, where

 - `e_int` internal energy per unit mass
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
function air_temperature(e_int::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
  T_0 +
    (e_int - (q.tot - q.liq) * FT(e_int_v0) + q.ice * (FT(e_int_v0) + FT(e_int_i0))) /
    cv_m(q)
end

"""
    air_temperature(ts::ThermodynamicState)

The air temperature, given a thermodynamic state `ts`.
"""
air_temperature(ts::ThermodynamicState) = air_temperature(ts.e_int, PhasePartition(ts))
air_temperature(ts::PhaseDry{FT}) where {FT<:Real} = FT(T_0) + ts.e_int / cv_m(ts)
air_temperature(ts::PhaseEquil) = ts.T


"""
    internal_energy(T[, q::PhasePartition])

The internal energy per unit mass, given a thermodynamic state `ts` or

 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
internal_energy(T::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  cv_m(q) * (T - FT(T_0)) +
  (q.tot - q.liq) * FT(e_int_v0) - q.ice * (FT(e_int_v0) + FT(e_int_i0))

"""
    internal_energy(ts::ThermodynamicState)

The internal energy per unit mass, given a thermodynamic state `ts`.
"""
internal_energy(ts::PhaseEquil) = ts.e_int
internal_energy(ts::PhaseNonEquil) = ts.e_int
internal_energy(ts::PhaseDry) = ts.e_int

"""
    internal_energy(ρ::FT, ρe::FT, ρu::AbstractVector{FT}, e_pot::FT)

The internal energy per unit mass, given
 - `ρ` (moist-)air density
 - `ρe` total energy
 - `ρu` momentum vector
 - `e_pot` potential energy (e.g., gravitational) per unit mass
"""
@inline function internal_energy(ρ::FT, ρe::FT, ρu::AbstractVector{FT}, e_pot::FT) where {FT<:Real}
  ρinv = 1 / ρ
  ρe_kin = ρinv*sum(abs2, ρu)/2
  ρe_pot = ρ * e_pot
  ρe_int = ρe - ρe_kin - ρe_pot
  e_int = ρinv*ρe_int
  return e_int
end

"""
    internal_energy_sat(T, ρ, q_tot)

The internal energy per unit mass in thermodynamic equilibrium at saturation where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
"""
internal_energy_sat(T::FT, ρ::FT, q_tot::FT) where {FT<:Real} =
  internal_energy(T, PhasePartition_equil(T, ρ, q_tot))

"""
    internal_energy_sat(ts::ThermodynamicState)

The internal energy per unit mass in
thermodynamic equilibrium at saturation,
given a thermodynamic state `ts`.
"""
internal_energy_sat(ts::PhaseDry{FT}) where {FT<:Real} = cv_m(ts) * (air_temperature(ts) - FT(T_0))
internal_energy_sat(ts::PhaseEquil) =
  internal_energy_sat(air_temperature(ts), air_density(ts), ts.q_tot)
internal_energy_sat(ts::PhaseNonEquil) =
  internal_energy_sat(air_temperature(ts), air_density(ts), ts.q.tot)


"""
    total_energy(e_kin, e_pot, T[, q::PhasePartition])

The total energy per unit mass, given

 - `e_kin` kinetic energy per unit mass
 - `e_pot` potential energy per unit mass
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.

"""
total_energy(e_kin::FT, e_pot::FT, T::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  e_kin + e_pot + internal_energy(T, q)

"""
    total_energy(e_kin, e_pot, ts::ThermodynamicState)

The total energy per unit mass
given a thermodynamic state `ts`.
"""
total_energy(e_kin::FT, e_pot::FT, ts::ThermodynamicState{FT}) where {FT<:Real} = internal_energy(ts) + FT(e_kin) + FT(e_pot)

"""
    soundspeed_air(T[, q::PhasePartition])

The speed of sound in air, where
 - `T` temperature
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.

Without the specific humidity arguments, the results
are that of dry air.
"""
function soundspeed_air(T::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
  γ   = cp_m(q) / cv_m(q)
  R_m = gas_constant_air(q)
  return sqrt(γ*R_m*T)
end

"""
    soundspeed_air(ts::ThermodynamicState)

The speed of sound in air given a thermodynamic state `ts`.
"""
soundspeed_air(ts::ThermodynamicState) =
  soundspeed_air(air_temperature(ts), PhasePartition(ts))
soundspeed_air(ts::PhaseDry) = sqrt(cp_m(ts)/cv_m(ts)*gas_constant_air(ts)*air_temperature(ts))


"""
    latent_heat_vapor(T::FT) where {FT<:Real}

The specific latent heat of vaporization where
 - `T` temperature
"""
latent_heat_vapor(T::FT) where {FT<:Real} =
  latent_heat_generic(T, FT(LH_v0), FT(cp_v) - FT(cp_l))

"""
    latent_heat_vapor(ts::ThermodynamicState)

The specific latent heat of vaporization
given a thermodynamic state `ts`.
"""
latent_heat_vapor(ts::ThermodynamicState) =
    latent_heat_vapor(air_temperature(ts))

"""
    latent_heat_sublim(T::FT) where {FT<:Real}

The specific latent heat of sublimation where
 - `T` temperature
"""
latent_heat_sublim(T::FT) where {FT<:Real} =
  latent_heat_generic(T, FT(LH_s0), FT(cp_v) - FT(cp_i))

"""
    latent_heat_sublim(ts::ThermodynamicState)

The specific latent heat of sublimation
given a thermodynamic state `ts`.
"""
latent_heat_sublim(ts::ThermodynamicState) =
  latent_heat_sublim(air_temperature(ts))

"""
    latent_heat_fusion(T::FT) where {FT<:Real}

The specific latent heat of fusion where
 - `T` temperature
"""
latent_heat_fusion(T::FT) where {FT<:Real} =
  latent_heat_generic(T, FT(LH_f0), FT(cp_l) - FT(cp_i))

"""
    latent_heat_fusion(ts::ThermodynamicState)

The specific latent heat of fusion
given a thermodynamic state `ts`.
"""
latent_heat_fusion(ts::ThermodynamicState{FT}) where {FT<:Real} =
  latent_heat_fusion(air_temperature(ts))

"""
    latent_heat_generic(T::FT, LH_0::FT, Δcp::FT) where {FT<:Real}

Return the specific latent heat of a generic phase transition between
two phases using Kirchhoff's relation.

The latent heat computation assumes constant isobaric specifc heat capacities
of the two phases. `T` is the temperature, `LH_0` is the latent heat of the
phase transition at `T_0`, and `Δcp` is the difference between the isobaric
specific heat capacities (heat capacity in the higher-temperature phase minus
that in the lower-temperature phase).
"""
latent_heat_generic(T::FT, LH_0::FT, Δcp::FT) where {FT<:Real} =
  LH_0 + Δcp * (T - FT(T_0))


"""
    Phase

A phase condensate, to dispatch over
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
of the Clausius-Clepeyron relation.

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
saturation_vapor_pressure(T::FT, ::Liquid) where {FT<:Real} = saturation_vapor_pressure(T, FT(LH_v0), FT(cp_v) - FT(cp_l))
function saturation_vapor_pressure(ts::ThermodynamicState{FT}, ::Liquid) where {FT<:Real}

    return saturation_vapor_pressure(air_temperature(ts), FT(LH_v0), FT(cp_v) - FT(cp_l))

end
saturation_vapor_pressure(T::FT, ::Ice) where {FT<:Real} = saturation_vapor_pressure(T, FT(LH_s0), FT(cp_v) - FT(cp_i))
saturation_vapor_pressure(ts::ThermodynamicState{FT}, ::Ice) where {FT<:Real} = saturation_vapor_pressure(air_temperature(ts), FT(LH_s0), FT(cp_v) - FT(cp_i))

function saturation_vapor_pressure(T::FT, LH_0::FT, Δcp::FT) where {FT<:Real}

    return FT(press_triple) * (T/FT(T_triple))^(Δcp/FT(R_v)) *
        exp( (FT(LH_0) - Δcp*FT(T_0))/FT(R_v) * (1 / FT(T_triple) - 1 / T) )

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
function q_vap_saturation_generic(T::FT, ρ::FT; phase::Phase=Liquid()) where {FT<:Real}
    p_v_sat = saturation_vapor_pressure(T, phase)
    return q_vap_saturation_from_pressure(T, ρ, p_v_sat)
end

"""
    q_vap_saturation(T, ρ[, q::PhasePartition])

Compute the saturation specific humidity, given

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q` [`PhasePartition`](@ref)

If the phase partition `q` is given, the saturation specific humidity is that over a
mixture of liquid and ice, computed in a thermodynamically consistent way from the
weighted sum of the latent heats of the respective phase transitions (Pressel et al.,
JAMES, 2015).  That is, the saturation vapor pressure and from it the saturation specific
humidity are computed from a weighted mean of the latent heats of vaporization and
sublimation, with the weights given by the fractions of condensate `q.liq/(q.liq + q.ice)`
and `q.ice/(q.liq + q.ice)` that are liquid and ice, respectively.

If the condensate specific humidities `q.liq` and `q.ice` are not given or are both
zero, the saturation specific humidity is that over a mixture of liquid and ice,
with the fraction of liquid given by temperature dependent `liquid_fraction_equil(T)`
and the fraction of ice by the complement `1 - liquid_fraction_equil(T)`.
"""
function q_vap_saturation(T::FT, ρ::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0),FT(0),FT(0))) where {FT<:Real}

    # get phase partitioning
    _liquid_frac = liquid_fraction_equil(T, q)
    _ice_frac    = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0    = _liquid_frac * FT(LH_v0) + _ice_frac * FT(LH_s0)
    Δcp     = _liquid_frac * (FT(cp_v) - FT(cp_l)) + _ice_frac * (FT(cp_v) - FT(cp_i))

    # saturation vapor pressure over possible mixture of liquid and ice
    p_v_sat = saturation_vapor_pressure(T, FT(LH_0), Δcp)

    return q_vap_saturation_from_pressure(T, ρ, p_v_sat)

end

"""
    q_vap_saturation(ts::ThermodynamicState)

Compute the saturation specific humidity, given a thermodynamic state `ts`.
"""
q_vap_saturation(ts::ThermodynamicState) =
  q_vap_saturation(air_temperature(ts), air_density(ts), PhasePartition(ts))
function q_vap_saturation(ts::PhaseDry{FT}) where {FT<:Real}
    # get phase partitioning
    T = air_temperature(ts)
    ρ = air_density(ts)
    _liquid_frac = liquid_fraction_equil(ts)
    _ice_frac    = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0    = _liquid_frac * FT(LH_v0) + _ice_frac * FT(LH_s0)
    Δcp     = _liquid_frac * (FT(cp_v) - FT(cp_l)) + _ice_frac * (FT(cp_v) - FT(cp_i))

    # saturation vapor pressure over possible mixture of liquid and ice
    p_v_sat = saturation_vapor_pressure(T, FT(LH_0), Δcp)

    return q_vap_saturation_from_pressure(T, ρ, p_v_sat)
end

"""
    q_vap_saturation_from_pressure(T, ρ, p_v_sat)

Compute the saturation specific humidity, given

 - `T` ambient air temperature,
 - `ρ` density
 - `p_v_sat` saturation vapor pressure
"""
q_vap_saturation_from_pressure(T::FT, ρ::FT, p_v_sat::FT) where {FT<:Real} =
  min(1, p_v_sat / (ρ * FT(R_v) * T))

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
saturation_excess(T::FT, ρ::FT, q::PhasePartition{FT}) where {FT<:Real} =
  max(0, q.tot - q_vap_saturation(T, ρ, q))

"""
    saturation_excess(ts::ThermodynamicState)

Compute the saturation excess in equilibrium,
given a thermodynamic state `ts`.
"""
saturation_excess(ts::ThermodynamicState) =
  saturation_excess(air_temperature(ts), air_density(ts), PhasePartition(ts))
saturation_excess(ts::PhaseDry) = max(0, - q_vap_saturation(ts))

"""
    liquid_fraction_equil(T[, q::PhasePartition])

The fraction of condensate, assuming phase equilibrium, that is liquid where

 - `T` temperature
 - `q` [`PhasePartition`](@ref)

If `q.liq` or `q.ice` are nonzero, the liquid fraction is computed from
them.

Otherwise, the fraction of liquid is a function that is 1 above `T_freeze` and goes to
zero below `T_freeze`.
"""
function liquid_fraction_equil(T::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
  q_c = q.liq + q.ice     # condensate specific humidity
  if q_c > 0
    return q.liq / q_c
  else
    # For now: Heaviside function for partitioning into liquid and ice: all liquid
    # for T > T_freeze; all ice for T <= T_freeze
    return FT(T > FT(T_freeze))
  end
end

"""
    liquid_fraction_equil(ts::ThermodynamicState)

The fraction of condensate that is liquid given a thermodynamic state `ts`.
"""
liquid_fraction_equil(ts::ThermodynamicState) =
  liquid_fraction_equil(air_temperature(ts), PhasePartition(ts))
liquid_fraction_equil(ts::PhaseDry{FT}) where {FT<:Real} = FT(air_temperature(ts) > FT(T_freeze))

"""
    liquid_fraction_nonequil(T[, q::PhasePartition])

The fraction of condensate, assuming phase non-equilibrium, that is liquid where

 - `T` temperature
 - `q` [`PhasePartition`](@ref)

If `q.liq` or `q.ice` are nonzero, the liquid fraction is computed from
them.

Otherwise, the fraction of liquid is a function that is 1 above `T_freeze` and goes to
zero below `T_freeze`.

!!! todo
    Currently [`liquid_fraction_nonequil`](@ref) calls [`liquid_fraction_equil`](@ref),
    but we should implement a more general function here.
"""
liquid_fraction_nonequil(T::FT, q::PhasePartition{FT}) where {FT<:Real} =
  liquid_fraction_equil(T, q)

"""
    liquid_fraction_nonequil(ts::ThermodynamicState)

The fraction of condensate that is liquid given a thermodynamic state `ts`.
"""
liquid_fraction_nonequil(ts::ThermodynamicState) =
  liquid_fraction_nonequil(air_temperature(ts), PhasePartition(ts))
liquid_fraction_nonequil(ts::PhaseDry) = liquid_fraction_equil(ts)

"""
    PhasePartition_equil(T, ρ, q_tot)

Partition the phases in equilibrium, returning a [`PhasePartition`](@ref) object using the
[`liquid_fraction_equil`](@ref) function where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity

The residual `q.tot - q.liq - q.ice` is the vapor specific humidity.
"""
function PhasePartition_equil(T::FT, ρ::FT, q_tot::FT) where {FT<:Real}
    _liquid_frac = liquid_fraction_equil(T)   # fraction of condensate that is liquid
    q_c   = saturation_excess(T, ρ, PhasePartition(q_tot))   # condensate specific humidity
    q_liq = _liquid_frac * q_c  # liquid specific humidity
    q_ice = (1 - _liquid_frac) * q_c # ice specific humidity

    return PhasePartition(q_tot, q_liq, q_ice)
end
PhasePartition_equil(ts::PhaseNonEquil) = PhasePartition_equil(air_temperature(ts), air_density(ts), ts.q_tot)

PhasePartition(ts::PhaseDry{FT}) where {FT<:Real} = PhasePartition(FT(0), FT(0), FT(0))
PhasePartition(ts::PhaseEquil) = PhasePartition_equil(air_temperature(ts), air_density(ts), ts.q_tot)
PhasePartition(ts::PhaseNonEquil) = ts.q

"""
    saturation_adjustment(e_int, ρ, q_tot)

Compute the temperature that is consistent with

 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity

See also [`saturation_adjustment_q_t_θ_l`](@ref).
"""
function saturation_adjustment(e_int::FT, ρ::FT, q_tot::FT) where {FT<:Real}
  T_1 = max(FT(T_min), air_temperature(e_int, PhasePartition(q_tot))) # Assume all vapor
  q_v_sat = q_vap_saturation(T_1, ρ)
  if q_tot <= q_v_sat # If not saturated return T_1
    return T_1
  else # If saturated, iterate
    # FIXME here: need to revisit bounds for saturation adjustment to guarantee bracketing of zero.
    T_2 = air_temperature(e_int, PhasePartition(q_tot, FT(0), q_tot)) # Assume all ice
    T, converged = find_zero(
      T -> internal_energy_sat(T, ρ, q_tot) - e_int,
      T_1, T_2, SecantMethod(), FT(1e-3), 10)
      if !converged
        error("saturation adjustment did not converge")
      end
    return T
  end
end


"""
    saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ, p)

Compute the temperature that is consistent with

 - `θ_liq_ice` liquid-ice potential temperature
 - `q_tot` total specific humidity
 - `ρ` density
 - `p` pressure

See also [`saturation_adjustment`](@ref).
"""
function saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice::FT, q_tot::FT, ρ::FT, p::FT) where {FT<:Real}
  T_1 = air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p) # Assume all vapor
  q_v_sat = q_vap_saturation(T_1, ρ)
  if q_tot <= q_v_sat # If not saturated
    return T_1
  else  # If saturated, iterate
    T_2 = air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p, PhasePartition(q_tot, FT(0), q_tot)) # Assume all ice
    T, converged = find_zero(
      T -> θ_liq_ice - liquid_ice_pottemp_sat(T, p, PhasePartition_equil(T, ρ, q_tot)),
      T_1, T_2, SecantMethod(), FT(1e-3), 10)
    return T
  end
end

"""
    liquid_ice_pottemp(T, p, q::PhasePartition)

The liquid-ice potential temperature where
 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
function liquid_ice_pottemp(T::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp(T, p, q) * (1 - (FT(LH_v0)*q.liq + FT(LH_s0)*q.ice)/(cp_m(q)*T))
end

"""
    liquid_ice_pottemp(ts::ThermodynamicState)

The liquid-ice potential temperature,
given a thermodynamic state `ts`.
"""
liquid_ice_pottemp(ts::ThermodynamicState) =
  liquid_ice_pottemp(air_temperature(ts), air_pressure(ts), PhasePartition(ts))
liquid_ice_pottemp(ts::PhaseDry) = dry_pottemp(ts)

"""
    dry_pottemp(T, p[, q::PhasePartition])

The dry potential temperature where

 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
 """
dry_pottemp(T::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  T / exner(p, q)

"""
    dry_pottemp(ts::ThermodynamicState)

The dry potential temperature, given a thermodynamic state `ts`.
"""
dry_pottemp(ts::ThermodynamicState) =
  dry_pottemp(air_temperature(ts), air_pressure(ts), PhasePartition(ts))
dry_pottemp(ts::PhaseDry) = air_temperature(ts) / exner(ts)

"""
    air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p[, q::PhasePartition])

The air temperature, where

 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
air_temperature_from_liquid_ice_pottemp(θ_liq_ice::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  θ_liq_ice*exner(p, q) + (FT(LH_v0)*q.liq + FT(LH_s0)*q.ice) / cp_m(q)

"""
    virtual_pottemp(T, p[, q::PhasePartition])

The virtual temperature where

 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
virtual_pottemp(T::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real} =
  gas_constant_air(q) / FT(R_d) * dry_pottemp(T, p, q)

"""
    virtual_pottemp(ts::ThermodynamicState)

The virtual potential temperature,
given a thermodynamic state `ts`.
"""
virtual_pottemp(ts::ThermodynamicState) =
  virtual_pottemp(air_temperature(ts), air_pressure(ts), PhasePartition(ts))
virtual_pottemp(ts::PhaseDry{FT}) where {FT<:Real} = gas_constant_air(ts) / FT(R_d) * dry_pottemp(ts)

"""
    liquid_ice_pottemp_sat(T, p[, q::PhasePartition])

The saturated liquid ice potential temperature where

 - `T` temperature
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
function liquid_ice_pottemp_sat(T::FT, p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
    ρ = air_density(T, p, q)
    q_v_sat = q_vap_saturation(T, ρ, q)
    return liquid_ice_pottemp(T, p, PhasePartition(q_v_sat))
end

"""
    liquid_ice_pottemp_sat(ts::ThermodynamicState)

The liquid potential temperature given a thermodynamic state `ts`.
"""
liquid_ice_pottemp_sat(ts::ThermodynamicState) =
  liquid_ice_pottemp_sat(air_temperature(ts), air_pressure(ts), PhasePartition(ts))
liquid_ice_pottemp_sat(ts::PhaseDry) =
  liquid_ice_pottemp(air_temperature(ts), air_pressure(ts), PhasePartition(q_vap_saturation(ts)))

"""
    exner(p[, q::PhasePartition])

The Exner function where
 - `p` pressure
and, optionally,
 - `q` [`PhasePartition`](@ref). Without this argument the results are that of dry air.
"""
function exner(p::FT, q::PhasePartition{FT}=PhasePartition{FT}(FT(0), FT(0), FT(0))) where {FT<:Real}
    # gas constant and isobaric specific heat of moist air
    _R_m    = gas_constant_air(q)
    _cp_m   = cp_m(q)

    return (p/FT(MSLP))^(_R_m/_cp_m)
end

"""
    exner(ts::ThermodynamicState)

The Exner function, given a thermodynamic state `ts`.
"""
exner(ts::ThermodynamicState) =
  exner(air_pressure(ts), PhasePartition(ts))
exner(ts::PhaseDry{FT}) where {FT<:Real} = (air_pressure(ts)/FT(MSLP))^(gas_constant_air(ts)/cp_m(ts))

"""
    relative_humidity(ts::ThermodynamicState)

The relative humidity, given a thermodynamic state `ts`.
"""
relative_humidity(ts::ThermodynamicState) = air_pressure(ts)/saturation_vapor_pressure(ts, Liquid())

end #module MoistThermodynamics.jl
