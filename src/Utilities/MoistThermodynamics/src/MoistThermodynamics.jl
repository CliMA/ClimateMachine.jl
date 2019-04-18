"""
    MoistThermodynamics

Moist thermodynamic functions, e.g., for air pressure (atmosphere equation
of state), latent heats of phase transitions, saturation vapor pressures, and
saturation specific humidities
"""
module MoistThermodynamics

using DocStringExtensions

using ..RootSolvers
using ...PlanetParameters

# Atmospheric equation of state
export air_pressure, air_temperature, air_density, specific_volume, soundspeed_air

# Energies
export total_energy, internal_energy, internal_energy_sat

# Specific heats of moist air
export cp_m, cv_m, gas_constant_air, moist_gas_constants

# Latent heats
export latent_heat_vapor, latent_heat_sublim, latent_heat_fusion

# Saturation vapor pressures and specific humidities over liquid and ice
export Liquid, Ice
export saturation_vapor_pressure, saturation_shum_generic, saturation_shum
export saturation_excess

# Functions used in thermodynamic equilibrium among phases (liquid and ice
# determined diagnostically from total water specific humidity)
export liquid_fraction_equil, liquid_fraction_nonequil, saturation_adjustment, phase_partitioning_eq

# Auxiliary functions, e.g., for diagnostic purposes
export air_temperature_from_liquid_ice_pottemp, dry_pottemp, virtual_pottemp, exner
export liquid_ice_pottemp, liquid_ice_pottemp_sat

include("states.jl")


"""
    gas_constant_air([q_tot=0, q_liq=0, q_ice=0])

The specific gas constant of moist air given

 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
gas_constant_air(q_tot::DT=0.0, q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real} =
  DT(R_d) * ( 1 +  (DT(molmass_ratio) - 1)*q_tot - DT(molmass_ratio)*(q_liq + q_ice) )
gas_constant_air(::Type{DT}) where {DT <: Real} =
  gas_constant_air(DT(0))

"""
    gas_constant_air(ts::ThermodynamicState)

The specific gas constant of moist air given
a thermodynamic state `ts`.
"""
gas_constant_air(ts::ThermodynamicState) =
  gas_constant_air(ts.q_tot, phase_partitioning_eq(ts)...)


"""
    air_pressure(T, ρ[, q_tot=0, q_liq=0, q_ice=0])

The air pressure from the equation of state
(ideal gas law) where:

 - `T` air temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
air_pressure(T::DT, ρ::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real} =
  gas_constant_air(q_tot, q_liq, q_ice) * ρ * T

"""
    air_pressure(ts::ThermodynamicState)

The air pressure from the equation of state
(ideal gas law), given a thermodynamic state `ts`.
"""
air_pressure(ts::ThermodynamicState) =
  air_pressure(air_temperature(ts), air_density(ts), phase_partitioning_eq(ts)...)


"""
    air_density(T, p[, q_tot=0, q_liq=0, q_ice=0])

The (moist-)air density from the equation of state
(ideal gas law) where

 - `T` air temperature
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
air_density(T::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real} =
  p / (gas_constant_air(q_tot, q_liq, q_ice) * T)


"""
    air_density(ts::ThermodynamicState)

The (moist-)air density from the equation of state
(ideal gas law), given a thermodynamic state `ts`.
"""
air_density(ts::ThermodynamicState) = ts.density

"""
    specific_volume(T, p[, q_tot=0, q_liq=0, q_ice=0])

The (moist-)air specific volume from the equation of
state (ideal gas law) where

 - `T` air temperature
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments,the results
are that of dry air.
"""
specific_volume(T::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real} =
  (gas_constant_air(q_tot, q_liq, q_ice) * T) / p

"""
    specific_volume(ts::ThermodynamicState)

The (moist-)air specific volume from the equation of
state (ideal gas law), given a thermodynamic state `ts`.
"""
specific_volume(ts::ThermodynamicState) = 1/ts.density

"""
    cp_m([q_tot=0, q_liq=0, q_ice=0])

The isobaric specific heat capacity of moist
air where, optionally,

 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
cp_m(q_tot::DT=0.0, q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT <: Real} =
  DT(cp_d) + (DT(cp_v) - DT(cp_d))*q_tot + (DT(cp_l) - DT(cp_v))*q_liq + (DT(cp_i) - DT(cp_v))*q_ice
cp_m(::Type{DT}) where DT = cp_m(DT(0))

"""
    cp_m(ts::[q_tot=0, q_liq=0, q_ice=0]))

The isobaric specific heat capacity of moist
air, given a thermodynamic state `ts`.
"""
cp_m(ts::ThermodynamicState) =
  cp_m(ts.q_tot, phase_partitioning_eq(ts)...)


"""
    cv_m([q_tot=0, q_liq=0, q_ice=0])

The isochoric specific heat capacity of moist
air where, optionally,

 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
cv_m(q_tot::DT=0.0, q_liq::DT=0, q_ice::DT=0) where {DT <: Real} =
  DT(cv_d) + (DT(cv_v) - DT(cv_d))*q_tot + (DT(cv_l) - DT(cv_v))*q_liq + (DT(cv_i) - DT(cv_v))*q_ice
cv_m(::Type{DT}) where {DT<:Real} = cv_m(DT(0))

"""
    cv_m(ts::ThermodynamicState)

The isochoric specific heat capacity of moist
air given a thermodynamic state `ts`.
"""
cv_m(ts::ThermodynamicState) =
  cv_m(ts.q_tot, phase_partitioning_eq(ts)...)


"""
    (R_m, cp_m, cv_m, γ_m) = moist_gas_constants([q_tot=0, q_liq=0, q_ice=0])

Wrapper to compute all gas constants at once, where optionally,

 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

The function returns a tuple of
 - `R_m` [`gas_constant_air`](@ref)
 - `cp_m` [`cp_m`](@ref)
 - `cv_m` [`cv_m`](@ref)
 - `γ_m = cp_m/cv_m`

Without the specific humidity arguments, the results
are that of dry air.
"""
function moist_gas_constants(q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where DT

    R_gas  = gas_constant_air(q_tot, q_liq, q_ice)
    cp = cp_m(q_tot, q_liq, q_ice)
    cv = cv_m(q_tot, q_liq, q_ice)
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
moist_gas_constants(ts::ThermodynamicState) =
  moist_gas_constants(ts.q_tot, phase_partitioning_eq(ts)...)

"""
    air_temperature(e_int[, q_tot=0, q_liq=0, q_ice=0])

The air temperature, where

 - `e_int` internal energy per unit mass
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
function air_temperature(e_int::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where DT
  T_0 +
    (e_int - (q_tot - q_liq) * DT(e_int_v0) + q_ice * (DT(e_int_v0) + DT(e_int_i0)) ) /
    cv_m(q_tot, q_liq, q_ice)
end

"""
    air_temperature(ts::ThermodynamicState)

The air temperature, given a thermodynamic state `ts`.
"""
air_temperature(ts::PhaseEquil) = ts.T
air_temperature(ts::PhaseNonEquil) =
  air_temperature(ts.e_int, ts.q_tot, ts.q_liq, ts.q_ice)


"""
    internal_energy(T[, q_tot=0, q_liq=0, q_ice=0])

The internal energy per unit mass, given a thermodynamic state `ts` or

 - `T` temperature
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
internal_energy(T::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real} =
  cv_m(q_tot, q_liq, q_ice) * (T - DT(T_0)) +
        (q_tot - q_liq) * DT(e_int_v0) - q_ice * (DT(e_int_v0) + DT(e_int_i0))

"""
    internal_energy(ts::ThermodynamicState)

The internal energy per unit mass, given a thermodynamic state `ts`.
"""
internal_energy(ts::PhaseEquil) = ts.e_int
internal_energy(ts::PhaseNonEquil) = ts.e_int

"""
    internal_energy_sat(T, ρ, q_tot)

The internal energy per unit mass in thermodynamic equilibrium at saturation where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
"""
function internal_energy_sat(T::DT, ρ::DT, q_tot::DT) where DT
    q_liq, q_ice = phase_partitioning_eq(T, ρ, q_tot)
    return internal_energy(T, q_tot, q_liq, q_ice)
end

"""
    internal_energy_sat(ts::ThermodynamicState)

The internal energy per unit mass in
thermodynamic equilibrium at saturation,
given a thermodynamic state `ts`.
"""
internal_energy_sat(ts::ThermodynamicState) =
  internal_energy_sat(air_temperature(ts), air_density(ts), ts.q_tot)


"""
    total_energy(e_kin, e_pot, T[, q_tot=0, q_liq=0, q_ice=0])

The total energy per unit mass, given

 - `e_kin` kinetic energy per unit mass
 - `e_pot` potential energy per unit mass
 - `T` temperature
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
total_energy(e_kin::DT, e_pot::DT, T::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT} =
  e_kin + e_pot + internal_energy(T, q_tot, q_liq, q_ice)

"""
    total_energy(e_kin, e_pot, ts::ThermodynamicState)

The total energy per unit mass
given a thermodynamic state `ts`.
"""
total_energy(e_kin, e_pot, ts::ThermodynamicState) = internal_energy(ts) + e_kin + e_pot

"""
    soundspeed_air(T[, q_tot=0, q_liq=0, q_ice=0])

The speed of sound in air, where
 - `T` temperature
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
function soundspeed_air(T::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real}
  γ   = cp_m(q_tot, q_liq, q_ice) / cv_m(q_tot, q_liq, q_ice)
  R_m = gas_constant_air(q_tot, q_liq, q_ice)
  return sqrt(γ*R_m*T)
end

"""
    soundspeed_air(ts::ThermodynamicState)

The speed of sound in air given a thermodynamic state `ts`.
"""
soundspeed_air(ts::ThermodynamicState) =
  soundspeed_air(air_temperature(ts), ts.q_tot, phase_partitioning_eq(ts)...)


"""
    latent_heat_vapor(T::Real)

The specific latent heat of vaporization where
 - `T` temperature
"""
latent_heat_vapor(T::DT) where {DT<:Real} =
  latent_heat_generic(T, DT(LH_v0), DT(cp_v) - DT(cp_l))

"""
    latent_heat_vapor(ts::ThermodynamicState)

The specific latent heat of vaporization
given a thermodynamic state `ts`.
"""
latent_heat_vapor(ts::ThermodynamicState{DT}) where {DT} =
    latent_heat_vapor(air_temperature(ts))

"""
    latent_heat_sublim(T::Real)

The specific latent heat of sublimation where
 - `T` temperature
"""
latent_heat_sublim(T::DT) where {DT<:Real} =
  latent_heat_generic(T, DT(LH_s0), DT(cp_v) - DT(cp_i))

"""
    latent_heat_sublim(ts::ThermodynamicState)

The specific latent heat of sublimation
given a thermodynamic state `ts`.
"""
latent_heat_sublim(ts::ThermodynamicState{DT}) where {DT} =
  latent_heat_sublim(air_temperature(ts))

"""
    latent_heat_fusion(T)

The specific latent heat of fusion where
 - `T` temperature
"""
latent_heat_fusion(T::DT) where {DT<:Real} =
  latent_heat_generic(T, DT(LH_f0), DT(cp_l) - DT(cp_i))

"""
    latent_heat_fusion(ts::ThermodynamicState)

The specific latent heat of fusion
given a thermodynamic state `ts`.
"""
latent_heat_fusion(ts::ThermodynamicState{DT}) where {DT} =
  latent_heat_fusion(air_temperature(ts))

"""
    latent_heat_generic(T::Real, LH_0::Real, Δcp::Real)

Return the specific latent heat of a generic phase transition between
two phases using Kirchhoff's relation.

The latent heat computation assumes constant isobaric specifc heat capacities
of the two phases. `T` is the temperature, `LH_0` is the latent heat of the
phase transition at `T_0`, and `Δcp` is the difference between the isobaric
specific heat capacities (heat capacity in the higher-temperature phase minus
that in the lower-temperature phase).
"""
latent_heat_generic(T::Real, LH_0::Real, Δcp::Real) =
  LH_0 + Δcp * (T - T_0)


"""
    Phase

A phase condensate, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`saturation_shum_generic`](@ref).
"""
abstract type Phase end

"""
    Liquid <: Phase

A liquid phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`saturation_shum_generic`](@ref).
"""
struct Liquid <: Phase end

"""
    Ice <: Phase

An ice phase, to dispatch over
[`saturation_vapor_pressure`](@ref) and
[`saturation_shum_generic`](@ref).
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
saturation_vapor_pressure(T::DT, ::Liquid) where DT = saturation_vapor_pressure(T, DT(LH_v0), DT(cp_v) - DT(cp_l))
function saturation_vapor_pressure(ts::ThermodynamicState, ::Liquid)

    return saturation_vapor_pressure(air_temperature(ts), LH_v0, cp_v - cp_l)

end
saturation_vapor_pressure(T::DT, ::Ice) where DT = saturation_vapor_pressure(T, DT(LH_s0), DT(cp_v) - DT(cp_i))
saturation_vapor_pressure(ts::ThermodynamicState, ::Ice) = saturation_vapor_pressure(air_temperature(ts), LH_s0, cp_v - cp_i)

function saturation_vapor_pressure(T::DT, LH_0::DT, Δcp::DT) where DT

    return DT(press_triple) * (T/DT(T_triple))^(Δcp/DT(R_v)) *
        exp( (DT(LH_0) - Δcp*DT(T_0))/DT(R_v) * (1 / DT(T_triple) - 1 / T) )

end

"""
    saturation_shum_generic(T, ρ[; phase=Liquid()])

Compute the saturation specific humidity over a plane surface of condensate, given

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `Liquid()` indicating condensate is liquid
 - `Ice()` indicating condensate is ice
"""
function saturation_shum_generic(T::DT, ρ::DT; phase::Phase=Liquid()) where DT

    p_v_sat = saturation_vapor_pressure(T, phase)

    return saturation_shum_from_pressure(T, ρ, p_v_sat)

end

"""
    saturation_shum(T, ρ[, q_liq=0, q_ice=0])

Compute the saturation specific humidity, given

 - `T` temperature
 - `ρ` (moist-)air density
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

If the optional liquid and ice specific humidities `q_tot` and `q_liq` are given,
the saturation specific humidity is that over a mixture of liquid and ice,
computed in a thermodynamically consistent way from the weighted sum of the
latent heats of the respective phase transitions (Pressel et al., JAMES, 2015).
That is, the saturation vapor pressure and from it the saturation
specific humidity are computed from a weighted mean of the latent heats of
vaporization and sublimation, with the weights given by the fractions of
condensate `q_liq/(q_liq + q_ice)` and `q_ice/(q_liq + q_ice)` that are liquid and
ice, respectively.

If the condensate specific humidities `q_liq` and `q_ice` are not given or are both
zero, the saturation specific humidity is that over a mixture of liquid and ice,
with the fraction of liquid given by temperature dependent `liquid_fraction_equil(T)`
and the fraction of ice by the complement `1 - liquid_fraction_equil(T)`.
"""
function saturation_shum(T::DT, ρ::DT, q_liq::DT=DT(0), q_ice::DT=DT(0)) where DT

    # get phase partitioning
    _liquid_frac = liquid_fraction_equil(T, q_liq, q_ice)
    _ice_frac    = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0    = _liquid_frac * DT(LH_v0) + _ice_frac * DT(LH_s0)
    Δcp     = _liquid_frac * (DT(cp_v) - DT(cp_l)) + _ice_frac * (DT(cp_v) - DT(cp_i))

    # saturation vapor pressure over possible mixture of liquid and ice
    p_v_sat = saturation_vapor_pressure(T, DT(LH_0), Δcp)

    return saturation_shum_from_pressure(T, ρ, p_v_sat)

end

"""
    saturation_shum(ts::ThermodynamicState)

Compute the saturation specific humidity, given a thermodynamic state `ts`.
"""
saturation_shum(ts::ThermodynamicState) =
  saturation_shum(air_temperature(ts), air_density(ts), phase_partitioning_eq(ts)...)

"""
    saturation_shum_from_pressure(T, ρ, p_v_sat)

Compute the saturation specific humidity, given

 - `T` ambient air temperature,
 - `ρ` density
 - `p_v_sat` saturation vapor pressure
"""
saturation_shum_from_pressure(T::DT, ρ::DT, p_v_sat::DT) where {DT<:Real} =
  min(1, p_v_sat / (ρ * DT(R_v) * T))

"""
    saturation_excess(T, ρ, q_tot, q_liq=0, q_ice=0)

The saturation excess in equilibrium where

 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity
and, optionally,
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

The saturation excess is the difference between the total specific humidity `q_tot`
and the saturation specific humidity in equilibrium, and it is defined to be
nonzero only if this difference is positive.
"""
saturation_excess(T::DT, ρ::DT, q_tot::DT, q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT} =
  max(0, q_tot - saturation_shum(T, ρ, q_liq, q_ice))

"""
    saturation_excess(ts::ThermodynamicState)

Compute the saturation excess in equilibrium,
given a thermodynamic state `ts`.
"""
saturation_excess(ts::ThermodynamicState) =
  saturation_excess(air_temperature(ts), air_density(ts), ts.q_tot, phase_partitioning_eq(ts)...)

"""
    liquid_fraction_equil(T[, q_liq=0, q_ice=0])

The fraction of condensate, assuming phase equilibrium, that is liquid where

 - `T` temperature
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

If the optional input arguments `q_liq` and `q_ice` are not given or are zero, the
fraction of liquid is a function that is 1 above `T_freeze` and goes to zero below
`T_freeze`. If `q_liq` or `q_ice` are nonzero, the liquid fraction is computed from
them.
"""
function liquid_fraction_equil(T::DT, q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real}
  q_c         = q_liq + q_ice     # condensate specific humidity

  # For now: Heaviside function for partitioning into liquid and ice: all liquid
  # for T > T_freeze; all ice for T <= T_freeze
  _liquid_frac = heaviside(T - T_freeze)

  return ifelse(q_c > 0, q_liq / q_c, _liquid_frac)
end

"""
    liquid_fraction_equil(ts::ThermodynamicState)

The fraction of condensate that is liquid given a thermodynamic state `ts`.
"""
liquid_fraction_equil(ts::ThermodynamicState) =
  liquid_fraction_equil(air_temperature(ts), phase_partitioning_eq(ts)...)

"""
    liquid_fraction_nonequil(T[, q_liq=0, q_ice=0])

The fraction of condensate, assuming phase non-equilibrium, that is liquid where

 - `T` temperature
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

If the optional input arguments `q_liq` and `q_ice` are not given or are zero, the
fraction of liquid is a function that is 1 above `T_freeze` and goes to zero below
`T_freeze`. If `q_liq` or `q_ice` are nonzero, the liquid fraction is computed from
them.

!!! todo
    Currently [`liquid_fraction_nonequil`](@ref) calls [`liquid_fraction_equil`](@ref),
    but we should implement a more general function here.
"""
function liquid_fraction_nonequil(T::DT, q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT<:Real}
  return liquid_fraction_equil(T, q_liq, q_ice)
end

"""
    liquid_fraction_nonequil(ts::ThermodynamicState)

The fraction of condensate that is liquid given a thermodynamic state `ts`.
"""
liquid_fraction_nonequil(ts::ThermodynamicState) =
  liquid_fraction_nonequil(air_temperature(ts), phase_partitioning_eq(ts)...)

"""
    heaviside(t)

Return the Heaviside step function at `t`.
"""
function heaviside(t::DT) where DT
   DT(1//2) * (sign(t) + 1)
end


"""
    (q_liq, q_ice) = phase_partitioning_eq(T, ρ, q_tot)

Partition the phases in equilibrium into
 - the liquid specific humidity `q_liq`, and
 - the ice specific humidity `q_ice`,

using the [`liquid_fraction_equil`](@ref) function where
 - `T` temperature
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity

The residual `q_tot - q_liq - q_ice` is the vapor specific humidity.
"""
function phase_partitioning_eq(T::DT, ρ::DT, q_tot::DT) where {DT}
    _liquid_frac = liquid_fraction_equil(T)   # fraction of condensate that is liquid
    q_c   = saturation_excess(T, ρ, q_tot)   # condensate specific humidity
    q_liq = _liquid_frac * q_c  # liquid specific humidity
    q_ice = (1 - _liquid_frac) * q_c # ice specific humidity

    return q_liq, q_ice
end

"""
    (q_liq, q_ice) = phase_partitioning_eq(ts::ThermodynamicState)

Partition the phases in equilibrium into
 - the liquid specific humidity `q_liq`, and
 - the ice specific humidity `q_ice`,
using the [`liquid_fraction_equil`](@ref) function given a thermodynamic state `ts`.
"""
phase_partitioning_eq(ts::ThermodynamicState) =
  phase_partitioning_eq(air_temperature(ts), air_density(ts), ts.q_tot)
phase_partitioning_eq(ts::PhaseNonEquil) = ts.q_liq, ts.q_ice

"""
    saturation_adjustment(e_int, ρ, q_tot)

Compute the temperature that is consistent with

 - `e_int` internal energy
 - `ρ` (moist-)air density
 - `q_tot` total specific humidity

See also [`saturation_adjustment_q_t_θ_l`](@ref).
"""
function saturation_adjustment(e_int::DT, ρ::DT, q_tot::DT) where DT
  T_1 = max(DT(T_min), air_temperature(e_int, q_tot)) # Assume all vapor
  q_v_sat = saturation_shum(T_1, ρ)
  if q_tot <= q_v_sat # If not saturated return T_1
    return T_1
  else # If saturated, iterate
    # FIXME here: need to revisit bounds for saturation adjustment to guarantee bracketing of zero.
    T_2 = air_temperature(e_int, q_tot, DT(0), q_tot) # Assume all ice
    eos_root(T, args) = internal_energy_sat(T, ρ, q_tot) - e_int
    T, converged = find_zero(eos_root, T_1, T_2, Tuple(1),
                             IterParams(DT(1e-3)*DT(cv_d), 10),
                             SecantMethod())
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
function saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice::DT, q_tot::DT, ρ::DT, p::DT) where DT
  T_1 = air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p) # Assume all vapor
  q_v_sat = saturation_shum(T_1, ρ)
  if q_tot <= q_v_sat # If not saturated
    return T_1
  else  # If saturated, iterate
    T_2 = air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p, q_tot, DT(0), q_tot) # Assume all ice
    eos_root(T, args) = θ_liq_ice - liquid_ice_pottemp_sat(T, p, q_tot, phase_partitioning_eq(T, ρ, q_tot)...)
    T, converged = find_zero(eos_root, T_1, T_2, Tuple(1),
                             IterParams(DT(1e-3), 10),
                             SecantMethod())
    return T
  end
end

"""
    liquid_ice_pottemp(T, p[, q_tot=0, q_liq=0, q_ice=0])

The liquid-ice potential temperature where
 - `T` temperature
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity
"""
function liquid_ice_pottemp(T::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where DT
    # isobaric specific heat of moist air
    _cp_m   = cp_m(q_tot, q_liq, q_ice)

    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp(T, p, q_tot, q_liq, q_ice) * (1 - (DT(LH_v0)*q_liq + DT(LH_s0)*q_ice)/(_cp_m*T))
end

"""
    liquid_ice_pottemp(ts::ThermodynamicState)

The liquid-ice potential temperature,
given a thermodynamic state `ts`.
"""
liquid_ice_pottemp(ts::ThermodynamicState) =
  liquid_ice_pottemp(air_temperature(ts), air_pressure(ts), ts.q_tot,
                     phase_partitioning_eq(ts)...)

"""
    dry_pottemp(T, p, q_tot, q_liq, q_ice)

The dry potential temperature where

 - `T` temperature
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity
 """
dry_pottemp(T::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT} =
  T / exner(p, q_tot, q_liq, q_ice)

"""
    dry_pottemp(ts::ThermodynamicState)

The dry potential temperature, given a thermodynamic state `ts`.
"""
dry_pottemp(ts::ThermodynamicState) =
  dry_pottemp(air_temperature(ts), air_pressure(ts), ts.q_tot,
              phase_partitioning_eq(ts)...)

"""
    air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p[, q_tot=0, q_liq=0, q_ice=0])

The air temperature, where

 - `θ_liq_ice` liquid-ice potential temperature
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity

Without the specific humidity arguments, the results
are that of dry air.
"""
function air_temperature_from_liquid_ice_pottemp(θ_liq_ice::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where DT
    _cp_m = cp_m(q_tot, q_liq, q_ice)
    return θ_liq_ice*exner(p, q_tot, q_liq, q_ice) + (DT(LH_v0)*q_liq + DT(LH_s0)*q_ice)/_cp_m
end

"""
    virtual_pottemp(T, p, q_tot, q_liq, q_ice)

The virtual temperature where

 - `T` temperature
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity
"""
virtual_pottemp(T::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT} =
  gas_constant_air(q_tot, q_liq, q_ice)/R_d*dry_pottemp(T, p, q_tot, q_liq, q_ice)

"""
    virtual_pottemp(ts::ThermodynamicState)

The virtual potential temperature,
given a thermodynamic state `ts`.
"""
virtual_pottemp(ts::ThermodynamicState) =
  virtual_pottemp(air_temperature(ts), ts.q_tot, phase_partitioning_eq(ts)...)


"""
    liquid_ice_pottemp_sat(T, p, q_tot, q_liq, q_ice)

The saturated liquid ice potential temperature where

 - `T` temperature
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity
"""
function liquid_ice_pottemp_sat(T::DT, p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where {DT}
    ρ = air_density(T, p, q_tot, q_liq, q_ice)
    q_v_sat = saturation_shum(T, ρ, q_liq, q_ice)
    return liquid_ice_pottemp(T, p, q_v_sat)
end

"""
    liquid_ice_pottemp_sat(ts::ThermodynamicState)

The liquid potential temperature given a thermodynamic state `ts`.
"""
liquid_ice_pottemp_sat(ts::ThermodynamicState) =
  liquid_ice_pottemp_sat(air_temperature(ts), air_pressure(ts), ts.q_tot,
                 phase_partitioning_eq(ts)...)

"""
    exner(p, q_tot, q_liq, q_ice)

Compute the Exner function where
 - `p` pressure
and, optionally,
 - `q_tot` total specific humidity
 - `q_liq` liquid specific humidity
 - `q_ice` ice specific humidity
"""
function exner(p::DT, q_tot::DT=DT(0), q_liq::DT=DT(0), q_ice::DT=DT(0)) where DT
    # gas constant and isobaric specific heat of moist air
    _R_m    = gas_constant_air(q_tot, q_liq, q_ice)
    _cp_m   = cp_m(q_tot, q_liq, q_ice)

    return (p/DT(MSLP))^(_R_m/_cp_m)
end

function virtual_pottemp(T, p, q_tot=0, q_liq=0, q_ice=0)
  q_vap = q_tot - q_liq - q_ice
  theta = dry_pottemp(T, p, q_tot, q_liq, q_ice)
  return theta * (1.0 + 0.61 * q_vap - q_liq - q_ice)
end
"""
    exner(ts::ThermodynamicState)

Compute the Exner function, given a thermodynamic state `ts`.
"""
exner(ts::ThermodynamicState) =
  exner(air_pressure(ts), ts.q_tot, phase_partitioning_eq(ts)...)


end #module MoistThermodynamics.jl
