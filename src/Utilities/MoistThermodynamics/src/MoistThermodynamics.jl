"""
    MoistThermodynamics

Moist thermodynamic functions, e.g., for air pressure (atmosphere equation
of state), latent heats of phase transitions, saturation vapor pressures, and
saturation specific humidities
"""
module MoistThermodynamics

using ..RootSolvers
using ...PlanetParameters

# Atmospheric equation of state
export air_pressure, air_temperature, air_density

# Energies
export total_energy, internal_energy, internal_energy_sat

# Specific heats of moist air
export cp_m, cv_m, gas_constant_air

# Latent heats
export latent_heat_vapor, latent_heat_sublim, latent_heat_fusion

# Speed of sound in air
export soundspeed_air

# Saturation vapor pressures and specific humidities over liquid and ice
export Liquid, Ice
export saturation_vapor_pressure, saturation_shum_generic, saturation_shum
export saturation_excess

# Functions used in thermodynamic equilibrium among phases (liquid and ice
# determined diagnostically from total water specific humidity)
export liquid_fraction, phase_partitioning_eq!, saturation_adjustment, phase_partitioning_eq

# Auxiliary functions, e.g., for diagnostic purposes
export liquid_ice_pottemp, dry_pottemp, exner

# Thermodynamic states
export ThermodynamicState, InternalEnergy_Shum_eq, InternalEnergy_Shum_neq

abstract type ThermodynamicState{DT} end

"""
    InternalEnergy_Shum_eq{DT} <: ThermodynamicState

A thermodynamic state initialized by internal energy
`e_int`, specific humidity `q_t`,  and density `ρ`,
assuming thermodynamic equilibrium (therefore, saturation
adjustment is needed).
"""
struct InternalEnergy_Shum_eq{DT} <: ThermodynamicState{DT}
  e_int::DT
  q_t::DT
  ρ::DT
  T::DT
  function InternalEnergy_Shum_eq(e_int, q_t, ρ)
    DT = typeof(ρ)
    return new{DT}(e_int, q_t, ρ, saturation_adjustment(e_int, ρ, q_t))
  end
end

"""
    InternalEnergy_Shum_neq{DT} <: ThermodynamicState

A thermodynamic state initialized by internal energy
`e_int`, specific humidity `q_t`,  and density `ρ`.
assuming thermodynamic non-equilibrium (therefore,
temperature can be computed directly).
"""
struct InternalEnergy_Shum_neq{DT} <: ThermodynamicState{DT}
  e_int::DT
  q_t::DT
  q_l::DT
  q_i::DT
  ρ::DT
  T::DT
  function InternalEnergy_Shum_neq(e_int, q_t, q_l, q_i, ρ)
    DT = typeof(ρ)
    return new{DT}(e_int, q_t, q_l, q_i, ρ, air_temperature(e_int, q_t, q_l, q_i))
  end
end

"""
    gas_constant_air([q_t=0, q_l=0, q_i=0])

Return the specific gas constant of moist air, given the total specific
humidity `q_t`, and, optionally, the liquid specific humidity `q_l`,
and the ice specific humidity `q_i`. When no input argument is given, it
returns the specific gas constant of dry air.
"""
function gas_constant_air(q_t=0, q_l=0, q_i=0)

    return R_d * ( 1 +  (molmass_ratio - 1)*q_t - molmass_ratio*(q_l + q_i) )

end

"""
    gas_constant_air(ts::ThermodynamicState)

Return the specific gas constant of moist air given the thermodynamic state `ts`.
"""
function gas_constant_air(ts::ThermodynamicState)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return R_d * ( 1 +  (molmass_ratio - 1)*q_t - molmass_ratio*(q_l + q_i) )

end


"""
    air_pressure(T, ρ[, q_t=0, q_l=0, q_i=0])

Return the air pressure from the equation of state (ideal gas law), given
the air temperature `T`, the (moist-)air density `ρ`, and, optionally, the total
specific humidity `q_t`, the liquid specific humidity `q_l`, and the ice
specific humidity `q_i`. Without the specific humidity arguments, it returns
the air pressure from the equation of state of dry air.
"""
function air_pressure(T, ρ, q_t=0, q_l=0, q_i=0)

    return gas_constant_air(q_t, q_l, q_i) * ρ * T

end

"""
    air_pressure(ts::ThermodynamicState)

Return the air pressure from the equation of state (ideal gas law),
given the thermodynamic state `ts`.
"""
function air_pressure(ts::ThermodynamicState)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return gas_constant_air(ts.q_t, q_l, q_i) * ts.ρ * ts.T

end

"""
    air_density(T, p[, q_t=0, q_l=0, q_i=0])

Return the (moist-)air density from the equation of state (ideal gas law), given
the air temperature `T`, the pressure `p`, and, optionally, the total specific
humidity `q_t`, the liquid specific humidity `q_l`, and the ice specific
humidity `q_i`. Without the specific humidity arguments, it returns the
(moist-)air density from the equation of state of dry air.
"""
function air_density(T, p, q_t=0, q_l=0, q_i=0)

    return p / (gas_constant_air(q_t, q_l, q_i) * T)

end

"""
    air_density(ts::ThermodynamicState)

Return the (moist-)air density from the equation of state (ideal gas law),
given the thermodynamic state `ts`.
"""
function air_density(ts::ThermodynamicState)
    p = air_pressure(ts)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return p / (gas_constant_air(ts.q_t, q_l, q_i) * ts.T)

end

"""
    cp_m([q_t=0, q_l=0, q_i=0])

Return the isobaric specific heat capacity of moist air, given the
total water specific humidity `q_t`, liquid specific humidity `q_l`, and
ice specific humidity `q_i`. Without the specific humidity arguments, it returns
the isobaric specific heat capacity of dry air.
"""
function cp_m(q_t=0, q_l=0, q_i=0)

    return cp_d + (cp_v - cp_d)*q_t + (cp_l - cp_v)*q_l + (cp_i - cp_v)*q_i

end

"""
    cp_m(ts::ThermodynamicState)

Return the isobaric specific heat capacity of moist air,
given the thermodynamic state `ts`.
"""
function cp_m(ts::ThermodynamicState)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return cp_d + (cp_v - cp_d)*ts.q_t + (cp_l - cp_v)*q_l + (cp_i - cp_v)*q_i

end

"""
    cv_m([q_t=0, q_l=0, q_i=0])

Return the isochoric specific heat capacity of moist air, given the
total water specific humidity `q_t`, liquid specific humidity `q_l`, and
ice specific humidity `q_i`. Without the specific humidity arguments, it returns
the isochoric specific heat capacity of dry air.
"""
function cv_m(q_t=0, q_l=0, q_i=0)

    return cv_d + (cv_v - cv_d)*q_t + (cv_l - cv_v)*q_l + (cv_i - cv_v)*q_i

end

"""
    cv_m(ts::ThermodynamicState)

Return the isochoric specific heat capacity of moist air, given the thermodynamic state `ts`.
"""
function cv_m(ts::ThermodynamicState)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return cv_d + (cv_v - cv_d)*ts.q_t + (cv_l - cv_v)*q_l + (cv_i - cv_v)*q_i

end

"""
    air_temperature(e_int[, q_t=0, q_l=0, q_i=0])

Return the air temperature, given the internal energy `e_int` per unit mass,
and, optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function air_temperature(e_int, q_t=0, q_l=0, q_i=0)

    return T_0 +
        ( e_int - (q_t - q_l) * e_int_v0 + q_i * (e_int_v0 + e_int_i0) )/
            cv_m(q_t, q_l, q_i)

end

"""
    air_temperature(ts::InternalEnergy_Shum_eq)

Return the air temperature, given the thermodynamic state `ts`.
"""
function air_temperature(ts::InternalEnergy_Shum_eq)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return T_0 +
        ( ts.e_int - (ts.q_t - q_l) * e_int_v0 + q_i * (e_int_v0 + e_int_i0) )/
            cv_m(ts.q_t, q_l, q_i)

end

"""
    air_temperature(ts::InternalEnergy_Shum_neq)

Return the air temperature, given the thermodynamic state `ts`.
"""
function air_temperature(ts::InternalEnergy_Shum_neq)
    return T_0 +
        ( ts.e_int - (ts.q_t - ts.q_l) * e_int_v0 + ts.q_i * (e_int_v0 + e_int_i0) )/
            cv_m(ts.q_t, ts.q_l, ts.q_i)

end

"""
    internal_energy(T[, q_t=0, q_l=0, q_i=0])

Return the internal energy per unit mass, given the temperature `T`, and,
optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function internal_energy(T, q_t=0, q_l=0, q_i=0)

    return cv_m(q_t, q_l, q_i) * (T - T_0) +
        (q_t - q_l) * e_int_v0 - q_i * (e_int_v0 + e_int_i0)

end

"""
    internal_energy(ts::ThermodynamicState)

Return the internal energy per unit mass, given the thermodynamic state `ts`.
"""
function internal_energy(ts::ThermodynamicState)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)

    return cv_m(ts.q_t, q_l, q_i) * (ts.T - T_0) +
        (ts.q_t - q_l) * e_int_v0 - q_i * (e_int_v0 + e_int_i0)

end

"""
    internal_energy_sat(T, ρ, q_t)

Return the internal energy per unit mass in thermodynamic equilibrium at
saturation, given the temperature `T`, (moist-)air density `ρ`, and total
specific humidity `q_t`.
"""
function internal_energy_sat(T, ρ, q_t)

    # get equilibrium phase partitioning
    _q_l, _q_i = phase_partitioning_eq(T, ρ, q_t)

    return internal_energy(T, q_t, _q_l, _q_i)

end

"""
    internal_energy_sat(ts::ThermodynamicState)

Return the internal energy per unit mass in thermodynamic equilibrium at
saturation, given the thermodynamic state `ts`.
"""
internal_energy_sat(ts::ThermodynamicState) = internal_energy(ts)

"""
    total_energy(e_kin, e_pot, T[, q_t=0, q_l=0, q_i=0])

Return the total energy per unit mass, given the kinetic energy per unit
mass `e_kin`, the potential energy per unit mass `e_pot`, the temperature `T`, and,
optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function total_energy(e_kin, e_pot, T, q_t=0, q_l=0, q_i=0)

    return e_kin + e_pot + internal_energy(T, q_t, q_l, q_i)

end

"""
    total_energy(ts::ThermodynamicState, e_kin, e_pot)

Return the total energy per unit mass, given the thermodynamic state `ts`.
"""
function total_energy(ts::ThermodynamicState, e_kin, e_pot)

    return internal_energy(ts) + e_kin + e_pot

end

"""
    soundspeed_air(T[, q_t=0, q_l=0, q_i=0])

Return the speed of sound in air, given the temperature `T`, and,
optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function soundspeed_air(T, q_t=0, q_l=0, q_i=0)

    _γ   = cp_m(q_t, q_l, q_i)/cv_m(q_t, q_l, q_i)
    _R_m = gas_constant_air(q_t, q_l, q_i)
    return sqrt(_γ * _R_m * T)

end

"""
    soundspeed_air(ts::ThermodynamicState)

Return the speed of sound in air, given the thermodynamic state `ts`.
"""
function soundspeed_air(ts::ThermodynamicState)

    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
    _γ   = cp_m(q_t, q_l, q_i)/cv_m(q_t, q_l, q_i)
    _R_m = gas_constant_air(q_t, q_l, q_i)
    return sqrt(_γ * _R_m * T)

end

"""
    latent_heat_vapor(T)

Return the specific latent heat of vaporization at temperature `T`.
"""
function latent_heat_vapor(T)

     return latent_heat_generic(T, LH_v0, cp_v - cp_l)

end

"""
    latent_heat_vapor(ts::ThermodynamicState)

Return the specific latent heat of vaporization given the thermodynamic state `ts`.
"""
function latent_heat_vapor(ts::ThermodynamicState)

     return latent_heat_generic(ts.T, LH_v0, cp_v - cp_l)

end

"""
    latent_heat_sublim(T)

Return the specific latent heat of sublimation at temperature `T`.
"""
function latent_heat_sublim(T)

    return latent_heat_generic(T, LH_s0, cp_v - cp_i)

end

"""
    latent_heat_sublim(ts::ThermodynamicState)

Return the specific latent heat of sublimation given the thermodynamic state `ts`.
"""
function latent_heat_sublim(ts::ThermodynamicState)

    return latent_heat_generic(ts.T, LH_s0, cp_v - cp_i)

end

"""
    latent_heat_fusion(T)

Return the specific latent heat of fusion at temperature `T`.
"""
function latent_heat_fusion(T)

    return latent_heat_generic(T, LH_f0, cp_l - cp_i)

end

"""
    latent_heat_fusion(ts::ThermodynamicState)

Return the specific latent heat of fusion given the thermodynamic state `ts`.
"""
function latent_heat_fusion(ts::ThermodynamicState)

    return latent_heat_generic(ts.T, LH_f0, cp_l - cp_i)

end

"""
    latent_heat_generic(T, LH_0, cp_diff)

Return the specific latent heat of a generic phase transition between
two phases using Kirchhoff's relation.

The latent heat computation assumes constant isobaric specifc heat capacities
of the two phases. `T` is the temperature, `LH_0` is the latent heat of the
phase transition at `T_0`, and `cp_diff` is the difference between the isobaric
specific heat capacities (heat capacity in the higher-temperature phase minus
that in the lower-temperature phase).
"""
function latent_heat_generic(T, LH_0, cp_diff)

    return LH_0 + cp_diff * (T - T_0)

end

abstract type Phase end
struct Liquid <: Phase end
struct Ice <: Phase end

"""
    `saturation_vapor_pressure(T, Liquid())`

Return the saturation vapor pressure over a plane liquid surface at
temperature `T`.

    `saturation_vapor_pressure(T, Ice())`

Return the saturation vapor pressure over a plane ice surface at
temperature `T`.

    `saturation_vapor_pressure(T, LH_0, cp_diff)`

Compute the saturation vapor pressure over a plane surface by integration
of the Clausius-Clepeyron relation.

The Clausius-Clapeyron relation

    dlog(p_vs)/dT = [LH_0 + cp_diff * (T-T_0)]/(R_v*T^2)

is integrated from the triple point temperature `T_triple`, using
Kirchhoff's relation

    L = LH_0 + cp_diff * (T - T_0)

for the specific latent heat `L` with constant isobaric specific
heats of the phases. The linear dependence of the specific latent heat
on temperature `T` allows analytic integration of the Clausius-Clapeyron
relation to obtain the saturation vapor pressure `p_vs` as a function of
the triple point pressure `press_triple`.

"""
saturation_vapor_pressure(ts::ThermodynamicState, ::Liquid) = saturation_vapor_pressure(ts.T, LH_v0, cp_v - cp_l)
saturation_vapor_pressure(T, ::Liquid) = saturation_vapor_pressure(T, LH_v0, cp_v - cp_l)
saturation_vapor_pressure(ts::ThermodynamicState, ::Ice) = saturation_vapor_pressure(ts.T, LH_s0, cp_v - cp_i)
saturation_vapor_pressure(T, ::Ice) = saturation_vapor_pressure(T, LH_s0, cp_v - cp_i)

function saturation_vapor_pressure(T, LH_0, cp_diff)

    return press_triple * (T/T_triple)^(cp_diff/R_v) *
        exp( (LH_0 - cp_diff*T_0)/R_v * (1 / T_triple - 1 / T) )

end

"""
    saturation_shum_generic(T, ρ[; phase=Liquid()])

Compute the saturation specific humidity over a plane surface of
condensate, given the temperature `T` and the (moist-)air density `ρ`.

The optional argument `phase` can be ``Liquid()`` or ``"ice"`` and indicates
the condensed phase.
"""
function saturation_shum_generic(T, ρ; phase::Phase=Liquid())

    p_vs = saturation_vapor_pressure(T, phase)

    return saturation_shum_from_pressure(T, ρ, p_vs)

end

"""
    saturation_shum(T, ρ[, q_l=0, q_i=0])

Compute the saturation specific humidity, given the temperature `T` and
(moist-)air density `ρ`.

If the optional liquid, and ice specific humdities `q_t` and `q_l` are given,
the saturation specific humidity is that over a mixture of liquid and ice,
computed in a thermodynamically consistent way from the weighted sum of the
latent heats of the respective phase transitions (Pressel et al., JAMES, 2015).
That is, the saturation vapor pressure and from it the saturation
specific humidity are computed from a weighted mean of the latent heats of
vaporization and sublimation, with the weights given by the fractions of
condensate `q_l`/(`q_l` + `q_i`) and `q_i`/(`q_l` + `q_i`) that are liquid and
ice, respectively.

If the condensate specific humidities `q_l` and `q_i` are not given or are both
zero, the saturation specific humidity is that over a mixture of liquid and ice,
with the fraction of liquid given by temperature dependent `liquid_fraction(T)`
and the fraction of ice by the complement `1 - liquid_fraction(T)`.
"""
function saturation_shum(T, ρ, q_l=0, q_i=0)

    # get phase partitioning
    _liquid_frac = liquid_fraction(T, q_l, q_i)
    _ice_frac    = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0        = _liquid_frac * LH_v0 + _ice_frac * LH_s0
    cp_diff     = _liquid_frac * (cp_v - cp_l) + _ice_frac * (cp_v - cp_i)

    # saturation vapor pressure over possible mixture of liquid and ice
    p_vs        = saturation_vapor_pressure(T, LH_0, cp_diff)

    return saturation_shum_from_pressure(T, ρ, p_vs)

end

"""
    saturation_shum(ts::ThermodynamicState)

Compute the saturation specific humidity, given the thermodynamic state `ts`.
"""
function saturation_shum(ts::ThermodynamicState)

    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
    # get phase partitioning
    _liquid_frac = liquid_fraction(ts.T, q_l, q_i)
    _ice_frac    = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0        = _liquid_frac * LH_v0 + _ice_frac * LH_s0
    cp_diff     = _liquid_frac * (cp_v - cp_l) + _ice_frac * (cp_v - cp_i)

    # saturation vapor pressure over possible mixture of liquid and ice
    p_vs        = saturation_vapor_pressure(ts.T, LH_0, cp_diff)

    return saturation_shum_from_pressure(ts.T, ts.ρ, p_vs)

end

"""
    saturation_shum(ts::InternalEnergy_Shum_neq)

Compute the saturation specific humidity, given the thermodynamic state `ts`.
"""
function saturation_shum(ts::InternalEnergy_Shum_neq)

    # get phase partitioning
    _liquid_frac = liquid_fraction(ts.T, ts.q_l, ts.q_i)
    _ice_frac    = 1 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0        = _liquid_frac * LH_v0 + _ice_frac * LH_s0
    cp_diff     = _liquid_frac * (cp_v - cp_l) + _ice_frac * (cp_v - cp_i)

    # saturation vapor pressure over possible mixture of liquid and ice
    p_vs        = saturation_vapor_pressure(ts.T, LH_0, cp_diff)

    return saturation_shum_from_pressure(ts.T, ts.ρ, p_vs)

end


"""
    saturation_shum_from_pressure(T, ρ, p_vs)

Compute the saturation specific humidity, given the ambient air temperature `T`,
density `ρ`, and the saturation vapor pressure `p_vs`.
"""
function saturation_shum_from_pressure(T, ρ, p_vs)

    return min(typeof(ρ)(1), p_vs / (ρ * R_v * T))

end

"""
    saturation_excess(T, ρ, q_t, q_l=0, q_i=0)

Compute the saturation excess in equilibrium, given the ambient air temperature
`T`, the (moist-)air density `ρ`, the total specific humidity `q_t`, and,
optionally, the liquid specific humidity `q_l`, and the ice specific humidity `q_i`.

The saturation excess is the difference between the total specific humidity `q_t`
and the saturation specific humidity in equilibrium, and it is defined to be
nonzero only if this difference is positive.
"""
function saturation_excess(T, ρ, q_t, q_l=0, q_i=0)

    return max(typeof(q_t)(0), q_t - saturation_shum(T, ρ, q_l, q_i))

end

"""
    saturation_excess(ts::ThermodynamicState)

Compute the saturation excess in equilibrium, given the thermodynamic state `ts`.
"""
function saturation_excess(ts::ThermodynamicState)

    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
    return max(typeof(ts.q_t)(0), ts.q_t - saturation_shum(ts.T, ts.ρ, q_l, q_i))

end

"""
    liquid_fraction(T[, q_l=0, q_i=0])

Return the fraction of condensate that is liquid.

If the optional input arguments `q_l` and `q_i` are not given or are zero, the
fraction of liquid is a function that is 1 above `T_freeze` and goes to zero below
`T_freeze`. If `q_l` or `q_i` are nonzero, the liquid fraction is computed from
them.
"""
function liquid_fraction(T, q_l=0, q_i=0)

  q_c         = q_l + q_i     # condensate specific humidity

  # For now: Heaviside function for partitioning into liquid and ice: all liquid
  # for T > T_freeze; all ice for T <= T_freeze
  _liquid_frac = heaviside(T - T_freeze)

  return ifelse(q_c > 0, q_l / q_c, _liquid_frac)

end

"""
    liquid_fraction(ts::ThermodynamicState)

Return the fraction of condensate that is liquid.

If the optional input arguments `q_l` and `q_i` are not given or are zero, the
fraction of liquid is a function that is 1 above `T_freeze` and goes to zero below
`T_freeze`. If `q_l` or `q_i` are nonzero, the liquid fraction is computed from
them.
"""
function liquid_fraction(ts::ThermodynamicState)

  q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
  q_c         = q_l + q_i     # condensate specific humidity

  # For now: Heaviside function for partitioning into liquid and ice: all liquid
  # for T > T_freeze; all ice for T <= T_freeze
  _liquid_frac = heaviside(ts.T - T_freeze)

  return ifelse(q_c > 0, q_l / q_c, _liquid_frac)

end

"""
    heaviside(t)

Return the Heaviside step function at `t`.
"""
function heaviside(t)
   typeof(t)(1//2) * (sign(t) + 1)
end

"""
    phase_partitioning_eq(q_l, q_i, T, ρ, q_t)

Return the partitioning of the phases in equilibrium.

Given the temperature `T` and (moist-)air density `ρ`, `phase_partitioning_eq`
partitions the total specific humidity `q_t` into the liquid specific humidity
`q_l` and ice specific humiditiy `q_l` using the `liquid_fraction`
function. The residual `q_t - q_l - q_i` is the vapor specific humidity.
"""
function phase_partitioning_eq(T, ρ, q_t)

    _liquid_frac = liquid_fraction(T)   # fraction of condensate that is liquid
    q_c          = saturation_excess(T, ρ, q_t)   # condensate specific humidity
    q_l_out      = _liquid_frac * q_c  # liquid specific humidity
    q_i_out      = (1 - _liquid_frac) * q_c # ice specific humidity

    return q_l_out, q_i_out

end

"""
    ts::ThermodynamicState

Return the partitioning of the phases in equilibrium given the thermodynamic state `ts`.
"""
function phase_partitioning_eq(ts::ThermodynamicState)

    _liquid_frac = liquid_fraction(ts.T)   # fraction of condensate that is liquid
    q_c          = saturation_excess(ts.T, ts.ρ, ts.q_t)   # condensate specific humidity
    q_l_out      = _liquid_frac * q_c  # liquid specific humidity
    q_i_out      = (1 - _liquid_frac) * q_c # ice specific humidity

    return q_l_out, q_i_out

end

"""
    saturation_adjustment(e_int, ρ, q_t[, T_init = T_triple])
Return the temperature that is consistent with the internal energy `e_int`,
(moist-)air density `ρ`, and total specific humidity `q_t`.
The optional input value of the temperature `T_init` is taken as the initial
value of the saturation adjustment iterations.
"""
function saturation_adjustment(e_int, ρ, q_t, T_init = T_triple)
    tol_abs = 1e-3*cv_d
    iter_max = 10
    args = (ρ, q_t, e_int)
    T0 = max(T_min, air_temperature(e_int, q_t, typeof(q_t)(0), typeof(q_t)(0)))
    T1 = air_temperature(e_int, q_t, typeof(q_t)(0), q_t)
    roots_equation(x, ρ, q_t, e_int) = internal_energy_sat(x, ρ, q_t) - e_int
    T, converged = find_zero(roots_equation,
                             T0, T1,
                             args,
                             IterParams(tol_abs, iter_max),
                             SecantMethod()
                             )
    return T

end

"""
    liquid_ice_pottemp(T, p[, q_t=0, q_l=0, q_i=0])

Return the liquid-ice potential temperature, given the temperature `T`,
pressure `p`, total specific humidity `q_t`, liquid specific humidity `q_l`,
and ice specific humidity `q_i`.
"""
function liquid_ice_pottemp(T, p, q_t=0, q_l=0, q_i=0)

    # isobaric specific heat of moist air
    _cp_m   = cp_m(q_t, q_l, q_i)

    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp(T, p, q_t, q_l, q_i) * exp(-(LH_v0*q_l + LH_s0*q_i)/(_cp_m*T))

end

"""
    liquid_ice_pottemp(ts::ThermodynamicState)

Return the liquid-ice potential temperature, given the thermodynamic state `ts`.
"""
function liquid_ice_pottemp(ts::ThermodynamicState)

    # isobaric specific heat of moist air
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
    _cp_m   = cp_m(ts.q_t, q_l, q_i)
    p = air_pressure(ts.T, ts.ρ, ts.q_t, q_l, q_i)

    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp(ts.T, p, ts.q_t, q_l, q_i) * exp(-(LH_v0*q_l + LH_s0*q_i)/(_cp_m*ts.T))

end

"""
    dry_pottemp(T, p, q_t=0, q_l=0, q_i=0)

Return the dry potential temperature, given the temperature `T`,
pressure `p`, total specific humidity `q_t`, liquid specific humidity `q_l`,
and ice specific humidity `q_i`.
"""
dry_pottemp(T, p, q_t=0, q_l=0, q_i=0) = T / exner(p, q_t, q_l, q_i)

"""
    dry_pottemp(T, p, q_t=0, q_l=0, q_i=0)

Return the dry potential temperature, given the thermodynamic state `ts`.
"""
function dry_pottemp(ts::ThermodynamicState)
    p = air_pressure(ts.T, ts.ρ, ts.q_t, q_l, q_i)
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
    return T / exner(p, ts.q_t, q_l, q_i)
end

"""
    exner(p, q_t=0, q_l=0, q_i=0)

Return the Exner function, given the pressure `p`, total specific
humidity `q_t`, liquid specific humidity `q_l`, and ice specific humidity `q_i`.
"""
function exner(p, q_t=0, q_l=0, q_i=0)

    # gas constant and isobaric specific heat of moist air
    _R_m    = gas_constant_air(q_t, q_l, q_i)
    _cp_m   = cp_m(q_t, q_l, q_i)

    return (p/MSLP)^(_R_m/_cp_m)

end

"""
    exner(ts::ThermodynamicState)

Return the Exner function, given the pressure `p`, total specific
humidity `q_t`, liquid specific humidity `q_l`, and ice specific humidity `q_i`.
"""
function exner(ts::ThermodynamicState)

    # gas constant and isobaric specific heat of moist air
    q_l, q_i = phase_partitioning_eq(ts.T, ts.ρ, ts.q_t)
    _R_m    = gas_constant_air(ts.q_t, q_l, q_i)
    _cp_m   = cp_m(ts.q_t, q_l, q_i)
    p = air_pressure(ts.T, ts.ρ, ts.q_t, q_l, q_i)

    return (p/MSLP)^(_R_m/_cp_m)

end


end #module MoistThermodynamics.jl
