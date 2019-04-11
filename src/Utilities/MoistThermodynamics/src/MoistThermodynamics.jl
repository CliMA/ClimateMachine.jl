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
export air_pressure, air_temperature, air_density, soundspeed_air

# Energies
export total_energy, internal_energy, internal_energy_sat, kinetic_energy

# Specific heats of moist air
export cp_m, cv_m, gas_constant_air, moist_gas_constants

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
export liquid_fraction, saturation_adjustment, phase_partitioning_eq

# Auxiliary functions, e.g., for diagnostic purposes
export liquid_ice_pottemp, dry_pottemp, exner


"""
<<<<<<< HEAD
    soundspeed_air(T[, q_t=0, q_l=0, q_i=0])
Return the speed of sound in air, given the temperature `T`, and,
optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function soundspeed_air(T, q_t=0, q_l=0, q_i=0)

    _γ   = cp_m(q_t, q_l, q_i)/cv_m(q_t, q_l, q_i)
    _R_m = gas_constant_air(q_t, q_l, q_i)
=======
    soundspeed_air(T[, q_tot=0, q_liq=0, q_ice=0])
Return the speed of sound in air, given the temperature `T`, and,
optionally, the total specific humidity `q_tot`, the liquid specific humidity
`q_liq`, and the ice specific humidity `q_ice`.
"""
function soundspeed_air(T, q_tot=0, q_liq=0, q_ice=0)

    _γ   = cp_m(q_tot, q_liq, q_ice)/cv_m(q_tot, q_liq, q_ice)
    _R_m = gas_constant_air(q_tot, q_liq, q_ice)
>>>>>>> master
    return sqrt(_γ * _R_m * T)

end

"""
    gas_constant_air([q_tot=0, q_liq=0, q_ice=0])

Return the specific gas constant of moist air, given the total specific
humidity `q_tot`, and, optionally, the liquid specific humidity `q_liq`,
and the ice specific humidity `q_ice`. When no input argument is given, it
returns the specific gas constant of dry air.
"""
function gas_constant_air(q_tot=0, q_liq=0, q_ice=0)

    return R_d * ( 1 +  (molmass_ratio - 1)*q_tot - molmass_ratio*(q_liq + q_ice) )

end

"""
    air_pressure(T, ρ[, q_tot=0, q_liq=0, q_ice=0])

Return the air pressure from the equation of state (ideal gas law), given
the air temperature `T`, the (moist-)air density `ρ`, and, optionally, the total
specific humidity `q_tot`, the liquid specific humidity `q_liq`, and the ice
specific humidity `q_ice`. Without the specific humidity arguments, it returns
the air pressure from the equation of state of dry air.
"""
function air_pressure(T, ρ, q_tot=0, q_liq=0, q_ice=0)

    return gas_constant_air(q_tot, q_liq, q_ice) * ρ * T

end

"""
    air_density(T, p[, q_tot=0, q_liq=0, q_ice=0])

Return the (moist-)air density from the equation of state (ideal gas law), given
the air temperature `T`, the pressure `p`, and, optionally, the total specific
humidity `q_tot`, the liquid specific humidity `q_liq`, and the ice specific
humidity `q_ice`. Without the specific humidity arguments, it returns the air
density from the equation of state of dry air.
"""
function air_density(T, p, q_tot=0, q_liq=0, q_ice=0)

    return p / (gas_constant_air(q_tot, q_liq, q_ice) * T)

end

"""
    cp_m([q_tot=0, q_liq=0, q_ice=0])

Return the isobaric specific heat capacity of moist air, given the
total water specific humidity `q_tot`, liquid specific humidity `q_liq`, and
ice specific humidity `q_ice`. Without the specific humidity arguments, it returns
the isobaric specific heat capacity of dry air.
"""
function cp_m(q_tot=0, q_liq=0, q_ice=0)

    return cp_d + (cp_v - cp_d)*q_tot + (cp_l - cp_v)*q_liq + (cp_i - cp_v)*q_ice

end

"""
    cv_m([q_tot=0, q_liq=0, q_ice=0])

Return the isochoric specific heat capacity of moist air, given the
total water specific humidity `q_tot`, liquid specific humidity `q_liq`, and
ice specific humidity `q_ice`. Without the specific humidity arguments, it returns
the isochoric specific heat capacity of dry air.
"""
function cv_m(q_tot=0, q_liq=0, q_ice=0)

    return cv_d + (cv_v - cv_d)*q_tot + (cv_l - cv_v)*q_liq + (cv_i - cv_v)*q_ice

end


"""
    moist_gas_constants([q_tot=0, q_liq=0, q_ice=0])

Wrapper to return R_m, cv_m, cp_m, and gamma_m all at once
"""
function moist_gas_constants(q_tot=0, q_liq=0, q_ice=0)

    R_gas  = gas_constant_air(q_tot, q_liq, q_ice)
    cp = cp_m(q_tot, q_liq, q_ice)
    cv = cv_m(q_tot, q_liq, q_ice)
    gamma = cp/cv

    return (R_gas, cp, cv, gamma)
end

"""
    air_temperature(e_int[, q_tot=0, q_liq=0, q_ice=0])

Return the air temperature, given the internal energy `e_int` per unit mass,
and, optionally, the total specific humidity `q_tot`, the liquid specific humidity
`q_liq`, and the ice specific humidity `q_ice`.
"""
function air_temperature(internal_energy, q_tot=0, q_liq=0, q_ice=0)

    return T_0 +
        ( internal_energy - (q_tot - q_liq) * e_int_v0 + q_ice * (e_int_v0 + e_int_i0) )/
            cv_m(q_tot, q_liq, q_ice)

end

"""
    internal_energy(T[, q_tot=0, q_liq=0, q_ice=0])

Return the internal energy per unit mass, given the temperature `T`, and,
optionally, the total specific humidity `q_tot`, the liquid specific humidity
`q_liq`, and the ice specific humidity `q_ice`.
"""
function internal_energy(T, q_tot=0, q_liq=0, q_ice=0)

    return cv_m(q_tot, q_liq, q_ice) * (T - T_0) +
        (q_tot - q_liq) * e_int_v0 - q_ice * (e_int_v0 + e_int_i0)

end

"""
    internal_energy_sat(T, ρ, q_tot)

Return the internal energy per unit mass in thermodynamic equilibrium at
saturation, given the temperature `T`, (moist-)air density `ρ`, and total
specific humidity `q_tot`.
"""
function internal_energy_sat(T, ρ, q_tot)

    # get equilibrium phase partitioning
    _q_liq, _q_ice = phase_partitioning_eq(T, ρ, q_tot)

    return internal_energy(T, q_tot, _q_liq, _q_ice)

end

"""
    total_energy(KE, PE, T[, q_tot=0, q_liq=0, q_ice=0])

Return the total energy per unit mass, given the kinetic energy per unit
mass `e_kin`, the potential energy per unit mass `e_pot`, the temperature `T`, and,
optionally, the total specific humidity `q_tot`, the liquid specific humidity
`q_liq`, and the ice specific humidity `q_ice`.
"""
function total_energy(e_kin, e_pot, T, q_tot=0, q_liq=0, q_ice=0)

    return e_kin + e_pot + internal_energy(T, q_tot, q_liq, q_ice)

end

"""
    latent_heat_vapor(T)

Return the specific latent heat of vaporization at temperature `T`.
"""
function latent_heat_vapor(T)

     return latent_heat_generic(T, LH_v0, cp_v - cp_l)

end

"""
    latent_heat_sublim(T)

Return the specific latent heat of sublimation at temperature `T`.
"""
function latent_heat_sublim(T)

    return latent_heat_generic(T, LH_s0, cp_v - cp_i)

end

"""
    latent_heat_fusion(T)

Return the specific latent heat of fusion at temperature `T`.
"""
function latent_heat_fusion(T)

    return latent_heat_generic(T, LH_f0, cp_l - cp_i)

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
saturation_vapor_pressure(T, ::Liquid) = saturation_vapor_pressure(T, LH_v0, cp_v - cp_l)
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
    saturation_shum(T, ρ[, q_liq=0, q_ice=0])

Compute the saturation specific humidity, given the temperature `T` and
(moist-)air density `ρ`.

If the optional liquid, and ice specific humdities `q_tot` and `q_liq` are given,
the saturation specific humidity is that over a mixture of liquid and ice,
computed in a thermodynamically consistent way from the weighted sum of the
latent heats of the respective phase transitions (Pressel et al., JAMES, 2015).
That is, the saturation vapor pressure and from it the saturation
specific humidity are computed from a weighted mean of the latent heats of
vaporization and sublimation, with the weights given by the fractions of
condensate `q_liq`/(`q_liq` + `q_ice`) and `q_ice`/(`q_liq` + `q_ice`) that are liquid and
ice, respectively.

If the condensate specific humidities `q_liq` and `q_ice` are not given or are both
zero, the saturation specific humidity is that over a mixture of liquid and ice,
with the fraction of liquid given by temperature dependent `liquid_fraction(T)`
and the fraction of ice by the complement `1 - liquid_fraction(T)`.
"""
function saturation_shum(T, ρ, q_liq=0, q_ice=0)

    # get phase partitioning
    _liquid_frac = liquid_fraction(T, q_liq, q_ice)
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
    saturation_shum_from_pressure(T, ρ, p_vs)

Compute the saturation specific humidity, given the ambient air temperature `T`,
density `ρ`, and the saturation vapor pressure `p_vs`.
"""
function saturation_shum_from_pressure(T, ρ, p_vs)

    return min(typeof(ρ)(1), p_vs / (ρ * R_v * T))

end

"""
    saturation_excess(T, ρ, q_tot, q_liq=0, q_ice=0)

Compute the saturation excess in equilibrium, given the ambient air temperature
`T`, the (moist-)air density `ρ`, the total specific humidity `q_tot`, and,
optionally, the liquid specific humidity `q_liq`, and the ice specific humidity `q_ice`.

The saturation excess is the difference between the total specific humidity `q_tot`
and the saturation specific humidity in equilibrium, and it is defined to be
nonzero only if this difference is positive.
"""
function saturation_excess(T, ρ, q_tot, q_liq=0, q_ice=0)

    return max(typeof(q_tot)(0), q_tot - saturation_shum(T, ρ, q_liq, q_ice))

end

"""
    liquid_fraction(T[, q_liq=0, q_ice=0])

Return the fraction of condensate that is liquid.

If the optional input arguments `q_liq` and `q_ice` are not given or are zero, the
fraction of liquid is a function that is 1 above `T_freeze` and goes to zero below
`T_freeze`. If `q_liq` or `q_ice` are nonzero, the liquid fraction is computed from
them.
"""
function liquid_fraction(T, q_liq=0, q_ice=0)

q_c         = q_liq + q_ice     # condensate specific humidity

# For now: Heaviside function for partitioning into liquid and ice: all liquid
# for T > T_freeze; all ice for T <= T_freeze
_liquid_frac = heaviside(T - T_freeze)

return ifelse(q_c > 0, q_liq / q_c, _liquid_frac)

end

"""
    heaviside(t)

Return the Heaviside step function at `t`.
"""
function heaviside(t)
   typeof(t)(1//2) * (sign(t) + 1)
end

"""
    phase_partitioning_eq!(q_liq, q_ice, T, ρ, q_tot)

    # ASR Do we need this function at all(?)

<<<<<<< HEAD
    # ASR Do we need this function at all(?)

=======
>>>>>>> master
    Return the partitioning of the phases in equilibrium.

Given the temperature `T` and (moist-)air density `ρ`, `phase_partitioning_eq!`
partitions the total specific humidity `q_tot` into the liquid specific humidity
`q_liq` and ice specific humiditiy `q_l` using the `liquid_fraction`
function. The residual `q_tot - q_liq - q_ice` is the vapor specific humidity.
"""
<<<<<<< HEAD

function phase_partitioning_eq(T, ρ, q_t)
    _liquid_frac = liquid_fraction(T)   # fraction of condensate that is liquid
    q_vs         = saturation_shum(T, ρ) # saturation specific humidity
    q_c          = max(q_t - q_vs, 0) # condensate specific humidity
    q_l_out      = _liquid_frac * q_c  # liquid specific humidity
    q_i_out      = (1 - _liquid_frac) * q_c # ice specific humidity
    return q_l_out, q_i_out
end
=======
function phase_partitioning_eq(T, ρ, q_tot)
    _liquid_frac = liquid_fraction(T)   # fraction of condensate that is liquid
    q_vs         = saturation_shum(T, ρ) # saturation specific humidity
    q_c          = max(q_tot - q_vs, 0) # condensate specific humidity
    q_liq_out      = _liquid_frac * q_c  # liquid specific humidity
    q_ice_out      = (1 - _liquid_frac) * q_c # ice specific humidity
    return q_liq_out, q_ice_out

  end
>>>>>>> master

"""
    saturation_adjustment(e_int, ρ, q_tot[, T_init = T_triple])

Return the temperature that is consistent with the internal energy `e_int`,
(moist-)air density `ρ`, and total specific humidity `q_tot`.

The optional input value of the temperature `T_init` is taken as the initial
value of the saturation adjustment iterations.
"""
<<<<<<< HEAD
function saturation_adjustment(e_int, ρ, q_t, T_init = T_triple)
  if q_t < saturation_shum(air_temperature(e_int), ρ)
    return air_temperature(e_int, q_t)
  else
=======
function saturation_adjustment(e_int, ρ, q_tot, T_init = T_triple)
    if q_tot <= saturation_shum(max(0,air_temperature(e_int, q_tot)), ρ)
      return air_temperature(e_int, q_tot)   
   else
>>>>>>> master
    tol_abs = 1e-3*cv_d
    iter_max = 10
    args = (ρ, q_tot, e_int)
    T0 = max(T_min, air_temperature(e_int, q_tot, 0.0, 0.0))
    T1 = air_temperature(e_int, q_tot, 0.0, q_tot)
    roots_equation(x, ρ, q_tot, e_int) = internal_energy_sat(x, ρ, q_tot) - e_int
    T, converged = find_zero(roots_equation,
                             T0, T1,
                             args,
                             IterParams(tol_abs, iter_max),
                             SecantMethod()
                             )
<<<<<<< HEAD
    q_l, q_i = phase_partitioning_eq(T, ρ, q_t)
    return air_temperature(e_int, q_t, q_l, q_i)
=======
    q_liq, q_ice = phase_partitioning_eq(T, ρ, q_tot)
    return air_temperature(e_int, q_tot, q_liq, q_ice)
>>>>>>> master
  end
end

"""
    liquid_ice_pottemp(T, p[, q_tot=0, q_liq=0, q_ice=0])

Return the liquid-ice potential temperature, given the temperature `T`,
pressure `p`, total specific humidity `q_tot`, liquid specific humidity `q_liq`,
and ice specific humidity `q_ice`.
"""
function liquid_ice_pottemp(T, p, q_tot=0, q_liq=0, q_ice=0)

    # isobaric specific heat of moist air
    _cp_m   = cp_m(q_tot, q_liq, q_ice)

    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp(T, p, q_tot, q_liq, q_ice) * exp(-(LH_v0*q_liq + LH_s0*q_ice)/(_cp_m*T))

end

"""
    dry_pottemp(T, p, q_tot=0, q_liq=0, q_ice=0)

Return the dry potential temperature, given the temperature `T`,
pressure `p`, total specific humidity `q_tot`, liquid specific humidity `q_liq`,
and ice specific humidity `q_ice`.
"""
dry_pottemp(T, p, q_tot=0, q_liq=0, q_ice=0) = T / exner(p, q_tot, q_liq, q_ice)

"""
    exner(p, q_tot=0, q_liq=0, q_ice=0)

Return the Exner function, given the pressure `p`, total specific
humidity `q_tot`, liquid specific humidity `q_liq`, and ice specific humidity `q_ice`.
"""
function exner(p, q_tot=0, q_liq=0, q_ice=0)

    # gas constant and isobaric specific heat of moist air
    _R_m    = gas_constant_air(q_tot, q_liq, q_ice)
    _cp_m   = cp_m(q_tot, q_liq, q_ice)

    return (p/MSLP)^(_R_m/_cp_m)

end


end #module MoistThermodynamics.jl
