"""
    MoistThermodynamics

Moist thermodynamic functions, e.g., for air pressure (atmosphere equation
of state), latent heats of phase transitions, saturation vapor pressures, and
saturation specific humidities
"""
module MoistThermodynamics

using PlanetParameters

# Atmospheric equation of state
export air_pressure, air_temperature

# Energies
export total_energy, internal_energy

# Specific heats of moist air
export cp_m, cv_m, gas_constant_air

# Latent heats
export latent_heat_vapor, latent_heat_sublim, latent_heat_fusion

# Saturation vapor pressures and specific humidities over liquid and ice
export sat_vapor_press_liquid, sat_vapor_press_ice, sat_shum_generic, sat_shum

# Condensate partitioning
export liquid_fraction

"""
    gas_constant_air([q_t=0, q_l=0, q_i=0])

Return the specific gas constant of moist air, given the total specific
humidity `q_t`, and, optionally, the liquid specific humidity `q_l`,
and the ice specific humidity `q_i`. When no input argument is given, it
returns the specific gas constant of dry air.
"""
function gas_constant_air(q_t=0, q_l=0, q_i=0)

    return R_d * ( 1 .+  (molmass_ratio - 1)*q_t .- molmass_ratio*(q_l .+ q_i) )

end

"""
    air_pressure(T, density[, q_t=0, q_l=0, q_i=0])

Return the air pressure from the equation of state (ideal gas law), given
the air temperature `T`, the `density`, and, optionally, the total specific
humidity `q_t`, the liquid specific humidity `q_l`, and the ice specific
humidity `q_i`. Without the specific humidity arguments, it returns the air
pressure from the equation of state of dry air.
"""
function air_pressure(T, density, q_t=0, q_l=0, q_i=0)

    return gas_constant_air(q_t, q_l, q_i) .* density .* T

end

"""
    cp_m([q_t=0, q_l=0, q_i=0])

Return the isobaric specific heat capacity of moist air, given the
total water specific humidity `q_t`, liquid specific humidity `q_l`, and
ice specific humidity `q_i`. Without the specific humidity arguments, it returns
the isobaric specific heat capacity of dry air.
"""
function cp_m(q_t=0, q_l=0, q_i=0)

    return cp_d .+ (cp_v - cp_d)*q_t .- (cp_v - cp_l)*q_l .- (cp_v - cp_i)*q_i

end

"""
    cv_m([q_t=0, q_l=0, q_i=0])

Return the isochoric specific heat capacity of moist air, given the
total water specific humidity `q_t`, liquid specific humidity `q_l`, and
ice specific humidity `q_i`. Without the specific humidity arguments, it returns
the isochoric specific heat capacity of dry air.
"""
function cv_m(q_t=0, q_l=0, q_i=0)

    return cv_d .+ (cv_v - cv_d)*q_t .- (cv_v - cv_l)*q_l .- (cv_v - cv_i)*q_i

end

"""
    air_temperature(E_int[, q_t=0, q_l=0, q_i=0])

Return the air temperature, given the internal energy `E_int` per unit mass,
and, optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function air_temperature(internal_energy, q_t=0, q_l=0, q_i=0)

    return T_0 .+
        ( internal_energy .- (q_t .- q_l) * IE_v0 .+ q_i * (IE_v0 + IE_i0) )./
            cv_m(q_t, q_l, q_i)

end

"""
    internal_energy(T[, q_t=0, q_l=0, q_i=0])

Return the internal energy per unit mass, given the temperature `T`, and,
optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function internal_energy(T, q_t=0, q_l=0, q_i=0)

    return cv_m(q_t, q_l, q_i) .* (T .- T_0) .+
        (q_t .- q_l) * IE_v0 .- q_i * (IE_v0 + IE_i0)

end


"""
    total_energy(KE, PE, T[, q_t=0, q_l=0, q_i=0])

Return the total energy per unit mass, given the kinetic energy per unit
mass `KE`, the potential energy per unit mass `PE`, the temperature `T`, and,
optionally, the total specific humidity `q_t`, the liquid specific humidity
`q_l`, and the ice specific humidity `q_i`.
"""
function total_energy(kinetic_energy, potential_energy, T, q_t=0, q_l=0, q_i=0)

    return kinetic_energy .+ potential_energy .+
        internal_energy(T, q_t, q_l, q_i)

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

    return LH_0 .+ cp_diff * (T .- T_0)

end

"""
    sat_vapor_press_liquid(T)

Return the saturation vapor pressure over a plane liquid surface at
temperature `T`.
"""
function sat_vapor_press_liquid(T)

    return sat_vapor_press_generic(T, LH_v0, cp_v - cp_l)

end

"""
    sat_vapor_press_ice(T)

Return the saturation vapor pressure over a plane ice surface at
temperature `T`.
"""
function sat_vapor_press_ice(T)

    return sat_vapor_press_generic(T, LH_s0, cp_v - cp_i)

end

"""
    sat_vapor_press_generic(T, LH_0, cp_diff)

Compute the saturation vapor pressure over a plane surface by integration
of the Clausius-Clepeyron relation.

The Clausius-Clapeyron relation

    dlog(p_vs)/dT = [LH_0 + cp_diff * (T-T_0)]/(R_v*T^2)

is integrated from the triple point temperature `T_triple`, using
Kirchhoff's relation

    L = LH_0 + cp_diff * (T - T_0)

for the specific latent heat L with constant isobaric specific
heats of the phases. The linear dependence of the specific latent heat
on temperature `T` allows analytic integration of the Clausius-Clapeyron
relation to obtain the saturation vapor pressure `p_vs` as a function of
the triple point pressure `press_triple`.
"""
function sat_vapor_press_generic(T, LH_0, cp_diff)

    return press_triple * (T/T_triple).^(cp_diff/R_v).*
        exp.( (LH_0 - cp_diff*T_0)/R_v * (1 / T_triple .- 1 ./ T) )

end

"""
    sat_shum_generic(T, p, q_t[; phase="liquid"])

Compute the saturation specific humidity over a plane surface of
condensate, at temperature `T`, pressure `p`, and total water specific
humidity `q_t`. The argument `phase` can be ``"liquid"`` or ``"ice"`` and
indicates the condensed phase.
"""
function sat_shum_generic(T, p, q_t; phase="liquid")

    saturation_vapor_pressure_function = Symbol(string("sat_vapor_press_", phase))
    p_vs = eval(saturation_vapor_pressure_function)(T)

    return sat_shum_from_pressure(p, p_vs, q_t)

end

"""
    sat_shum(T, p, q_t[, q_l=0, q_i=0])

Compute the saturation specific humidity at the temperature `T`, pressure `p`,
and total water specific humidity `q_t`.

If the optional liquid specific humdity `q_l` and ice specific humidity `q_i`
are given, the saturation specific humidity is that over a mixture of
liquid and ice, computed in a thermodynamically consistent way from the weighted
sum of the latent heats of the respective phase transitions (Pressel et al.,
JAMES, 2015). That is, the saturation vapor pressure and from it the saturation
specific humidity are computed from a weighted mean of the latent heats of
vaporization and sublimation, with the weights given by the fractions of condensate
`q_l`/(`q_l` + `q_i`) and `q_i`/(`q_l` + `q_i`) that are liquid and ice,
respectively.

If the condensate specific humidities `q_l` and `q_i` are not given or are both
zero, the saturation specific humidity is that over a mixture of liquid and ice,
with the fraction of liquid given by temperature dependent `liquid_fraction(T)`
and the fraction of ice by the complement `1 - liquid_fraction(T)`.
"""
function sat_shum(T, p, q_t, q_l=0, q_i=0)

    #FIXME some problem here with variable types, probably with T_freeze in liquid_fraction

    # get phase partitioning
    _liquid_frac = liquid_fraction(T, q_l, q_i)
    _ice_frac    = 1 .- _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0        = _liquid_frac * LH_v0 .+ _ice_frac * LH_s0
    cp_diff     = _liquid_frac * (cp_v - cp_l) .+ _ice_frac * (cp_v - cp_i)

    # saturation vapor pressure over possible mixture of liquid and ice
    p_vs        = sat_vapor_press_generic(T, LH_0, cp_diff)

    return sat_shum_from_pressure(p, p_vs, q_t)

end

"""
    sat_shum_from_pressure(p, p_vs, q_t)

Compute the saturation specific humidity given the ambient total pressure `p`,
the saturation vapor pressure `p_vs`, and the total water specific humidity `q_t`.
"""
function sat_shum_from_pressure(p, p_vs, q_t)

    return 1/molmass_ratio * (1 .- q_t) .* p_vs ./ (p .- p_vs)

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
# FIXME: Need to specify Number(T_freeze) here in Boolean expression;
#         can we change ParametersType to avoid this?
_liquid_frac = ifelse.(T .> Number(T_freeze), 1, 0)

return ifelse.(q_c .> 0, q_l ./ q_c, _liquid_frac)

end

"""
    saturation_adjustment(E_int, T, q_t, q_l, q_i)

Compute temperature `T` and specific humidities of condensate from the internal
energy `E_int` by saturation adjustment.

The function takes the internal energy per unit mass `E_int` and total water
specific humidity `q_t` as input variables and returns the temperature `T`, the
liquid water specific humidity `q_l`, and the ice specific humidity `q_i`. Input
values for `q_l`, and `q_i` are used as initial values for the saturation
adjustment.
"""
function saturation_adjustment(E_int, T, q_t, q_l, q_i)

# initially, assume condensate as given at Input and compute temperature
T = air_temperature(E_int, q_t, q_l, q_i)

# FIXME need to complete saturation adjustment

end
