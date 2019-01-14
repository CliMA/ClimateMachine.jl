module MoistThermodynamics
"""
    MoistThermodynamics

Module containing moist thermodynamic functions, e.g., for air pressure
(atmosphere equation of state), latent heats of phase transitions, and
saturation vapor pressures.
"""

using PlanetParameters

# Atmospheric equation of state
export air_pressure, air_temperature, energy_tot

# Specific heats of moist air
export cp_m, cv_m, gas_constant_moist

# Latent heats
export latent_heat_vapor, latent_heat_sublim, latent_heat_fusion

# Saturation vapor pressures and specific humidities over liquid and ice
export sat_vapor_press_liquid, sat_vapor_press_ice, sat_shum

"""
    gas_constant_moist(q_t, q_l, q_i)

Computes the specific gas constant of moist air given the total specific
humidity `q_t`, the liquid specific humidity `q_l`, and the ice specific
humidity `q_i`.
"""
function gas_constant_moist(q_t, q_l, q_i)

    return R_d * ( 1 .+  (molmass_ratio - 1)*q_t .- molmass_ratio*(q_l .+ q_i) )

end

"""
    air_pressure(T, density, q_t, q_l, q_i)

Computes the air pressure from the equation of state (ideal gas law) given
the air temperature `T`, the `density`, the total specific humidity `q_t`, the
liquid specific humidity `q_l`, and the ice specific humidity `q_i`.
"""
function air_pressure(T, density, q_t, q_l, q_i)

    return gas_constant_moist(q_t, q_l, q_i) .* density .* T

end

"""
    cp_m(q_t, q_l, q_i)

Computes the isobaric specific heat capacity of moist air given the
total water specific humidity `q_t`, liquid specific humidity `q_l`, and
ice specific humidity `q_i`.
"""
function cp_m(q_t, q_l, q_i)

    return cp_d .+ (cp_v - cp_d)*q_t .- (cp_v - cp_l)*q_l .- (cp_v - cp_i)*q_i

end

"""
    cv_m(q_t, q_l, q_i)

Computes the isochoric specific heat capacity of moist air given the
total water specific humidity `q_t`, liquid specific humidity `q_l`, and
ice specific humidity `q_i`.
"""
function cv_m(q_t, q_l, q_i)

    return cv_d .+ (cv_v - cv_d)*q_t .- (cv_v - cv_l)*q_l .- (cv_v - cv_i)*q_i

end

"""
    air_temperature(E_tot, KE, PE, q_t, q_l, q_i)

Computes the temperature given the total energy `E_tot`, kinetic energy `KE`,
potential energy `PE` (all per unit mass), and the total specific humidity `q_t`,
the liquid specific humidity `q_l`, and the ice specific humidity `q_i`.
"""
function air_temperature(energy_tot, kinetic_energy, potential_energy, q_t, q_l, q_i)

    return T_0 .+ ( energy_tot .- kinetic_energy .- potential_energy
                .- (q_t .- q_l) * IE_v0 .+ q_i * (IE_i0 + IE_v0)
               )./ cv_m(q_t, q_l, q_i)

end

"""
    energy_tot(KE, PE, T, q_t, q_l, q_i)

Computes the total energy per unit mass given the kinetic energy per unit
mass `KE`, the potential energy per unit mass `PE`, the temperature `T`, the
total specific humidity `q_t`, the liquid specific humidity `q_l`, and the
ice specific humidity `q_i`. (U)
"""
function energy_tot(kinetic_energy, potential_energy, T, q_t, q_l, q_i)

    return kinetic_energy .+ potential_energy .+
        cv_m(q_t, q_l, q_i) .* (T .- T_0) .+
        (q_t .- q_l) * IE_v0 .- q_i * (IE_v0 + IE_i0)

end

"""
    latent_heat_vapor(T)

Computes the specific latent heat of vaporization at temperature `T`.
"""
function latent_heat_vapor(T)

     return latent_heat_generic(T, LH_v0, cp_v - cp_l)

end

"""
    latent_heat_sublim(T)

Computes the specific latent heat of sublimation at temperature `T`.
"""
function latent_heat_sublim(T)

    return latent_heat_generic(T, LH_s0, cp_v - cp_i)

end

"""
    latent_heat_fusion(T)

Computes the specific latent heat of fusion at temperature `T`.
"""
function latent_heat_fusion(T)

    return latent_heat_generic(T, LH_f0, cp_l - cp_i)

end

"""
    latent_heat_generic(T, LH_0, cp_diff)

Computes the specific latent heat of a generic phase transition between
two phases using Kirchhoff's relation and assuming constant isobaric
specifc heat capacities of the two phases. `T` is the temperature, `LH_0` is
the latent heat of the phase transition at `T_0`, and `cp_diff` is the
difference between the isobaric specific heat capacities (heat capacity in
higher-temperature phase minus that in lower-temperature phase).
"""
function latent_heat_generic(T, LH_0, cp_diff)

    return LH_0 .+ cp_diff * (T .- T_0)

end

"""
    sat_vapor_press_liquid(T)

Returns the saturation vapor pressure over a plane liquid surface at
temperature `T`.
"""
function sat_vapor_press_liquid(T)

    return sat_vapor_press_generic(T, LH_v0, cp_v - cp_l)

end

"""
    sat_vapor_press_ice(T)

Returns the saturation vapor pressure over a plane ice surface at
temperature `T`.
"""
function sat_vapor_press_ice(T)

    return sat_vapor_press_generic(T, LH_s0, cp_v - cp_i)

end

"""
    sat_vapor_press_generic(T, LH_0, cp_diff)

Computes the saturation vapor pressure by integration of the
Clausius-Clepeyron relation from the triple point temperature `T_triple`,
using Kirchhoff's relation

    L = LH_0 + cp_diff * (T - T_0)

for the specific latent heat L with constant isobaric specific
heats of the phases. The linear dependence of the specific latent heat
on temperature `T` allows analytic integration of the Clausius-Clapeyron
relation

    dlog(e_s)/dT = [LH_0 + cp_diff * (T-T_0)]/(R_v*T^2)

to obtain the saturation vapor pressure `es` as a function of the triple
point pressure `press_triple`.
"""
function sat_vapor_press_generic(T, LH_0, cp_diff)

    return press_triple * (T/T_triple).^(cp_diff/R_v).*
        exp.( (LH_0 - cp_diff*T_0)/R_v * (1 / T_triple .- 1 ./ T) )

end

"""
    sat_shum_generic(T, p, q_t, phase)

Computes the saturation specific humidity over a plane surface of
condensate, at temperature `T`, pressure `p`, and total water specific
humidity `q_t`. The argument `phase` can be ``"liquid"`` or ``"ice"`` and
indicates the condensed phase.
"""
function sat_shum_generic(T, p, q_t, phase)

    saturation_vapor_pressure_function = Symbol(string("sat_vapor_press_", phase))
    p_vs = eval(saturation_vapor_pressure_function)(T)

    return sat_shum_from_pressure(p, p_vs, q_t)

end

"""
    sat_shum(T, p, q_t, q_l, q_i)

Computes the saturation specific humidity at the temperature `T`, pressure `p`,
total water specific humidity `q_t`, liquid specific humdity `q_l`, and ice
specific humidity `q_i`. The saturation specific humidity is obtained from the
weighted mean of the saturation vapor pressures over a plane liquid surface and
a plane ice surface, with the weights given by the fractions of condensate
`q_l`/(`q_l` + `q_i`) and `q_i`/(`q_l` + `q_i`) that are liquid and ice,
respectively. In case the condensate specific humidities `q_l` and `q_i` are
both zero, the saturation specific humidity over liquid is returned.
"""
function sat_shum(T, p, q_t, q_l, q_i)

    # adding machine precision epsilon to the liquid specific humidity so that
    # division by zero is avoided, and the saturation vapor pressure over liquid
    # is used if both q_l and q_i are zero
    q_l         = q_l .+ eps(typeof(q_l[1]))
    liquid_frac = q_l ./ (q_l .+ q_i)
    ice_frac    = q_i ./ (q_l .+ q_i)

    p_vs = liquid_frac .* sat_vapor_press_liquid(T) .+
        ice_frac .* sat_vapor_press_ice(T)

    return sat_shum_from_pressure(p, p_vs, q_t)

end

"""
    sat_shum_from_pressure(p, p_vs, q_t)

Computes the saturation specific humidity given the ambient total pressure `p`,
the saturation vapor pressure `p_vs`, and the total water specific humidity `q_t`.
"""
function sat_shum_from_pressure(p, p_vs, q_t)

    return 1/molmass_ratio * (1 .- q_t) .* p_vs ./ (p .- p_vs)

end

end
