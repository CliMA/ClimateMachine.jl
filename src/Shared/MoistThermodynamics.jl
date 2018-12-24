module MoistThermodynamics
"""
    MoistThermodynamics

Module containing moist thermodynamic functions, e.g., for air pressure
(atmosphere equation of state), latent heats of phase transitions, and
saturation vapor pressures.
"""

# Tests to add:
# air_pressure([1, 1, 1], [1, 1, 2], [1, 0, 1], [0, 0, 0.5], [0, 0, 0]) == [R_v, R_d, R_v]
# gas_constant_moist([0, 1, 0.5], [0, 0, 0.5], [0, 0, 0]) == [R_d, R_v, R_d/2]
# cp_m([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) == [cp_d, cp_v, cp_l, cp_i]
# cv_m([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) == [cp_d - R_d, cp_v - R_v, cv_l, cv_i]
# air_temperature(cv_d*(300-T_0).+[0, 10, 10], [0, 10, 0], [0, 0, 10], 0, 0, 0) == [300, 300, 300]
# qt=0.23; air_temperature(cv_m([0, qt], 0, 0).*(300-T_0).+[0, qt*IE_v0], 0, 0, [0, qt], 0, 0) == [300, 300]
# latent_heat_vapor(T_0) == LH_v0
# latent_heat_fusion(T_0) == LH_f0
# latent_heat_sublim(T_0) == LH_s0
# sat_vapor_press_liquid(T_triple) == press_triple
# sat_vapor_press_ice(T_triple) == press_triple

using PlanetParameters

# Atmospheric equation of state
export air_pressure, air_temperature

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

Computes air pressure from the equation of state (ideal gas law) given
the air_temperature `T`, `density`, and the total specific humidity `q_t`, the
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
                .- (q_t .- q_l) * IE_v0 .+ q_i * (IE_i0 - IE_v0)
               )./ cv_m(q_t, q_l, q_i)

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
    sat_shum(T, p, q_t, phase)

Computes the saturation specific humidity over a plane surface of
condensate at temperature `T`, pressure `p`, and total water specific
humidity `q_t`. The argument `phase` can be ``"liquid"`` or ``"ice"`` and indicates
the condensed phase.
"""
function sat_shum(T, p, q_t, phase)

    saturation_vapor_pressure_function = Symbol(string("sat_vapor_press_", phase))
    es = eval(saturation_vapor_pressure_function)(T)

    return 1/molmass_ratio * (1 .- q_t) .* es ./ (p .- es)

end

end
