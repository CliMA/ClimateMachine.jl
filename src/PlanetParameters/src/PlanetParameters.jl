"""
    PlanetParameters

Module containing physical constants and parameters characterizing the planet.
"""
module PlanetParameters
using ParametersType

# Physical constants
@exportparameter gas_constant      8.3144598     "Universal gas constant (J/mol/K)"
@exportparameter light_speed       2.99792458e8  "Speed of light in vacuum (m/s)"
@exportparameter h_Planck          6.626e-34     "Planck constant (m^2 kg/s)"
@exportparameter k_Boltzmann       1.381e-23     "Boltzmann constant (m^2 kg/s^2/K)"
@exportparameter Stefan            5.670e-8      "Stefan-Boltzmann constant (W/m^2/K^4)"
@exportparameter astro_unit        1.4959787e11  "Astronomical unit (m)"

# Properties of dry air
@exportparameter molmass_air       28.97e-3      "Molecular weight dry air (kg/mol)"
@exportparameter R_d               gas_constant/
                                   molmass_air   "Gas constant dry air (J/kg/K)"
@exportparameter kappa_d           2//7          "Adiabatic exponent dry air"
@exportparameter cp_d              R_d/kappa_d   "Isobaric specific heat dry air"

# Properties of water
@exportparameter dens_liquid       1e3           "Density of liquid water (kg/m^3)"
@exportparameter molmass_water     18.01528e-3   "Molecular weight (kg/mol)"
@exportparameter molmass_ratio     molmass_air/
                                   molmass_water "Molar mass ratio dry air/water"
@exportparameter R_v               gas_constant/
                                   molmass_water "Gas constant water vapor (J/kg/K)"
@exportparameter cp_v              1859          "Isobaric specific heat vapor (J/kg/K)"
@exportparameter cp_l              4181          "Isobaric specific heat liquid (J/kg/K)"
@exportparameter cp_i              2100          "Isobaric specific heat ice (J/kg/K)"
@exportparameter T_freeze          273.15        "Freezing point temperature (K)"
@exportparameter T_triple          273.16        "Triple point temperature (K)"
@exportparameter T_0               T_trip        "Reference temperature (K)"
@exportparameter L_v0              2.5008e6      "Latent heat vaporization at T_0 (J/kg)"
@exportparameter L_s0              2.8341e6      "Latent heat sublimation at T_0 (J/kg)"
@exportparameter L_f0              L_s0 - L_v0   "Latent heat of fusion at T_0 (J/kg)"
@exportparameter sat_vapor_press_triple  611.657 "Triple point saturation
                                                  vapor pressure (Pa)"

# Properties of sea water
@exportparameter dens_ocean        1.035e3       "Reference density sea water (kg/m^3)"
@exportparameter cp_ocean          3989.25       "Specific heat sea water (J/kg/K)"

# Planetary parameters
@exportparameter planet_radius     6.371e6       "Mean planetary radius (m)"
@exportparameter day               86400         "Length of day (s)"
@exportparameter Omega             7.2921159e-5  "Ang. velocity planetary rotation (1/s)"
@exportparameter grav              9.81          "Gravitational acceleration (m/s^2)"
@exportparameter year_anom         365.26*day    "Length of anomalistic year (s)"
@exportparameter orbit_semimaj     1*astro_unit  "Length of semimajor orbital axis (m)"
@exportparameter TSI               1362          "Total solar irradiance (W/m^2)"
@exportparameter MSLP              1.01325e5     "Mean sea level pressure (Pa)"

end
