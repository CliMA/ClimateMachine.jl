"""
    PlanetParameters

Module containing the planetary constants.
"""
module PlanetParameters
using ParametersType
using Unitful

getval = ParametersType.getval

# Physical constants
@parameter gas_constant       8.3144598u"J/mol/K"                         "Universal gas constant"            true
@parameter light_speed        2.99792458e8u"m/s"                          "Speed of light in vacuum"          true
@parameter h_Planck           6.626e-34u"m^2*kg/s"                        "Planck constant"                   true
@parameter k_Boltzmann        1.381e-23u"m^2*kg/s^2/K"                    "Boltzmann constant"                true
@parameter Stefan             5.670e-8u"W/m^2/K^4"                        "Stefan-Boltzmann constant"         true
@parameter astro_unit         1.4959787e11u"m"                            "Astronomical unit"                 true

# Properties of dry air
@parameter molmass_air        28.97e-3u"kg/mol"                           "Molecular weight dry air"          true
@parameter R_d                getval(gas_constant)/getval(molmass_air)    "Gas constant dry air"              true
@parameter kappa_d            2//7                                        "Adiabatic exponent dry air"        true
@parameter cp_d               getval(R_d)/getval(kappa_d)                 "Isobaric specific heat dry air"    true

# Properties of water
@parameter dens_liquid        1e3u"kg/m^3"                                "Density of liquid water"           true
@parameter molmass_water      18.01528e-3u"kg/mol"                        "Molecular weight"                  true
@parameter R_v                getval(gas_constant)/getval(molmass_water)  "Gas constant water vapor"          true
@parameter cp_v               1859u"J/kg/K"                               "Isobaric specific heat vapor"      true
@parameter cp_l               4181u"J/kg/K"                               "Isobaric specific heat liquid"     true
@parameter cp_i               2100u"J/kg/K"                               "Isobaric specific heat ice"        true
@parameter Tfreeze            273.15u"K"                                  "Freezing point temperature"        true
@parameter T0                 273.16u"K"                                  "Triple point temperature"          true
@parameter L_v0               2.5008e6u"J/kg"                             "Latent heat vaporization at T0"    true
@parameter L_s0               2.8341e6u"J/kg"                             "Latent heat sublimation at T0"     true
@parameter L_f0               getval(L_s0)-getval(L_v0)                   "Latent heat of fusion at T0"       true
@parameter sat_vapor_press_0  611.657u"Pa"                                "Saturation vapor pressure at T0"   true

# Properties of sea water
@parameter dens_ocean         1.035e3u"kg/m^3"                            "Reference density sea water"       true
@parameter cp_ocean           3989.25u"J/kg/K"                            "Specific heat sea water"           true

# Planetary parameters
@parameter planet_radius      6.371e6u"m"                                 "Mean planetary radius"             true
@parameter day                86400u"s"                                   "Length of day"                     true
@parameter Omega              7.2921159e-5u"1/s"                          "Ang. velocity planetary rotation"  true
@parameter grav               9.81u"m/s^2"                                "Gravitational acceleration"        true
@parameter year_anom          365.26*getval(day)                          "Length of anomalistic year"        true
@parameter orbit_semimaj      1*getval(astro_unit)                        "Length of semimajor orbital axis"  true
@parameter TSI                1362u"W/m^2"                                "Total solar irradiance"            true
@parameter mslp               1.01325e5u"Pa"                              "Mean sea level pressure"           true

end
