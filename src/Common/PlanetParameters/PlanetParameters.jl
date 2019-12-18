"""
    PlanetParameters

Module containing physical constants and parameters characterizing the planet.
"""
module PlanetParameters
using Unitful
import Unitful: Quantity
using ..ParametersType

# Physical constants
@exportparameter gas_constant      8.3144598u"J/mol/K"       "Universal gas constant (J/mol/K)"
@exportparameter light_speed       2.99792458e8u"m/s"        "Speed of light in vacuum (m/s)"
@exportparameter h_Planck          6.626e-34u"m^2*kg/s"      "Planck constant (m^2 kg/s)"
@exportparameter k_Boltzmann       1.381e-23u"m^2*kg/s^2/K"  "Boltzmann constant (m^2 kg/s^2/K)"
@exportparameter Stefan            5.670e-8u"W/m^2/K^4"      "Stefan-Boltzmann constant (W/m^2/K^4)"
@exportparameter astro_unit        1.4959787e11u"m"          "Astronomical unit (m)"
@exportparameter k_Karman          0.4                       "Von Karman constant (1)"

# Properties of dry air
@exportparameter molmass_dryair    28.97e-3u"kg/mol"         "Molecular weight dry air (kg/mol)"
@exportparameter R_d               gas_constant/
                                   molmass_dryair            "Gas constant dry air (J/kg/K)"
@exportparameter kappa_d           2//7                      "Adiabatic exponent dry air"
@exportparameter cp_d              R_d/kappa_d               "Isobaric specific heat dry air"
@exportparameter cv_d              cp_d - R_d                "Isochoric specific heat dry air"

# Properties of water
@exportparameter ρ_cloud_liq       1e3u"kg/m^3"              "Density of liquid water (kg/m^3)"
@exportparameter ρ_cloud_ice       916.7u"kg/m^3"            "Density of ice water (kg/m^3)"
@exportparameter molmass_water     18.01528e-3u"kg/mol"      "Molecular weight (kg/mol)"
@exportparameter molmass_ratio     molmass_dryair/
                                   molmass_water             "Molar mass ratio dry air/water"
@exportparameter R_v               gas_constant/
                                   molmass_water             "Gas constant water vapor (J/kg/K)"
@exportparameter cp_v              1859u"J/kg/K"             "Isobaric specific heat vapor (J/kg/K)"
@exportparameter cp_l              4181u"J/kg/K"             "Isobaric specific heat liquid (J/kg/K)"
@exportparameter cp_i              2100u"J/kg/K"             "Isobaric specific heat ice (J/kg/K)"
@exportparameter cv_v              cp_v - R_v                "Isochoric specific heat vapor (J/kg/K)"
@exportparameter cv_l              cp_l                      "Isochoric specific heat liquid (J/kg/K)"
@exportparameter cv_i              cp_i                      "Isochoric specific heat ice (J/kg/K)"
@exportparameter T_freeze          273.15u"K"                "Freezing point temperature (K)"
@exportparameter T_min             150.0u"K"                 "Minimum temperature guess in saturation adjustment (K)"
@exportparameter T_max             1000.0u"K"                "Maximum temperature guess in saturation adjustment (K)"
@exportparameter T_icenuc          233.00u"K"                "Homogeneous nucleation temperature (K)"
@exportparameter T_triple          273.16u"K"                "Triple point temperature (K)"
@exportparameter T_0               T_triple                  "Reference temperature (K)"
@exportparameter LH_v0             2.5008e6u"J/kg"           "Latent heat vaporization at T_0 (J/kg)"
@exportparameter LH_s0             2.8344e6u"J/kg"           "Latent heat sublimation at T_0 (J/kg)"
@exportparameter LH_f0             LH_s0 - LH_v0             "Latent heat of fusion at T_0 (J/kg)"
@exportparameter e_int_v0          LH_v0 - R_v*T_0           "Specific internal energy of vapor at T_0 (J/kg)"
@exportparameter e_int_i0          LH_f0                     "Specific internal energy of ice at T_0 (J/kg)"
@exportparameter press_triple      611.657u"Pa"              "Triple point vapor pressure (Pa)"

# Properties of sea water
@exportparameter ρ_ocean           1.035e3u"kg/m^3"          "Reference density sea water (kg/m^3)"
@exportparameter cp_ocean          3989.25u"J/kg/K"          "Specific heat sea water (J/kg/K)"

# Planetary parameters
@exportparameter planet_radius     6.371e6u"m"               "Mean planetary radius (m)"
@exportparameter day               86400u"s"                 "Length of day (s)"
@exportparameter Omega             7.2921159e-5u"s^-1"       "Ang. velocity planetary rotation (1/s)"
@exportparameter grav              9.81u"m/s^2"              "Gravitational acceleration (m/s^2)"
@exportparameter year_anom         365.26*day                "Length of anomalistic year (s)"
@exportparameter orbit_semimaj     1*astro_unit              "Length of semimajor orbital axis (m)"
@exportparameter TSI               1362u"W/m^2"              "Total solar irradiance (W/m^2)"
@exportparameter MSLP              1.01325e5u"Pa"            "Mean sea level pressure (Pa)"

end
