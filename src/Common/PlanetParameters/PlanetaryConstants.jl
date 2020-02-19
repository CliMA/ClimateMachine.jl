"""
    PlanetaryConstants

Module containing physical constants and parameters characterizing the planet.
"""
module PlanetaryConstants

using ..UniversalConstants

export PlanetConstants,
       AbstractPlanetConstants,
       EarthConstants
       # MarsConstants

struct PlanetConstants{FT}
  "Von Karman constant (1)"
  k_Karman::FT
  "Molecular weight dry air (kg/mol)"
  molmass_dryair::FT
  "Gas constant dry air (J/kg/K)"
  R_d::FT
  "Adiabatic exponent dry air"
  kappa_d::FT
  "Isobaric specific heat dry air"
  cp_d::FT
  "Isochoric specific heat dry air"
  cv_d::FT
  "Density of liquid water (kg/m^3)"
  ρ_cloud_liq::FT
  "Density of ice water (kg/m^3)"
  ρ_cloud_ice::FT
  "Molecular weight (kg/mol)"
  molmass_water::FT
  "Molar mass ratio dry air/water"
  molmass_ratio::FT
  "Gas constant water vapor (J/kg/K)"
  R_v::FT
  "Isobaric specific heat vapor (J/kg/K)"
  cp_v::FT
  "Isobaric specific heat liquid (J/kg/K)"
  cp_l::FT
  "Isobaric specific heat ice (J/kg/K)"
  cp_i::FT
  "Isochoric specific heat vapor (J/kg/K)"
  cv_v::FT
  "Isochoric specific heat liquid (J/kg/K)"
  cv_l::FT
  "Isochoric specific heat ice (J/kg/K)"
  cv_i::FT
  "Freezing point temperature (K)"
  T_freeze::FT
  "Minimum temperature guess in saturation adjustment (K)"
  T_min::FT
  "Maximum temperature guess in saturation adjustment (K)"
  T_max::FT
  "Homogeneous nucleation temperature (K)"
  T_icenuc::FT
  "Triple point temperature (K)"
  T_triple::FT
  "Reference temperature (K)"
  T_0::FT
  "Latent heat vaporization at T_0 (J/kg)"
  LH_v0::FT
  "Latent heat sublimation at T_0 (J/kg)"
  LH_s0::FT
  "Latent heat of fusion at T_0 (J/kg)"
  LH_f0::FT
  "Specific internal energy of vapor at T_0 (J/kg)"
  e_int_v0::FT
  "Specific internal energy of ice at T_0 (J/kg)"
  e_int_i0::FT
  "Triple point vapor pressure (Pa)"
  press_triple::FT
  "Reference density sea water (kg/m^3)"
  ρ_ocean::FT
  "Specific heat sea water (J/kg/K)"
  cp_ocean::FT
  "Mean planetary radius (m)"
  planet_radius::FT
  "Length of day (s)"
  day::FT
  "Ang. velocity planetary rotation (1/s)"
  Omega::FT
  "Gravitational acceleration (m/s^2)"
  grav::FT
  "Length of anomalistic year (s)"
  year_anom::FT
  "Length of semimajor orbital axis (m)"
  orbit_semimaj::FT
  "Total solar irradiance (W/m^2)"
  TSI::FT
  "Mean sea level pressure (Pa)"
  MSLP::FT
end


abstract type AbstractPlanetConstants end

struct EarthConstants <: AbstractPlanetConstants end
include("EarthConstants.jl")

# struct MarsConstants <: AbstractPlanetConstants end
# include("MarsConstants.jl")

end
