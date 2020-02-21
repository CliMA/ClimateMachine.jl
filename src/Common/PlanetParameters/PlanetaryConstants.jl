"""
    PlanetaryConstants

Module containing physical constants and parameters characterizing the planet.
"""
module PlanetaryConstants

using ..UniversalConstants
using DocStringExtensions

export PlanetConstants,
       AbstractPlanetConstants,
       EarthConstants
       # MarsConstants

"""
    PlanetConstants{FT<:AbstractFloat}

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PlanetConstants{FT<:AbstractFloat}
  "Von Karman constant (1)"
  k_Karman::FT
  "Molecular weight dry air (kg/mol)"
  molmass_dryair::FT
  "Gas constant dry air (J/kg/K)"
  R_d::FT = FT(gas_constant(FT)/molmass_dryair)
  "Adiabatic exponent dry air"
  kappa_d::FT
  "Isobaric specific heat dry air"
  cp_d::FT = FT(R_d/kappa_d)
  "Isochoric specific heat dry air"
  cv_d::FT = FT(cp_d - R_d)
  "Density of water (kg/m^3)"
  ρ_liq::FT
  "Density of liquid water (kg/m^3)"
  ρ_cloud_liq::FT = FT(ρ_liq)
  "Density of ice (kg/m^3)"
  ρ_ice::FT
  "Density of water ice (kg/m^3)"
  ρ_cloud_ice::FT = FT(ρ_ice)
  "Molecular weight (kg/mol)"
  molmass_water::FT
  "Molar mass ratio dry air/water"
  molmass_ratio::FT = FT(molmass_dryair/molmass_water)
  "Gas constant water vapor (J/kg/K)"
  R_v::FT = FT(gas_constant(FT)/molmass_water)
  "Isobaric specific heat vapor (J/kg/K)"
  cp_v::FT
  "Isobaric specific heat liquid (J/kg/K)"
  cp_l::FT
  "Isobaric specific heat ice (J/kg/K)"
  cp_i::FT
  "Isochoric specific heat vapor (J/kg/K)"
  cv_v::FT = FT(cp_v - R_v)
  "Isochoric specific heat liquid (J/kg/K)"
  cv_l::FT = FT(cp_l)
  "Isochoric specific heat ice (J/kg/K)"
  cv_i::FT = FT(cp_i)
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
  T_0::FT = FT(T_triple)
  "Latent heat vaporization at T_0 (J/kg)"
  LH_v0::FT
  "Latent heat sublimation at T_0 (J/kg)"
  LH_s0::FT
  "Latent heat of fusion at T_0 (J/kg)"
  LH_f0::FT = FT(LH_s0 - LH_v0)
  "Specific internal energy of vapor at T_0 (J/kg)"
  e_int_v0::FT = FT(LH_v0 - R_v*T_0)
  "Specific internal energy of ice at T_0 (J/kg)"
  e_int_i0::FT = FT(LH_f0)
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
  year_anom::FT = FT(365.26*day)
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
