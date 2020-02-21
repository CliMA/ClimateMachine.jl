function PlanetConstants{FT}(::Type{EarthConstants}; kwargs...) where {FT}

  defaults = (
      k_Karman =          FT(0.4)                             , # Von Karman constant (1)
      molmass_dryair =    FT(28.97e-3)                        , # Molecular weight dry air (kg/mol)
      kappa_d =           FT(2//7)                            , # Adiabatic exponent dry air
      ρ_liq =             FT(1e3)                             , # Density of water (kg/m^3)
      ρ_ice =             FT(916.7)                           , # Density of ice (kg/m^3)
      molmass_water =     FT(18.01528e-3)                     , # Molecular weight (kg/mol)
      cp_v =              FT(1859)                            , # Isobaric specific heat vapor (J/kg/K)
      cp_l =              FT(4181)                            , # Isobaric specific heat liquid (J/kg/K)
      cp_i =              FT(2100)                            , # Isobaric specific heat ice (J/kg/K)
      T_freeze =          FT(273.15)                          , # Freezing point temperature (K)
      T_min =             FT(150.0)                           , # Minimum temperature guess in saturation adjustment (K)
      T_max =             FT(1000.0)                          , # Maximum temperature guess in saturation adjustment (K)
      T_icenuc =          FT(233.00)                          , # Homogeneous nucleation temperature (K)
      T_triple =          FT(273.16)                          , # Triple point temperature (K)
      LH_v0 =             FT(2.5008e6)                        , # Latent heat vaporization at T_0 (J/kg)
      LH_s0 =             FT(2.8344e6)                        , # Latent heat sublimation at T_0 (J/kg)
      press_triple =      FT(611.657)                         , # Triple point vapor pressure (Pa)
      ρ_ocean =           FT(1.035e3)                         , # Reference density sea water (kg/m^3)
      cp_ocean =          FT(3989.25)                         , # Specific heat sea water (J/kg/K)
      planet_radius =     FT(6.371e6)                         , # Mean planetary radius (m)
      day =               FT(86400)                           , # Length of day (s)
      Omega =             FT(7.2921159e-5)                    , # Ang. velocity planetary rotation (1/s)
      grav =              FT(9.81)                            , # Gravitational acceleration (m/s^2)
      orbit_semimaj =     FT(1*astro_unit(FT))                , # Length of semimajor orbital axis (m)
      TSI =               FT(1362)                            , # Total solar irradiance (W/m^2)
      MSLP =              FT(1.01325e5)                       , # Mean sea level pressure (Pa)
  )
  return PlanetConstants{FT}(;pairs(merge(defaults, kwargs.data))...)

end
