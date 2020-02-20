function PlanetConstants{FT}(::Type{EarthConstants}; kwargs...) where {FT}
  settings = (; dummy=1) # cannot apply merge to (; nothing)

  @append_nt settings k_Karman =          FT(0.4)                             # Von Karman constant (1)
  @append_nt settings molmass_dryair =    FT(28.97e-3)                        # Molecular weight dry air (kg/mol)
  @append_nt settings R_d =               FT(gas_constant(FT)/molmass_dryair) # Gas constant dry air (J/kg/K)
  @append_nt settings kappa_d =           FT(2//7)                            # Adiabatic exponent dry air
  @append_nt settings cp_d =              FT(R_d/kappa_d)                     # Isobaric specific heat dry air
  @append_nt settings cv_d =              FT(cp_d - R_d)                      # Isochoric specific heat dry air
  @append_nt settings ρ_liq =             FT(1e3)                             # Density of water (kg/m^3)
  @append_nt settings ρ_cloud_liq =       FT(ρ_liq)                           # Density of liquid water (kg/m^3)
  @append_nt settings ρ_ice =             FT(916.7)                           # Density of ice (kg/m^3)
  @append_nt settings ρ_cloud_ice =       FT(ρ_ice)                           # Density of water ice (kg/m^3)
  @append_nt settings molmass_water =     FT(18.01528e-3)                     # Molecular weight (kg/mol)
  @append_nt settings molmass_ratio =     FT(molmass_dryair/molmass_water)    # Molar mass ratio dry air/water
  @append_nt settings R_v =               FT(gas_constant(FT)/molmass_water)  # Gas constant water vapor (J/kg/K)
  @append_nt settings cp_v =              FT(1859)                            # Isobaric specific heat vapor (J/kg/K)
  @append_nt settings cp_l =              FT(4181)                            # Isobaric specific heat liquid (J/kg/K)
  @append_nt settings cp_i =              FT(2100)                            # Isobaric specific heat ice (J/kg/K)
  @append_nt settings cv_v =              FT(cp_v - R_v)                      # Isochoric specific heat vapor (J/kg/K)
  @append_nt settings cv_l =              FT(cp_l)                            # Isochoric specific heat liquid (J/kg/K)
  @append_nt settings cv_i =              FT(cp_i)                            # Isochoric specific heat ice (J/kg/K)
  @append_nt settings T_freeze =          FT(273.15)                          # Freezing point temperature (K)
  @append_nt settings T_min =             FT(150.0)                           # Minimum temperature guess in saturation adjustment (K)
  @append_nt settings T_max =             FT(1000.0)                          # Maximum temperature guess in saturation adjustment (K)
  @append_nt settings T_icenuc =          FT(233.00)                          # Homogeneous nucleation temperature (K)
  @append_nt settings T_triple =          FT(273.16)                          # Triple point temperature (K)
  @append_nt settings T_0 =               FT(T_triple)                        # Reference temperature (K)
  @append_nt settings LH_v0 =             FT(2.5008e6)                        # Latent heat vaporization at T_0 (J/kg)
  @append_nt settings LH_s0 =             FT(2.8344e6)                        # Latent heat sublimation at T_0 (J/kg)
  @append_nt settings LH_f0 =             FT(LH_s0 - LH_v0)                   # Latent heat of fusion at T_0 (J/kg)
  @append_nt settings e_int_v0 =          FT(LH_v0 - R_v*T_0)                 # Specific internal energy of vapor at T_0 (J/kg)
  @append_nt settings e_int_i0 =          FT(LH_f0)                           # Specific internal energy of ice at T_0 (J/kg)
  @append_nt settings press_triple =      FT(611.657)                         # Triple point vapor pressure (Pa)
  @append_nt settings ρ_ocean =           FT(1.035e3)                         # Reference density sea water (kg/m^3)
  @append_nt settings cp_ocean =          FT(3989.25)                         # Specific heat sea water (J/kg/K)
  @append_nt settings planet_radius =     FT(6.371e6)                         # Mean planetary radius (m)
  @append_nt settings day =               FT(86400)                           # Length of day (s)
  @append_nt settings Omega =             FT(7.2921159e-5)                    # Ang. velocity planetary rotation (1/s)
  @append_nt settings grav =              FT(9.81)                            # Gravitational acceleration (m/s^2)
  @append_nt settings year_anom =         FT(365.26*day)                      # Length of anomalistic year (s)
  @append_nt settings orbit_semimaj =     FT(1*astro_unit(FT))                # Length of semimajor orbital axis (m)
  @append_nt settings TSI =               FT(1362)                            # Total solar irradiance (W/m^2)
  @append_nt settings MSLP =              FT(1.01325e5)                       # Mean sea level pressure (Pa)

  settings = merge(settings, kwargs)
  settings = Base.structdiff(settings, (;dummy=1)) # remove dummy
  return PlanetConstants{FT}(;settings...)
end
