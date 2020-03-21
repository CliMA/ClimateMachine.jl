module Planet

export molmass_dryair,
    R_d,
    kappa_d,
    cp_d,
    cv_d,
    ρ_cloud_liq,
    ρ_cloud_ice,
    molmass_water,
    molmass_ratio,
    R_v,
    cp_v,
    cp_l,
    cp_i,
    cv_v,
    cv_l,
    cv_i,
    T_freeze,
    T_min,
    T_max,
    T_icenuc,
    T_triple,
    T_0,
    LH_v0,
    LH_s0,
    LH_f0,
    e_int_v0,
    e_int_i0,
    press_triple,
    ρ_ocean,
    cp_ocean,
    planet_radius,
    day,
    Omega,
    grav,
    year_anom,
    orbit_semimaj,
    TSI,
    MSLP

# Properties of dry air
""" Molecular weight dry air (kg/mol) """
function molmass_dryair end
""" Gas constant dry air (J/kg/K) """
function R_d end
""" Adiabatic exponent dry air """
function kappa_d end
""" Isobaric specific heat dry air """
function cp_d end
""" Isochoric specific heat dry air """
function cv_d end

# Properties of water
""" Density of liquid water (kg/m^3) """
function ρ_cloud_liq end
""" Density of ice water (kg/m^3) """
function ρ_cloud_ice end
""" Molecular weight (kg/mol) """
function molmass_water end
""" Molar mass ratio dry air/water """
function molmass_ratio end
""" Gas constant water vapor (J/kg/K) """
function R_v end
""" Isobaric specific heat vapor (J/kg/K) """
function cp_v end
""" Isobaric specific heat liquid (J/kg/K) """
function cp_l end
""" Isobaric specific heat ice (J/kg/K) """
function cp_i end
""" Isochoric specific heat vapor (J/kg/K) """
function cv_v end
""" Isochoric specific heat liquid (J/kg/K) """
function cv_l end
""" Isochoric specific heat ice (J/kg/K) """
function cv_i end
""" Freezing point temperature (K) """
function T_freeze end
""" Minimum temperature guess in saturation adjustment (K) """
function T_min end
""" Maximum temperature guess in saturation adjustment (K) """
function T_max end
""" Homogeneous nucleation temperature (K) """
function T_icenuc end
""" Triple point temperature (K) """
function T_triple end
""" Reference temperature (K) """
function T_0 end
""" Latent heat vaporization at T_0 (J/kg) """
function LH_v0 end
""" Latent heat sublimation at T_0 (J/kg) """
function LH_s0 end
""" Latent heat of fusion at T_0 (J/kg) """
function LH_f0 end
""" Specific internal energy of vapor at T_0 (J/kg) """
function e_int_v0 end
""" Specific internal energy of ice at T_0 (J/kg) """
function e_int_i0 end
""" Triple point vapor pressure (Pa) """
function press_triple end

# Properties of sea water
""" Reference density sea water (kg/m^3) """
function ρ_ocean end
""" Specific heat sea water (J/kg/K) """
function cp_ocean end

# Planetary parameters
""" Mean planetary radius (m) """
function planet_radius end
""" Length of day (s) """
function day end
""" Ang. velocity planetary rotation (1/s) """
function Omega end
""" Gravitational acceleration (m/s^2) """
function grav end
""" Length of anomalistic year (s) """
function year_anom end
""" ngth of semimajor orbital axis (m) """
function orbit_semimaj end
""" Total solar irradiance (W/m^2) """
function TSI end
""" Mean sea level pressure (Pa) """
function MSLP end

end
