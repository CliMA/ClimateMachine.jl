"""
    PlanetParameters

Module containing the planetary constants.
"""
module PlanetParameters

export @parameter

# Physical constants
export gas_constant, light_speed, h_Planck, k_Boltzmann, Stefan, astro_unit

# Properties of dry air
export molmass_air, R_d, kappa_d, cp_d

# Properties of water
export dens_liquid, molmass_water, R_v, cp_v, cp_l, cp_i, Tfreeze, T0, L_v0,
       L_s0, L_f0, sat_vapor_press_0

# Properties of sea water
export dens_ocean, cp_ocean

# Planetary parameters
export planet_radius, day, Omega, grav, year_anom, orbit_semimaj, TSI, mslp

include("planet_constants.jl")

end
