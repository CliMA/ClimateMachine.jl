# NOTE: This file will soon be automatically generated

import CLIMA

struct ParameterSet <: CLIMA.Parameters.AbstractParameterSet end

# Properties of dry air
CLIMA.Parameters.Planet.molmass_dryair(ps::ParameterSet) = 28.97e-3
CLIMA.Parameters.Planet.R_d(ps::ParameterSet) =
    gas_constant() / molmass_dryair(ps)
CLIMA.Parameters.Planet.kappa_d(ps::ParameterSet) = 2 / 7
CLIMA.Parameters.Planet.cp_d(ps::ParameterSet) = R_d(ps) / kappa_d(ps)
CLIMA.Parameters.Planet.cv_d(ps::ParameterSet) = cp_d(ps) - R_d(ps)

# Properties of water
CLIMA.Parameters.Planet.ρ_cloud_liq(ps::ParameterSet) = 1e3
CLIMA.Parameters.Planet.ρ_cloud_ice(ps::ParameterSet) = 916.7
CLIMA.Parameters.Planet.molmass_water(ps::ParameterSet) = 18.01528e-3
CLIMA.Parameters.Planet.molmass_ratio(ps::ParameterSet) =
    molmass_dryair(ps) / molmass_water(ps)
CLIMA.Parameters.Planet.R_v(ps::ParameterSet) =
    gas_constant() / molmass_water(ps)
CLIMA.Parameters.Planet.cp_v(ps::ParameterSet) = 1859
CLIMA.Parameters.Planet.cp_l(ps::ParameterSet) = 4181
CLIMA.Parameters.Planet.cp_i(ps::ParameterSet) = 2100
CLIMA.Parameters.Planet.cv_v(ps::ParameterSet) = cp_v(ps) - R_v(ps)
CLIMA.Parameters.Planet.cv_l(ps::ParameterSet) = cp_l(ps)
CLIMA.Parameters.Planet.cv_i(ps::ParameterSet) = cp_i(ps)
CLIMA.Parameters.Planet.T_freeze(ps::ParameterSet) = 273.15
CLIMA.Parameters.Planet.T_min(ps::ParameterSet) = 150.0
CLIMA.Parameters.Planet.T_max(ps::ParameterSet) = 1000.0
CLIMA.Parameters.Planet.T_icenuc(ps::ParameterSet) = 233.00
CLIMA.Parameters.Planet.T_triple(ps::ParameterSet) = 273.16
CLIMA.Parameters.Planet.T_0(ps::ParameterSet) = T_triple(ps)
CLIMA.Parameters.Planet.LH_v0(ps::ParameterSet) = 2.5008e6
CLIMA.Parameters.Planet.LH_s0(ps::ParameterSet) = 2.8344e6
CLIMA.Parameters.Planet.LH_f0(ps::ParameterSet) = LH_s0(ps) - LH_v0(ps)
CLIMA.Parameters.Planet.e_int_v0(ps::ParameterSet) =
    LH_v0(ps) - R_v(ps) * T_0(ps)
CLIMA.Parameters.Planet.e_int_i0(ps::ParameterSet) = LH_f0(ps)
CLIMA.Parameters.Planet.press_triple(ps::ParameterSet) = 611.657

# Properties of sea water
CLIMA.Parameters.Planet.ρ_ocean(ps::ParameterSet) = 1.035e3
CLIMA.Parameters.Planet.cp_ocean(ps::ParameterSet) = 3989.25

# Planetary parameters
CLIMA.Parameters.Planet.planet_radius(ps::ParameterSet) = 6.371e6
CLIMA.Parameters.Planet.day(ps::ParameterSet) = 86400
CLIMA.Parameters.Planet.Omega(ps::ParameterSet) = 7.2921159e-5
CLIMA.Parameters.Planet.grav(ps::ParameterSet) = 9.81
CLIMA.Parameters.Planet.year_anom(ps::ParameterSet) = 365.26 * day(ps)
CLIMA.Parameters.Planet.orbit_semimaj(ps::ParameterSet) = 1 * astro_unit()
CLIMA.Parameters.Planet.TSI(ps::ParameterSet) = 1362
CLIMA.Parameters.Planet.MSLP(ps::ParameterSet) = 1.01325e5
