import CLIMA

# Properties of dry air
CLIMA.Parameters.Planet.molmass_dryair(ps::EarthParameterSet) = 28.97e-3
CLIMA.Parameters.Planet.R_d(ps::EarthParameterSet) =
    gas_constant() / molmass_dryair(ps)
CLIMA.Parameters.Planet.kappa_d(ps::EarthParameterSet) = 2 / 7
CLIMA.Parameters.Planet.cp_d(ps::EarthParameterSet) = R_d(ps) / kappa_d(ps)
CLIMA.Parameters.Planet.cv_d(ps::EarthParameterSet) = cp_d(ps) - R_d(ps)

# Properties of water
CLIMA.Parameters.Planet.ρ_cloud_liq(ps::EarthParameterSet) = 1e3
CLIMA.Parameters.Planet.ρ_cloud_ice(ps::EarthParameterSet) = 916.7
CLIMA.Parameters.Planet.molmass_water(ps::EarthParameterSet) = 18.01528e-3
CLIMA.Parameters.Planet.molmass_ratio(ps::EarthParameterSet) =
    molmass_dryair(ps) / molmass_water(ps)
CLIMA.Parameters.Planet.R_v(ps::EarthParameterSet) =
    gas_constant() / molmass_water(ps)
CLIMA.Parameters.Planet.cp_v(ps::EarthParameterSet) = 1859
CLIMA.Parameters.Planet.cp_l(ps::EarthParameterSet) = 4181
CLIMA.Parameters.Planet.cp_i(ps::EarthParameterSet) = 2100
CLIMA.Parameters.Planet.cv_v(ps::EarthParameterSet) = cp_v(ps) - R_v(ps)
CLIMA.Parameters.Planet.cv_l(ps::EarthParameterSet) = cp_l(ps)
CLIMA.Parameters.Planet.cv_i(ps::EarthParameterSet) = cp_i(ps)
CLIMA.Parameters.Planet.T_freeze(ps::EarthParameterSet) = 273.15
CLIMA.Parameters.Planet.T_min(ps::EarthParameterSet) = 150.0
CLIMA.Parameters.Planet.T_max(ps::EarthParameterSet) = 1000.0
CLIMA.Parameters.Planet.T_icenuc(ps::EarthParameterSet) = 233.00
CLIMA.Parameters.Planet.T_triple(ps::EarthParameterSet) = 273.16
CLIMA.Parameters.Planet.T_0(ps::EarthParameterSet) = T_triple(ps)
CLIMA.Parameters.Planet.LH_v0(ps::EarthParameterSet) = 2.5008e6
CLIMA.Parameters.Planet.LH_s0(ps::EarthParameterSet) = 2.8344e6
CLIMA.Parameters.Planet.LH_f0(ps::EarthParameterSet) = LH_s0(ps) - LH_v0(ps)
CLIMA.Parameters.Planet.e_int_v0(ps::EarthParameterSet) =
    LH_v0(ps) - R_v(ps) * T_0(ps)
CLIMA.Parameters.Planet.e_int_i0(ps::EarthParameterSet) = LH_f0(ps)
CLIMA.Parameters.Planet.press_triple(ps::EarthParameterSet) = 611.657

# Properties of sea water
CLIMA.Parameters.Planet.ρ_ocean(ps::EarthParameterSet) = 1.035e3
CLIMA.Parameters.Planet.cp_ocean(ps::EarthParameterSet) = 3989.25

# Planetary parameters
CLIMA.Parameters.Planet.planet_radius(ps::EarthParameterSet) = 6.371e6
CLIMA.Parameters.Planet.day(ps::EarthParameterSet) = 86400
CLIMA.Parameters.Planet.Omega(ps::EarthParameterSet) = 7.2921159e-5
CLIMA.Parameters.Planet.grav(ps::EarthParameterSet) = 9.81
CLIMA.Parameters.Planet.year_anom(ps::EarthParameterSet) = 365.26 * day(ps)
CLIMA.Parameters.Planet.orbit_semimaj(ps::EarthParameterSet) = 1 * astro_unit()
CLIMA.Parameters.Planet.TSI(ps::EarthParameterSet) = 1362
CLIMA.Parameters.Planet.MSLP(ps::EarthParameterSet) = 1.01325e5
