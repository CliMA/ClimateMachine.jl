# NOTE: This file will soon be automatically generated

import CLIMA

struct ParameterSet{FT} <: CLIMA.Parameters.AbstractParameterSet{FT} end

# Properties of dry air
CLIMA.Parameters.Planet.molmass_dryair(ps::ParameterSet{FT}) where {FT} =
    FT(28.97e-3)
CLIMA.Parameters.Planet.R_d(ps::ParameterSet{FT}) where {FT} =
    gas_constant(FT) / molmass_dryair(ps)
CLIMA.Parameters.Planet.kappa_d(ps::ParameterSet{FT}) where {FT} = FT(2 / 7)
CLIMA.Parameters.Planet.cp_d(ps::ParameterSet{FT}) where {FT} =
    R_d(ps) / kappa_d(ps)
CLIMA.Parameters.Planet.cv_d(ps::ParameterSet{FT}) where {FT} =
    cp_d(ps) - R_d(ps)

# Properties of water
CLIMA.Parameters.Planet.ρ_cloud_liq(ps::ParameterSet{FT}) where {FT} = FT(1e3)
CLIMA.Parameters.Planet.ρ_cloud_ice(ps::ParameterSet{FT}) where {FT} = FT(916.7)
CLIMA.Parameters.Planet.molmass_water(ps::ParameterSet{FT}) where {FT} =
    FT(18.01528e-3)
CLIMA.Parameters.Planet.molmass_ratio(ps::ParameterSet{FT}) where {FT} =
    molmass_dryair(ps) / molmass_water(ps)
CLIMA.Parameters.Planet.R_v(ps::ParameterSet{FT}) where {FT} =
    gas_constant(FT) / molmass_water(ps)
CLIMA.Parameters.Planet.cp_v(ps::ParameterSet{FT}) where {FT} = FT(1859)
CLIMA.Parameters.Planet.cp_l(ps::ParameterSet{FT}) where {FT} = FT(4181)
CLIMA.Parameters.Planet.cp_i(ps::ParameterSet{FT}) where {FT} = FT(2100)
CLIMA.Parameters.Planet.cv_v(ps::ParameterSet{FT}) where {FT} =
    cp_v(ps) - R_v(ps)
CLIMA.Parameters.Planet.cv_l(ps::ParameterSet{FT}) where {FT} = cp_l(ps)
CLIMA.Parameters.Planet.cv_i(ps::ParameterSet{FT}) where {FT} = cp_i(ps)
CLIMA.Parameters.Planet.T_freeze(ps::ParameterSet{FT}) where {FT} = FT(273.15)
CLIMA.Parameters.Planet.T_min(ps::ParameterSet{FT}) where {FT} = FT(150.0)
CLIMA.Parameters.Planet.T_max(ps::ParameterSet{FT}) where {FT} = FT(1000.0)
CLIMA.Parameters.Planet.T_icenuc(ps::ParameterSet{FT}) where {FT} = FT(233.00)
CLIMA.Parameters.Planet.T_triple(ps::ParameterSet{FT}) where {FT} = FT(273.16)
CLIMA.Parameters.Planet.T_0(ps::ParameterSet{FT}) where {FT} = T_triple(ps)
CLIMA.Parameters.Planet.LH_v0(ps::ParameterSet{FT}) where {FT} = FT(2.5008e6)
CLIMA.Parameters.Planet.LH_s0(ps::ParameterSet{FT}) where {FT} = FT(2.8344e6)
CLIMA.Parameters.Planet.LH_f0(ps::ParameterSet{FT}) where {FT} =
    LH_s0(ps) - LH_v0(ps)
CLIMA.Parameters.Planet.e_int_v0(ps::ParameterSet{FT}) where {FT} =
    LH_v0(ps) - R_v(ps) * T_0(ps)
CLIMA.Parameters.Planet.e_int_i0(ps::ParameterSet{FT}) where {FT} = LH_f0(ps)
CLIMA.Parameters.Planet.press_triple(ps::ParameterSet{FT}) where {FT} =
    FT(611.657)

# Properties of sea water
CLIMA.Parameters.Planet.ρ_ocean(ps::ParameterSet{FT}) where {FT} = FT(1.035e3)
CLIMA.Parameters.Planet.cp_ocean(ps::ParameterSet{FT}) where {FT} = FT(3989.25)

# Planetary parameters
CLIMA.Parameters.Planet.planet_radius(ps::ParameterSet{FT}) where {FT} =
    FT(6.371e6)
CLIMA.Parameters.Planet.day(ps::ParameterSet{FT}) where {FT} = FT(86400)
CLIMA.Parameters.Planet.Omega(ps::ParameterSet{FT}) where {FT} =
    FT(7.2921159e-5)
CLIMA.Parameters.Planet.grav(ps::ParameterSet{FT}) where {FT} = FT(9.81)
CLIMA.Parameters.Planet.year_anom(ps::ParameterSet{FT}) where {FT} =
    FT(365.26) * day(ps)
CLIMA.Parameters.Planet.orbit_semimaj(ps::ParameterSet{FT}) where {FT} =
    FT(1 * astro_unit(FT))
CLIMA.Parameters.Planet.TSI(ps::ParameterSet{FT}) where {FT} = FT(1362)
CLIMA.Parameters.Planet.MSLP(ps::ParameterSet{FT}) where {FT} = FT(1.01325e5)
