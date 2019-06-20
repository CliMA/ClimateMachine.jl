"""
    one-moment bulk Microphysics scheme

Microphysics parameterization based on the ideas of Kessler_1995:
  - condensation/evaporation as relaxation to equilibrium
  - autoconversion
  - accretion
  - rain evaporation
  - rain terminal velocity
"""

module Microphysics

using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using ..MicrophysicsParameters

# rain fall speed
export terminal_velocity_single_drop_coeff
export terminal_velocity

# rates of conversion between microphysics categories
export conv_q_vap_to_q_liq
export conv_q_liq_to_q_rai_acnv
export conv_q_liq_to_q_rai_accr
export conv_q_rai_to_q_vap

"""
    terminal_velocity_single_drop_coeff(ρ)

where:
  - `ρ` is the density of air

Returns the proportionality coefficient between terminal velocity of an
individual water drop and the square root of its radius * g.
"""
function terminal_velocity_single_drop_coeff(ρ::DT) where {DT<:Real}

    # terminal_vel_of_individual_drop = v_drop_coeff * (g * drop_radius)^(1/2)
    v_c::DT = sqrt(DT(8/3) / C_drag * (dens_liquid / ρ - DT(1)))

    return v_c
end

"""
    terminal_velocity(q_rai, ρ)

where:
  - `q_rai` - rain water specific humidity
  - `ρ`     - density of air

Returns the mass weighted average rain terminal velocity assuming
Marshall Palmer 1948 distribution of rain drops.
"""
function terminal_velocity(q_rai::DT, ρ::DT) where {DT<:Real}

    v_c = terminal_velocity_single_drop_coeff(ρ)

    # gamma(9/2)
    gamma_9_2::DT = DT(11.63)

    lambda::DT = (DT(8) * π * dens_liquid * MP_n_0 / ρ / q_rai)^DT(1/4)

    vel = gamma_9_2 * v_c / DT(6) * sqrt(grav / lambda)

    return vel
end


"""
    conv_q_vap_to_q_liq(q_sat, q)

where:
- `q_sat` - PhasePartition at equilibrium
- `q`     - current PhasePartition

Returns the q_liq tendency due to condensation/evaporation.
The tendency is obtained assuming a relaxation to equilibrium with
constant timescale.
"""
function conv_q_vap_to_q_liq(q_sat::PhasePartition, q::PhasePartition)

  DT = eltype(q.tot)

  src_q_liq::DT = (q_sat.liq - q.liq) / τ_cond_evap

  if q_sat.ice != DT(0)
    @show("1-moment bulk microphysics is not defined for snow/ice")
    #This should be the q_ice tendency due to sublimation/resublimation.
    #src_q_ice = (q_sat.ice - q.ice) / τ_subl_resubl
  end

  return src_q_liq

end


"""
    conv_q_liq_to_q_rai_acnv(q_liq)

where:
- `q_liq` - is the liquid water specific humidity

Returns the q_rai tendency due to collisions between cloud droplets
(autoconversion) parametrized following Kessler 1995.
"""
function conv_q_liq_to_q_rai_acnv(q_liq::DT) where {DT<:Real}

  src_q_rai::DT = max(DT(0), q_liq - q_liq_threshold) / τ_acnv

  return src_q_rai
end


"""
    conv_q_liq_to_q_rai_accr(q_liq, q_rai, ρ)

where:
- `q_liq` - is the liquid water specific humidity
- `q_rai` - is the rain water specific humidity
- `ρ` - is the density of air

Returns the q_rai tendency due to collisions between cloud droplets
and rain drops (accretion) parametrized following Kessler 1995.
"""
function conv_q_liq_to_q_rai_accr(q_liq::DT, q_rai::DT, ρ::DT) where {DT<:Real}

  # terminal_vel_of_individual_drop = v_drop_coeff * drop_radius^(1/2)
  v_c::DT = terminal_velocity_single_drop_coeff(ρ)

  #gamma(7/2)
  gamma_7_2::DT = DT(3.32)

  accr_coeff::DT = gamma_7_2 * DT(8)^DT(-7/8) * π^DT(1/8) * v_c * E_col *
                   (ρ / dens_liquid)^DT(7/8)

  src_q_rai::DT = accr_coeff * MP_n_0^DT(1/8) * sqrt(grav) *
                  q_liq * q_rai^DT(7/8)

  return src_q_rai
end

"""
    conv_q_rai_to_q_vap(q_rai, q, T, p, ρ)

where:
 - q_rai - rain water specific humidity
 - q - current PhasePartition
 - T - temperature
 - p - pressure
 - ρ - air density

Returns the q_rai tendency due to rain evaporation. Parameterized following
Smolarkiewicz and Grabowski 1996.
"""
function conv_q_rai_to_q_vap(qr::DT, q::PhasePartition,
                             T::DT, p::DT, ρ::DT) where {DT<:Real}

  qv_sat = saturation_shum(T, ρ, q)
  q_v::DT = q.tot - q.liq - q.ice
  S::DT = q_v/qv_sat - DT(1)

  L::DT = latent_heat_vapor(T)
  p_vs::DT = saturation_vapor_pressure(T, Liquid())
  G::DT = DT(1) / (
            L / K_therm / T * (L / R_v / T - DT(1)) + R_v * T / D_vapor / p_vs
          )

  # gamma(11/4)
  gamma_11_4::DT = DT(1.61)
  N_Sc::DT = ν_air / D_vapor
  v_c::DT = terminal_velocity_single_drop_coeff(ρ)

  av = sqrt(2 * π) * a_vent * sqrt(ρ / dens_liquid)
  bv = DT(2)^DT(7/16) * gamma_11_4 * π^DT(5/16) * b_vent * (N_Sc)^DT(1/3) *
       sqrt(v_c) * (ρ / dens_liquid)^DT(11/16)

  F::DT = av * sqrt(qr) +
          bv * grav^DT(1/4) / (MP_n_0)^DT(3/16) / sqrt(ν_air) * qr^DT(11/16)

  ret = S * F * G * sqrt(MP_n_0) / ρ

  return ret

end

end #module Microphysics.jl
