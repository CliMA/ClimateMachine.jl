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
function terminal_velocity_single_drop_coeff(ρ::FT) where {FT<:Real}

    # terminal_vel_of_individual_drop = v_drop_coeff * (g * drop_radius)^(1/2)
    return sqrt(FT(8/3) / C_drag * (ρ_cloud_liq / ρ - FT(1)))
end

"""
    terminal_velocity(q_rai, ρ)

where:
  - `q_rai` - rain water specific humidity
  - `ρ`     - density of air

Returns the mass weighted average rain terminal velocity assuming
Marshall Palmer 1948 distribution of rain drops.
"""
function terminal_velocity(q_rai::FT, ρ::FT) where {FT<:Real}

    v_c = terminal_velocity_single_drop_coeff(ρ)

    # gamma(9/2)
    gamma_9_2 = FT(11.631728396567448)

    lambda::FT = (FT(8) * π * ρ_cloud_liq * MP_n_0 / ρ / q_rai)^FT(1/4)

    return gamma_9_2 * v_c / FT(6) * sqrt(grav / lambda)
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
function conv_q_vap_to_q_liq(q_sat::PhasePartition{FT},
                             q::PhasePartition{FT}) where {FT<:Real}

  if q_sat.ice != FT(0)
    error("1-moment bulk microphysics is not defined for snow/ice")
    #This should be the q_ice tendency due to sublimation/resublimation.
    #src_q_ice = (q_sat.ice - q.ice) / τ_subl_resubl
  end

  return (q_sat.liq - q.liq) / τ_cond_evap
end


"""
    conv_q_liq_to_q_rai_acnv(q_liq)

where:
- `q_liq` - is the liquid water specific humidity

Returns the q_rai tendency due to collisions between cloud droplets
(autoconversion) parametrized following Kessler 1995.
"""
function conv_q_liq_to_q_rai_acnv(q_liq::FT) where {FT<:Real}

  return max(FT(0), q_liq - q_liq_threshold) / τ_acnv
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
function conv_q_liq_to_q_rai_accr(q_liq::FT, q_rai::FT, ρ::FT) where {FT<:Real}

  # terminal_vel_of_individual_drop = v_drop_coeff * drop_radius^(1/2)
  v_c = terminal_velocity_single_drop_coeff(ρ)

  #gamma(7/2)
  gamma_7_2 = FT(3.3233509704478426)

  accr_coeff::FT = gamma_7_2 * FT(8)^FT(-7/8) * π^FT(1/8) * v_c * E_col *
                   (ρ / ρ_cloud_liq)^FT(7/8)

  return accr_coeff * FT(MP_n_0)^FT(1/8) * sqrt(FT(grav)) *
         q_liq * q_rai^FT(7/8)
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
function conv_q_rai_to_q_vap(qr::FT, q::PhasePartition{FT},
                             T::FT, p::FT, ρ::FT) where {FT<:Real}

  qv_sat = q_vap_saturation(T, ρ, q)
  q_v = q.tot - q.liq - q.ice
  S = q_v/qv_sat - 1

  L = latent_heat_vapor(T)
  p_vs = saturation_vapor_pressure(T, Liquid())
  G::FT = FT(1) / (
            L / K_therm / T * (L / R_v / T - FT(1)) + R_v * T / D_vapor / p_vs
          )

  # gamma(11/4)
  gamma_11_4 = FT(1.6083594219855457)
  N_Sc::FT = ν_air / D_vapor
  v_c = terminal_velocity_single_drop_coeff(ρ)

  av::FT = sqrt(2 * π) * a_vent * sqrt(ρ / ρ_cloud_liq)
  bv::FT = FT(2)^FT(7/16) * gamma_11_4 * π^FT(5/16) * b_vent * (N_Sc)^FT(1/3) *
       sqrt(v_c) * (ρ / ρ_cloud_liq)^FT(11/16)

  F::FT = av * sqrt(qr) +
          bv * grav^FT(1/4) / (MP_n_0)^FT(3/16) / sqrt(ν_air) * qr^FT(11/16)

  return S * F * G * sqrt(MP_n_0) / ρ
end

end #module Microphysics.jl
