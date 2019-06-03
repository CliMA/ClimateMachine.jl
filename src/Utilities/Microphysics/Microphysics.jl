"""
    one-moment bulk Microphysics scheme

Microphysics parameterization based on Kessler_1995:
  - condensation/evaporation and sublimation/resublimation
    (as relaxation to equilibrium)
  - autoconversion
  - TODO: accretion
  - rain evaporation
  - rain terminal velocity
"""

#@article{Marshall_Palmer_1948,
#author = {Marshall, J. S. and Palmer, W. Mc K.},
#title = {The distribution of raindrops with size},
#journal = {Journal of Meteorology},
#volume = {5},
#number = {4},
#pages = {165-166},
#year = {1948},
#doi = {10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2}}

#@article{Ogura_and_Takahashi_1971,
#author = {Oqura, Yoshimitsu and Takahashi, Tsutomu},
#title = {Numerical simulation of the life cycle of a thunderstorm cell},
#journal = {Monthly Weather Review},
#volume = {99},
#number = {12},
#pages = {895-911},
#year = {1971},
#doi = {10.1175/1520-0493(1971)099<0895:NSOTLC>2.3.CO;2}}

#@article{Kessler_1995,
#author = "E. Kessler",
#title = "On the continuity and distribution of water substance in atmospheric circulations",
#journal = "Atmospheric Research",
#volume = "38",
#number = "1",
#pages = "109 - 145",
#year = "1995",
#doi = "https://doi.org/10.1016/0169-8095(94)00090-Z"}

module Microphysics

using ..MoistThermodynamics
using ..PlanetParameters
using ..MicrophysicsParameters

# rain fall speed
export terminal_velocity

# rates of conversion between microphysics categories
export conv_q_vap_to_q_liq
export conv_q_liq_to_q_rai_acnv
export conv_q_liq_to_q_rai_accr
export conv_q_rai_to_q_vap


"""
    terminal_velocity(q_rai, ρ)

where:
  - `q_rai` - rain water specific humidity
  - `ρ`     - density of air

Return mass weighted average rain terminal velocity assuming
Marshall_Palmer_1948 distribution of rain drops. It's derived following
Ogura_and_Takahashi_1971 but without substituting empirical relations.
"""
function terminal_velocity(q_rai::DT, ρ::DT) where {DT<:Real}

    # TODO - should it be multiplied by ρ/ρ_ground?
    # TODO - check how it compares with SG 1996
    #        vel = DT(14.34) * ρ_ground^DT(0.5) * ρ^-DT(0.3654) * rr^DT(0.1346)

    vel::DT = 0

    # terminal_vel_of_individual_drop = v_drop_coeff * drop_radius^(1/2)
    v_drop_coeff::DT = sqrt(DT(8/3) * grav/C_drag * (dens_liquid / ρ - DT(1)))

    # gamma(9/2)
    gamma_9_2::DT = 11.63

    v_coeff = gamma_9_2 * v_drop_coeff / DT(6) / sqrt(DT(2)) *
              (DT(1) / π / MP_n_0)^(DT(1/8))

    if (qr > DT(0)) # TODO - assert positive definite elsewhere
      vel = v_coeff * (ρ / dens_liquid)^(DT(1/8)) * q_rain
    end

    return vel
end


"""
    conv_q_vap_to_q_liq_ice(q_sat::PhasePartition, q::PhasePartition)

where:
- `q_sat` - PhasePartition at equilibrium saturation
- `q`     - current PhasePartition

Returns the q_liq tendency due to condensation/evaporation
and q_ice tendency due to sublimation/resublimation.
The tendencies are obtained assuming a relaxation to equilibrium with
constant timescales and the timescales are defined in
MicrophysicsParameters module.
"""
function conv_q_vap_to_q_liq_ice(q_sat::PhasePartition,
                                 q::PhasePartition,
                                ) where {DT<:Real}

  src_q_liq = (q_sat.liq - q.liq) / τ_cond_evap
  src_q_ice = (q_sat.ice - q.ice) / τ_subl_resubl

  return (src_q_liq, src_q_ice)
end


"""
    conv_q_liq_to_q_rai_acnv(q_liq)

Returns the q_rai tendency due to collisions between cloud droplets
(autoconversion) parametrized following Kessler_1995.
The timescale and autoconversion threshold q_liq_0 are parameters and are
defined in MicrophysicsParameters module.
"""
function conv_q_liq_to_q_rai_acnv(q_liq::DT) where {DT<:Real}

  src_q_rai = max(DT(0), q_liq - q_liq_threshold) / τ_acnv

  return src_q_rai
end


"""
    conv_q_liq_to_q_rai_accr(q_liq)

Returns the q_rai tendency due to collisions between cloud droplets
and rain drops (accretion) parametrized following Kessler_1995
and Ogura_and_Takahashi_1971.
"""
function conv_q_liq_to_q_rai_accr(q_liq::DT, q_rai::DT) where {DT<:Real}

  accr_coeff = E * (π * n_0)^DT(1/8) * (ρ/ρ_w)^DT(7/8) * ...TODO

  src_q_rai = accr_coeff * q_liq * q_rai^(DT(7/8))

  return src_q_rai
end





"""
    q2r(q_, qt)

Convert specific humidity to mixing ratio
"""
function q2r(q_::DT, qt::DT) where {DT<:Real}
    return q_ / (DT(1) - qt)
end

"""
    qr2qv(qt, PhasePartition, T, ρ, p)

Return rain evaporation rate.
TODO - add citation
"""
function qr2qv(q::PhasePartition, T::DT, ρ::DT, p::DT, qr::DT) where {DT<:Real}

  ret::DT = 0

  if (qr > 0) # TODO - assert positive definite elsewhere

    qv_sat = saturation_shum(T, ρ, q)
    qv = q.tot - q.liq - q.ice

    rr = q2r(qr, q.tot)
    rv = q2r(qv, q.tot)
    rv_sat = q2r(qv_sat, q.tot)

    # ventilation factor
    C = DT(1.6) + DT(124.9) * (DT(1e-3) * ρ * rr)^DT(0.2046)

    ret = (DT(1) - q.tot) * (DT(1) - rv/rv_sat) * C *
          (DT(1e-3) * ρ * rr)^DT(0.525) /
          ρ / (DT(540) + DT(2.55) * DT(1e5) / (p * rv_sat))
  end

  return ret

end

end #module Microphysics.jl
