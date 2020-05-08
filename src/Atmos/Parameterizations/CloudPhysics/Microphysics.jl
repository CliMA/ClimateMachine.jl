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

using ClimateMachine.MoistThermodynamics
using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav
using CLIMAParameters.Atmos.Microphysics:
    C_drag,
    MP_n_0,
    K_therm,
    τ_cond_evap,
    τ_sub_resub,
    τ_acnv,
    q_liq_threshold,
    E_col,
    D_vapor,
    ν_air,
    N_Sc,
    a_vent,
    b_vent

const APS = AbstractParameterSet

# rain fall speed
export terminal_velocity_single_drop_coeff
export terminal_velocity

# rates of conversion between microphysics categories
export conv_q_vap_to_q_liq
export conv_q_vap_to_q_ice
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
function terminal_velocity_single_drop_coeff(
    param_set::APS,
    ρ::FT,
) where {FT <: Real}
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
    _C_drag::FT = C_drag(param_set)

    # terminal_vel_of_individual_drop = v_drop_coeff * (g * drop_radius)^(1/2)
    return sqrt(FT(8 / 3) / _C_drag * (_ρ_cloud_liq / ρ - FT(1)))
end

"""
    terminal_velocity(q_rai, ρ)

where:
  - `q_rai` - rain water specific humidity
  - `ρ`     - density of air

Returns the mass weighted average rain terminal velocity assuming
Marshall Palmer 1948 distribution of rain drops.
"""
function terminal_velocity(param_set::APS, q_rai::FT, ρ::FT) where {FT <: Real}

    rain_w = FT(0)

    if q_rai > FT(0)
        v_c = terminal_velocity_single_drop_coeff(param_set, ρ)
        # gamma(9/2)
        gamma_9_2 = FT(11.631728396567448)
        _grav::FT = grav(param_set)
        _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
        _MP_n_0::FT = MP_n_0(param_set)

        lambda::FT = (FT(8) * π * _ρ_cloud_liq * _MP_n_0 / ρ / q_rai)^FT(1 / 4)
        rain_w = gamma_9_2 * v_c / FT(6) * sqrt(_grav / lambda)
    end

    return rain_w
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
function conv_q_vap_to_q_liq(
    param_set::APS,
    q_sat::PhasePartition{FT},
    q::PhasePartition{FT},
) where {FT <: Real}
    _τ_cond_evap::FT = τ_cond_evap(param_set)
    return (q_sat.liq - q.liq) / _τ_cond_evap
end

"""
    conv_q_vap_to_q_ice(q_sat, q)

where:
- `q_sat` - PhasePartition at equilibrium
- `q`     - current PhasePartition

Returns the q_ice tendency due to condensation/evaporation.
The tendency is obtained assuming a relaxation to equilibrium with
constant timescale.
"""
function conv_q_vap_to_q_ice(
    param_set::APS,
    q_sat::PhasePartition{FT},
    q::PhasePartition{FT},
) where {FT <: Real}

    _τ_sub_resub::FT = τ_sub_resub(param_set)
    return (q_sat.ice - q.ice) / _τ_sub_resub
end

"""
    conv_q_liq_to_q_rai_acnv(q_liq)

where:
- `q_liq` - is the liquid water specific humidity

Returns the q_rai tendency due to collisions between cloud droplets
(autoconversion) parametrized following Kessler 1995.
"""
function conv_q_liq_to_q_rai_acnv(param_set::APS, q_liq::FT) where {FT <: Real}

    _τ_acnv::FT = τ_acnv(param_set)
    _q_liq_threshold::FT = q_liq_threshold(param_set)
    return max(FT(0), q_liq - _q_liq_threshold) / _τ_acnv
end


"""
    conv_q_liq_to_q_rai_accr(param_set, q_liq, q_rai, ρ)

where:
- `param_set` - is an `AbstractParameterSet`
- `q_liq` - is the liquid water specific humidity
- `q_rai` - is the rain water specific humidity
- `ρ` - is the density of air

Returns the q_rai tendency due to collisions between cloud droplets
and rain drops (accretion) parametrized following Kessler 1995.
"""
function conv_q_liq_to_q_rai_accr(
    param_set::APS,
    q_liq::FT,
    q_rai::FT,
    ρ::FT,
) where {FT <: Real}

    accr_rate = FT(0)
    _MP_n_0::FT = MP_n_0(param_set)
    _E_col::FT = E_col(param_set)
    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
    _grav::FT = grav(param_set)

    if (q_rai > FT(0) && q_liq > FT(0))
        # terminal_vel_of_individual_drop = v_drop_coeff * drop_radius^(1/2)
        v_c = terminal_velocity_single_drop_coeff(param_set, ρ)

        #gamma(7/2)
        gamma_7_2 = FT(3.3233509704478426)

        accr_coeff::FT =
            gamma_7_2 *
            FT(8)^FT(-7 / 8) *
            π^FT(1 / 8) *
            v_c *
            _E_col *
            (ρ / _ρ_cloud_liq)^FT(7 / 8)

        accr_rate =
            accr_coeff *
            _MP_n_0^FT(1 / 8) *
            sqrt(_grav) *
            q_liq *
            q_rai^FT(7 / 8)
    end
    return accr_rate
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
function conv_q_rai_to_q_vap(
    param_set::APS,
    qr::FT,
    q::PhasePartition{FT},
    T::FT,
    p::FT,
    ρ::FT,
) where {FT <: Real}

    evap_rate = FT(0)
    _R_v = R_v(param_set)
    _D_vapor = D_vapor(param_set)
    _ν_air = ν_air(param_set)
    _grav = grav(param_set)
    _ρ_cloud_liq = ρ_cloud_liq(param_set)
    _N_Sc::FT = N_Sc(param_set)
    _b_vent::FT = b_vent(param_set)
    _a_vent::FT = a_vent(param_set)
    _K_therm::FT = K_therm(param_set)
    _MP_n_0::FT = MP_n_0(param_set)

    if qr > FT(0)
        qv_sat = q_vap_saturation(param_set, T, ρ, q)
        q_v = q.tot - q.liq - q.ice
        S = q_v / qv_sat - FT(1)

        L = latent_heat_vapor(param_set, T)
        p_vs = saturation_vapor_pressure(param_set, T, Liquid())
        G::FT =
            FT(1) / (
                L / _K_therm / T * (L / _R_v / T - FT(1)) +
                _R_v * T / _D_vapor / p_vs
            )

        # gamma(11/4)
        gamma_11_4 = FT(1.6083594219855457)
        v_c = terminal_velocity_single_drop_coeff(param_set, ρ)

        av::FT = sqrt(2 * π) * _a_vent * sqrt(ρ / _ρ_cloud_liq)
        bv::FT =
            FT(2)^FT(7 / 16) *
            gamma_11_4 *
            π^FT(5 / 16) *
            _b_vent *
            (_N_Sc)^FT(1 / 3) *
            sqrt(v_c) *
            (ρ / _ρ_cloud_liq)^FT(11 / 16)

        F::FT =
            av * sqrt(qr) +
            bv * _grav^FT(1 / 4) / (_MP_n_0)^FT(3 / 16) / sqrt(_ν_air) *
            qr^FT(11 / 16)

        evap_rate = min(FT(0), S * F * G * sqrt(_MP_n_0) / ρ)
    end
    return evap_rate
end
end #module Microphysics.jl
