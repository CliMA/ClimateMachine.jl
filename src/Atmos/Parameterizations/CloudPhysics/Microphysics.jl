"""
    One-moment bulk microphysics scheme, which includes:

  - terminal velocity of precipitation
  - condensation and evaporation of cloud liquid water and
    deposition and sublimation of cloud ice (relaxation to equilibrium)
  - autoconversion of cloud liquid water into rain and of cloud ice into snow
  - accretion due to collisions between categories of condensed species
  - evaporation and sublimation of hydrometeors
  - melting of snow into rain
"""
module Microphysics

using SpecialFunctions

using ClimateMachine.Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, T_freeze
using CLIMAParameters.Atmos.Microphysics

const APS = AbstractParameterSet
const ACloudPS = AbstractCloudParameterSet
const APrecipPS = AbstractPrecipParameterSet
const ALPS = AbstractLiquidParameterSet
const AIPS = AbstractIceParameterSet
const ARPS = AbstractRainParameterSet
const ASPS = AbstractSnowParameterSet

export τ_relax

export terminal_velocity

export conv_q_vap_to_q_liq_ice

export conv_q_liq_to_q_rai
export conv_q_ice_to_q_sno

export accretion
export accretion_rain_sink
export accretion_snow_rain

export evaporation_sublimation
export snow_melt

"""
    v0_rai(param_set, ρ)

 - `param_set` - abstract set with earth parameters
 - `ρ` air density

Returns the proportionality coefficient in terminal velocity(r/r0).
"""
function v0_rai(param_set::APS, rain_param_set::ARPS, ρ::FT) where {FT <: Real}

    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
    _C_drag::FT = Microphysics.C_drag(param_set)
    _grav::FT = grav(param_set)
    _r0::FT = r0(rain_param_set)

    return sqrt(FT(8 / 3) / _C_drag * (_ρ_cloud_liq / ρ - FT(1)) * _grav * _r0)
end

"""
    n0_sno(snow_param_set, q_sno, ρ)

 - `snow_param_set` - abstract set with snow parameters
 - `q_sno` -  snow specific humidity
 - `ρ` - air density

Returns the intercept parameter of the assumed Marshall-Palmer distribution of
snow particles.
"""
function n0_sno(snow_param_set::ASPS, q_sno::FT, ρ::FT) where {FT <: Real}

    _ν_sno::FT = ν_sno(snow_param_set)
    _μ_sno::FT = μ_sno(snow_param_set)

    return _μ_sno * (ρ * q_sno)^_ν_sno
end

"""
    unpack_params(param_set, microphysics_param_set, ρ, q_)

 - `param_set` - abstract set with earth parameters
 - `microphysics_param_set` - abstract set with microphysics parameters
 - `q_` - specific humidity
 - `ρ` - air density

Utility function that unpacks microphysics parameters.
"""
function unpack_params(
    param_set::APS,
    ice_param_set::AIPS,
    ρ::FT,
    q_ice::FT,
) where {FT <: Real}
    #TODO - make ρ and q_ice optional
    _n0::FT = n0(ice_param_set)
    _r0::FT = r0(ice_param_set)

    _m0::FT = m0(param_set, ice_param_set)
    _me::FT = me(ice_param_set)

    _χm::FT = χm(ice_param_set)
    _Δm::FT = Δm(ice_param_set)

    return (_n0, _r0, _m0, _me, _χm, _Δm)
end
function unpack_params(
    param_set::APS,
    rain_param_set::ARPS,
    ρ::FT,
    q_rai::FT,
) where {FT <: Real}
    #TODO - make q_rai optional
    _n0::FT = n0(rain_param_set)
    _r0::FT = r0(rain_param_set)

    _m0::FT = m0(param_set, rain_param_set)
    _me::FT = me(rain_param_set)
    _a0::FT = a0(rain_param_set)
    _ae::FT = ae(rain_param_set)
    _v0::FT = v0_rai(param_set, rain_param_set, ρ)
    _ve::FT = ve(rain_param_set)

    _χm::FT = χm(rain_param_set)
    _Δm::FT = Δm(rain_param_set)
    _χa::FT = χa(rain_param_set)
    _Δa::FT = Δa(rain_param_set)
    _χv::FT = χv(rain_param_set)
    _Δv::FT = Δv(rain_param_set)

    return (
        _n0,
        _r0,
        _m0,
        _me,
        _χm,
        _Δm,
        _a0,
        _ae,
        _χa,
        _Δa,
        _v0,
        _ve,
        _χv,
        _Δv,
    )
end
function unpack_params(
    param_set::APS,
    snow_param_set::ASPS,
    ρ::FT,
    q_sno::FT,
) where {FT <: Real}

    _n0::FT = n0_sno(snow_param_set, q_sno, ρ)
    _r0::FT = r0(snow_param_set)

    _m0::FT = m0(snow_param_set)
    _me::FT = me(snow_param_set)
    _a0::FT = a0(snow_param_set)
    _ae::FT = ae(snow_param_set)
    _v0::FT = v0(snow_param_set)
    _ve::FT = ve(snow_param_set)

    _χm::FT = χm(snow_param_set)
    _Δm::FT = Δm(snow_param_set)
    _χa::FT = χa(snow_param_set)
    _Δa::FT = Δa(snow_param_set)
    _χv::FT = χv(snow_param_set)
    _Δv::FT = Δv(snow_param_set)

    return (
        _n0,
        _r0,
        _m0,
        _me,
        _χm,
        _Δm,
        _a0,
        _ae,
        _χa,
        _Δa,
        _v0,
        _ve,
        _χv,
        _Δv,
    )
end


"""
    lambda(q, ρ, n0, m0, me, r0, χm, Δm)

 - `q` - specific humidity of rain, ice or snow
 - `ρ` - air density
 - `n0` - size distribution parameter
 - `m0`, `me`, `χm`, `Δm`, `r0` - mass(radius) parameters

Returns the rate parameter of the assumed size distribution of
particles (rain drops, ice crystals, snow crystals).
"""
function lambda(
    q::FT,
    ρ::FT,
    n0::FT,
    m0::FT,
    me::FT,
    r0::FT,
    χm::FT,
    Δm::FT,
) where {FT <: Real}

    λ::FT = FT(0)

    if q > FT(0)
        λ =
            (
                χm * m0 * n0 * gamma(me + Δm + FT(1)) / ρ / q / r0^(me + Δm)
            )^FT(1 / (me + Δm + 1))
    end
    return λ
end

"""
    τ_relax(liquid_param_set)
    τ_relax(ice_param_set)

 - `liquid_param_set` - abstract set with cloud liquid water parameters
 - `ice_param_set` - abstract set with cloud ice parameters

Returns the relaxation timescale for condensation and evaporation of
cloud liquid water or the relaxation timescale for sublimation and
deposition of cloud ice.
"""
function τ_relax(liquid_param_set::ALPS)

    _τ_relax = τ_cond_evap(liquid_param_set)
    return _τ_relax
end
function τ_relax(ice_param_set::AIPS)

    _τ_relax = τ_sub_dep(ice_param_set)
    return _τ_relax
end

"""
    G_func(param_set, T, Liquid())
    G_func(param_set, T, Ice())

 - `param_set` - abstract set with earth parameters
 - `T` - air temperature
 - `Liquid()`, `Ice()` - liquid or ice phase to dispatch over.

Utility function combining thermal conductivity and vapor diffusivity effects.
"""
function G_func(param_set::APS, T::FT, ::Liquid) where {FT <: Real}

    _K_therm::FT = K_therm(param_set)
    _R_v::FT = R_v(param_set)
    _D_vapor::FT = D_vapor(param_set)

    L = latent_heat_vapor(param_set, T)
    p_vs = saturation_vapor_pressure(param_set, T, Liquid())

    return FT(1) / (
        L / _K_therm / T * (L / _R_v / T - FT(1)) + _R_v * T / _D_vapor / p_vs
    )
end
function G_func(param_set::APS, T::FT, ::Ice) where {FT <: Real}

    _K_therm::FT = K_therm(param_set)
    _R_v::FT = R_v(param_set)
    _D_vapor::FT = D_vapor(param_set)

    L = latent_heat_sublim(param_set, T)
    p_vs = saturation_vapor_pressure(param_set, T, Ice())

    return FT(1) / (
        L / _K_therm / T * (L / _R_v / T - FT(1)) + _R_v * T / _D_vapor / p_vs
    )
end

"""
    terminal_velocity(param_set, precip_param_set, ρ, q_)

 - `param_set` - abstract set with earth parameters
 - `precip_param_set` - abstract set with rain or snow parameters
 - `ρ` - air density
 - `q_` - rain or snow specific humidity

Returns the mass weighted average terminal velocity assuming
a Marshall-Palmer (1948) distribution of rain drops and snow crystals.
"""
function terminal_velocity(
    param_set::APS,
    precip_param_set::APrecipPS,
    ρ::FT,
    q_::FT,
) where {FT <: Real}
    fall_w = FT(0)
    if q_ > FT(0)

        (_n0, _r0, _m0, _me, _χm, _Δm, _a0, _ae, _χa, _Δa, _v0, _ve, _χv, _Δv) =
            unpack_params(param_set, precip_param_set, ρ, q_)

        _λ::FT = lambda(q_, ρ, _n0, _m0, _me, _r0, _χm, _Δm)

        fall_w =
            _χv *
            _v0 *
            (_λ * _r0)^(-_ve - _Δv) *
            gamma(_me + _ve + _Δm + _Δv + FT(1)) / gamma(_me + _Δm + FT(1))
    end

    return fall_w
end

"""
    conv_q_vap_to_q_liq_ice(liquid_param_set::ALPS, q_sat, q)
    conv_q_vap_to_q_liq_ice(ice_param_set::AIPS, q_sat, q)

 - `liquid_param_set` - abstract set with cloud water parameters
 - `ice_param_set` - abstract set with cloud ice parameters
 - `q_sat` - PhasePartition at equilibrium
 - `q` - current PhasePartition

Returns the cloud water tendency due to condensation and evaporation
or cloud ice tendency due to sublimation and vapor deposition.
The tendency is obtained assuming a relaxation to equilibrium with
a constant timescale.
"""
function conv_q_vap_to_q_liq_ice(
    liquid_param_set::ALPS,
    q_sat::PhasePartition{FT},
    q::PhasePartition{FT},
) where {FT <: Real}

    _τ_cond_evap::FT = τ_relax(liquid_param_set)

    return (q_sat.liq - q.liq) / _τ_cond_evap
end
function conv_q_vap_to_q_liq_ice(
    ice_param_set::AIPS,
    q_sat::PhasePartition{FT},
    q::PhasePartition{FT},
) where {FT <: Real}

    _τ_sub_dep::FT = τ_relax(ice_param_set)

    return (q_sat.ice - q.ice) / _τ_sub_dep
end

"""
    conv_q_liq_to_q_rai(rain_param_set, q_liq)

 - `rain_param_set` - abstract set with rain microphysics parameters
 - `q_liq` - liquid water specific humidity

Returns the q_rai tendency due to collisions between cloud droplets
(autoconversion), parametrized following Kessler (1995).
"""
function conv_q_liq_to_q_rai(rain_param_set::ARPS, q_liq::FT) where {FT <: Real}

    _τ_acnv::FT = τ_acnv(rain_param_set)
    _q_liq_threshold::FT = q_liq_threshold(rain_param_set)

    return max(FT(0), q_liq - _q_liq_threshold) / _τ_acnv
end

"""
    conv_q_ice_to_q_sno(param_set, ice_param_set, q, ρ, T)

 - `param_set` - abstract set with earth parameters
 - `ice_param_set` - abstract set with ice microphysics parameters
 - `q` - phase partition
 - `ρ` - air density
 - `T` - air temperature

Returns the q_sno tendency due to autoconversion from ice.
Parameterized following Harrington et al. (1996) and Kaul et al. (2015).
"""
function conv_q_ice_to_q_sno(
    param_set::APS,
    ice_param_set::AIPS,
    q::PhasePartition{FT},
    ρ::FT,
    T::FT,
) where {FT <: Real}
    acnv_rate = FT(0)
    _S::FT = supersaturation(param_set, q, ρ, T, Ice())

    if (q.ice > FT(0) && _S > FT(0))

        _G::FT = G_func(param_set, T, Ice())

        _r_ice_snow::FT = r_ice_snow(ice_param_set)

        (_n0, _r0, _m0, _me, _χm, _Δm) =
            unpack_params(param_set, ice_param_set, ρ, q.ice)

        _λ::FT = lambda(q.ice, ρ, _n0, _m0, _me, _r0, _χm, _Δm)

        acnv_rate =
            4 * FT(π) * _S * _G * _n0 / ρ *
            exp(-_λ * _r_ice_snow) *
            (
                _r_ice_snow^FT(2) / (_me + _Δm) +
                (_r_ice_snow * _λ + FT(1)) / _λ^FT(2)
            )
    end
    return acnv_rate
end

"""
    accretion(param_set, cloud_param_set, precip_param_set, q_clo, q_pre, ρ)

 - `param_set` - abstract set with earth parameters
 - `cloud_param_set` - abstract set with cloud water or cloud ice parameters
 - `precip_param_set` - abstract set with rain or snow parameters
 - `q_clo` - cloud water or cloud ice specific humidity
 - `q_pre` - rain water or snow specific humidity
 - `ρ` - rain water or snow specific humidity

Returns the sink of cloud water (liquid or ice) due to collisions
with precipitating water (rain or snow).
"""
function accretion(
    param_set::APS,
    cloud_param_set::ACloudPS,
    precip_param_set::APrecipPS,
    q_clo::FT,
    q_pre::FT,
    ρ::FT,
) where {FT <: Real}

    accr_rate = FT(0)
    if (q_clo > FT(0) && q_pre > FT(0))

        (_n0, _r0, _m0, _me, _χm, _Δm, _a0, _ae, _χa, _Δa, _v0, _ve, _χv, _Δv) =
            unpack_params(param_set, precip_param_set, ρ, q_pre)

        _λ::FT = lambda(q_pre, ρ, _n0, _m0, _me, _r0, _χm, _Δm)
        _E::FT = E(cloud_param_set, precip_param_set)

        accr_rate =
            q_clo * _E * _n0 * _a0 * _v0 * _χa * _χv / _λ *
            gamma(_ae + _ve + _Δa + _Δv + FT(1)) /
            (_λ * _r0)^(_ae + _ve + _Δa + _Δv)
    end
    return accr_rate
end

"""
    accretion_rain_sink(param_set, ice_param_set, rain_param_set,
                        q_ice, q_rai, ρ)

 - `param_set` - abstract set with earth parameters
 - `ice_param_set` - abstract set with cloud ice parameters
 - `rain_param_set` - abstract set with rain parameters
 - `q_ice` - cloud ice specific humidity
 - `q_rai` - rain water specific humidity
 - `ρ` - air density

Returns the sink of rain water (partial source of snow) due to collisions
with cloud ice.
"""
function accretion_rain_sink(
    param_set::APS,
    ice_param_set::AIPS,
    rain_param_set::ARPS,
    q_ice::FT,
    q_rai::FT,
    ρ::FT,
) where {FT <: Real}

    accr_rate = FT(0)
    if (q_ice > FT(0) && q_rai > FT(0))

        (_n0_ice, _r0_ice, _m0_ice, _me_ice, _χm_ice, _Δm_ice) =
            unpack_params(param_set, ice_param_set, ρ, q_ice)

        (
            _n0_rai,
            _r0_rai,
            _m0_rai,
            _me_rai,
            _χm_rai,
            _Δm_rai,
            _a0_rai,
            _ae_rai,
            _χa_rai,
            _Δa_rai,
            _v0_rai,
            _ve_rai,
            _χv_rai,
            _Δv_rai,
        ) = unpack_params(param_set, rain_param_set, ρ, q_rai)

        _E::FT = E(ice_param_set, rain_param_set)

        _λ_rai::FT = lambda(
            q_rai,
            ρ,
            _n0_rai,
            _m0_rai,
            _me_rai,
            _r0_rai,
            _χm_rai,
            _Δm_rai,
        )
        _λ_ice::FT = lambda(
            q_ice,
            ρ,
            _n0_ice,
            _m0_ice,
            _me_ice,
            _r0_ice,
            _χm_ice,
            _Δm_ice,
        )

        accr_rate =
            _E / ρ *
            _n0_rai *
            _n0_ice *
            _m0_rai *
            _a0_rai *
            _v0_rai *
            _χm_rai *
            _χa_rai *
            _χv_rai / _λ_ice / _λ_rai * gamma(
                _me_rai +
                _ae_rai +
                _ve_rai +
                _Δm_rai +
                _Δa_rai +
                _Δv_rai +
                FT(1),
            ) /
            (
                _r0_rai * _λ_rai
            )^(_me_rai + _ae_rai + _ve_rai + _Δm_rai + _Δa_rai + _Δv_rai)
    end
    return accr_rate
end

"""
    accretion_snow_rain(param_set, i_param_set, j_param_set, q_i, q_j, ρ)

 - `i` - snow for temperatures below freezing
         or rain for temperatures above freezing
 - `j` - rain for temperatures below freezing
         or rain for temperatures above freezing
 - `param_set` - abstract set with earth parameters
 - `i_param_set`, `j_param_set` - abstract set with snow or rain
    microphysics parameters
 - `q_` - specific humidity of snow or rain
 - `ρ` - air density

Returns the accretion rate between rain and snow.
Collisions between rain and snow result in
snow at temperatures below freezing and in rain at temperatures above freezing.
"""
function accretion_snow_rain(
    param_set::APS,
    i_param_set::APrecipPS,
    j_param_set::APrecipPS,
    q_i::FT,
    q_j::FT,
    ρ::FT,
) where {FT <: Real}

    accr_rate = FT(0)
    if (q_i > FT(0) && q_j > FT(0))

        (
            _n0_i,
            _r0_i,
            _m0_i,
            _me_i,
            _χm_i,
            _Δm_i,
            _a0_i,
            _ae_i,
            _χa_i,
            _Δa_i,
            _v0_i,
            _ve_i,
            _χv_i,
            _Δv_i,
        ) = unpack_params(param_set, i_param_set, ρ, q_i)
        (
            _n0_j,
            _r0_j,
            _m0_j,
            _me_j,
            _χm_j,
            _Δm_j,
            _a0_j,
            _ae_j,
            _χa_j,
            _Δa_j,
            _v0_j,
            _ve_j,
            _χv_j,
            _Δv_j,
        ) = unpack_params(param_set, j_param_set, ρ, q_j)

        _E_ij::FT = E(i_param_set, j_param_set)

        _λ_i::FT = lambda(q_i, ρ, _n0_i, _m0_i, _me_i, _r0_i, _χm_i, _Δm_i)
        _λ_j::FT = lambda(q_j, ρ, _n0_j, _m0_j, _me_j, _r0_j, _χm_j, _Δm_j)

        _v_ti = terminal_velocity(param_set, i_param_set, ρ, q_i)
        _v_tj = terminal_velocity(param_set, j_param_set, ρ, q_j)

        accr_rate =
            FT(π) / ρ *
            _n0_i *
            _n0_j *
            _m0_j *
            _χm_j *
            _E_ij *
            abs(_v_ti - _v_tj) / _r0_j^(_me_j + _Δm_j) * (
                FT(2) * gamma(_me_j + _Δm_j + FT(1)) / _λ_i^FT(3) /
                _λ_j^(_me_j + _Δm_j + FT(1)) +
                FT(2) * gamma(_me_j + _Δm_j + FT(2)) / _λ_i^FT(2) /
                _λ_j^(_me_j + _Δm_j + FT(2)) +
                gamma(_me_j + _Δm_j + FT(3)) / _λ_i /
                _λ_j^(_me_j + _Δm_j + FT(3))
            )
    end
    return accr_rate
end

"""
    evaporation_sublimation(param_set, rain_param_set, q, q_rai, ρ, T)
    evaporation_sublimation(param_set, snow_param_set, q, q_sno, ρ, T)

 - `param_set` - abstract set with earth parameters
 - `rain_param_set` - abstract set with rain microphysics parameters
 - `snow_param_set` - abstract set with snow microphysics parameters
 - `q` - phase partition
 - `q_rai` - rain specific humidity
 - `q_sno` - snow specific humidity
 - `ρ` - air density
 - `T` - air temperature

Returns the tendency due to rain evaporation or snow sublimation.
"""
function evaporation_sublimation(
    param_set::APS,
    rain_param_set::ARPS,
    q::PhasePartition{FT},
    q_rai::FT,
    ρ::FT,
    T::FT,
) where {FT <: Real}
    evap_subl_rate = FT(0)
    _S::FT = supersaturation(param_set, q, ρ, T, Liquid())

    if (q_rai > FT(0) && _S < FT(0))

        _a_vent::FT = a_vent(rain_param_set)
        _b_vent::FT = b_vent(rain_param_set)
        _ν_air::FT = ν_air(param_set)
        _D_vapor::FT = D_vapor(param_set)

        _G::FT = G_func(param_set, T, Liquid())

        (_n0, _r0, _m0, _me, _χm, _Δm, _a0, _ae, _χa, _Δa, _v0, _ve, _χv, _Δv) =
            unpack_params(param_set, rain_param_set, ρ, q_rai)

        _λ::FT = lambda(q_rai, ρ, _n0, _m0, _me, _r0, _χm, _Δm)

        evap_subl_rate =
            4 * FT(π) * _n0 / ρ * _S * _G / _λ^FT(2) * (
                _a_vent +
                _b_vent * (_ν_air / _D_vapor)^FT(1 / 3) /
                (_r0 * _λ)^((_ve + _Δv) / FT(2)) *
                (FT(2) * _v0 * _χv / _ν_air / _λ)^FT(1 / 2) *
                gamma((_ve + _Δv + FT(5)) / FT(2))
            )
    end
    return evap_subl_rate
end
function evaporation_sublimation(
    param_set::APS,
    snow_param_set::ASPS,
    q::PhasePartition{FT},
    q_sno::FT,
    ρ::FT,
    T::FT,
) where {FT <: Real}
    evap_subl_rate = FT(0)
    if q_sno > FT(0)

        _a_vent::FT = a_vent(snow_param_set)
        _b_vent::FT = b_vent(snow_param_set)
        _ν_air::FT = ν_air(param_set)
        _D_vapor::FT = D_vapor(param_set)

        _S::FT = supersaturation(param_set, q, ρ, T, Ice())
        _G::FT = G_func(param_set, T, Ice())

        (_n0, _r0, _m0, _me, _χm, _Δm, _a0, _ae, _χa, _Δa, _v0, _ve, _χv, _Δv) =
            unpack_params(param_set, snow_param_set, ρ, q_sno)
        _λ::FT = lambda(q_sno, ρ, _n0, _m0, _me, _r0, _χm, _Δm)

        evap_subl_rate =
            4 * FT(π) * _n0 / ρ * _S * _G / _λ^FT(2) * (
                _a_vent +
                _b_vent * (_ν_air / _D_vapor)^FT(1 / 3) /
                (_r0 * _λ)^((_ve + _Δv) / FT(2)) *
                (FT(2) * _v0 * _χv / _ν_air / _λ)^FT(1 / 2) *
                gamma((_ve + _Δv + FT(5)) / FT(2))
            )
    end
    return evap_subl_rate
end

"""
    snow_melt(param_set, snow_param_set, q_sno, ρ, T)

 - `param_set` - abstract set with earth parameters
 - `snow_param_set` - abstract set with snow microphysics parameters
 - `q_sno` - snow water specific humidity
 - `ρ` - air density
 - `T` - air temperature

Returns the tendency due to snow melt.
"""
function snow_melt(
    param_set::APS,
    snow_param_set::ASPS,
    q_sno::FT,
    ρ::FT,
    T::FT,
) where {FT <: Real}

    snow_melt_rate = FT(0)
    _T_freeze = T_freeze(param_set)

    if (q_sno > FT(0) && T > _T_freeze)

        _a_vent::FT = a_vent(snow_param_set)
        _b_vent::FT = b_vent(snow_param_set)
        _ν_air::FT = ν_air(param_set)
        _D_vapor::FT = D_vapor(param_set)
        _K_therm::FT = K_therm(param_set)

        L = latent_heat_fusion(param_set, T)

        (_n0, _r0, _m0, _me, _χm, _Δm, _a0, _ae, _χa, _Δa, _v0, _ve, _χv, _Δv) =
            unpack_params(param_set, snow_param_set, ρ, q_sno)
        _λ::FT = lambda(q_sno, ρ, _n0, _m0, _me, _r0, _χm, _Δm)

        snow_melt_rate =
            4 * FT(π) * _n0 / ρ * _K_therm / L * (T - _T_freeze) / _λ^FT(2) * (
                _a_vent +
                _b_vent * (_ν_air / _D_vapor)^FT(1 / 3) /
                (_r0 * _λ)^((_ve + _Δv) / FT(2)) *
                (FT(2) * _v0 * _χv / _ν_air / _λ)^FT(1 / 2) *
                gamma((_ve + _Δv + FT(5)) / FT(2))
            )
    end
    return snow_melt_rate
end

end #module Microphysics.jl
