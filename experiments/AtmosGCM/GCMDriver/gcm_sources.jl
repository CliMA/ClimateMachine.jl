# GCM-specific Sources
# This file contains helpers and lists currently available options

# Current options for GCM-specific sources:

"""
    HeldSuarezForcing{PV <: Union{Momentum,Energy}} <: TendencyDef{Source, PV}

Defines a forcing that parametrises radiative and frictional effects using
Newtonian relaxation and Rayleigh friction, following Held and Suarez (1994)
"""
struct HeldSuarezForcing{PV <: Union{Momentum, Energy}} <:
       TendencyDef{Source, PV} end

HeldSuarezForcing() =
    (HeldSuarezForcing{Momentum}(), HeldSuarezForcing{Energy}())

import ClimateMachine.BalanceLaws: source
import ClimateMachine.Atmos: filter_source, atmos_source!

filter_source(pv::PV, m, s::HeldSuarezForcing{PV}) where {PV} = s
atmos_source!(::HeldSuarezForcing, args...) = nothing

function held_suarez_forcing_coefficients(bl, args)
    @unpack state, aux = args
    @unpack ts = args.precomputed
    FT = eltype(state)

    # Parameters
    T_ref = FT(255)

    _R_d = FT(R_d(bl.param_set))
    _day = FT(day(bl.param_set))
    _grav = FT(grav(bl.param_set))
    _cp_d = FT(cp_d(bl.param_set))
    _p0 = FT(MSLP(bl.param_set))

    # Held-Suarez parameters
    k_a = FT(1 / (40 * _day))
    k_f = FT(1 / _day)
    k_s = FT(1 / (4 * _day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)

    # Held-Suarez forcing
    φ = latitude(bl, aux)
    p = air_pressure(ts)

    #TODO: replace _p0 with dynamic surface pressure in Δσ calculations to account
    #for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor
    return (k_v = k_v, k_T = k_T, T_equil = T_equil)
end

function source(s::HeldSuarezForcing{Energy}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    nt = held_suarez_forcing_coefficients(m, args)
    FT = eltype(state)
    _cv_d = FT(cv_d(m.param_set))
    @unpack k_T, T_equil = nt
    T = air_temperature(ts)
    return -k_T * state.ρ * _cv_d * (T - T_equil)
end

function source(s::HeldSuarezForcing{Momentum}, m, args)
    nt = held_suarez_forcing_coefficients(m, args)
    return -nt.k_v * projection_tangential(m, args.aux, args.state.ρu)
end
