#### Entrainment-Detrainment kernels

"""
    entr_detr(
        m::AtmosModel{FT},
        entr::EntrainmentDetrainment,
        state::Vars,
        aux::Vars,
        t::Real,
        ts,
        env,
        i,
    ) where {FT}
Returns the dynamic entrainment and detrainment rates,
as well as the turbulent entrainment rate, following
Cohen et al. (JAMES, 2020), given:
 - `m`, an `AtmosModel`
 - `entr`, an `EntrainmentDetrainment` model
 - `state`, state variables
 - `aux`, auxiliary variables
 - `t`, the time
 - `ts`, NamedTuple of thermodynamic states
 - `env`, NamedTuple of environment variables
 - `i`, index of the updraft
"""
function entr_detr(
    m::AtmosModel{FT},
    entr::EntrainmentDetrainment,
    state::Vars,
    aux::Vars,
    t::Real,
    ts,
    env,
    i,
) where {FT}

    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    en_aux = aux.turbconv.environment
    up_aux = aux.turbconv.updraft

    N_up = n_updrafts(m.turbconv)
    ρ_inv = 1 / gm.ρ
    a_up_i = up[i].ρa * ρ_inv

    lim_ϵ = entr.lim_ϵ
    lim_amp = entr.lim_amp
    w_min = entr.w_min
    # precompute vars
    w_up_i = up[i].ρaw / up[i].ρa
    sqrt_tke = sqrt(max(en.ρatke, 0) * ρ_inv / env.a)
    # ensure far from zero
    Δw = filter_w(w_up_i - env.w, w_min)
    w_up_i = filter_w(w_up_i, w_min)

    Δb = up_aux[i].buoyancy - en_aux.buoyancy

    D_ε, D_δ, M_δ, M_ε =
        nondimensional_exchange_functions(m, entr, state, aux, t, ts, env, i)

    # I am commenting this out for now, to make sure there is no slowdown here
    Λ_w = abs(Δb / Δw)
    Λ_tke = entr.c_λ * abs(Δb / (max(en.ρatke * ρ_inv, 0) + w_min))
    λ = lamb_smooth_minimum(
        SVector(Λ_w, Λ_tke),
        m.turbconv.mix_len.smin_ub,
        m.turbconv.mix_len.smin_rm,
    )

    # compute limiters
    εt_lim = εt_limiter(w_up_i, lim_ϵ, lim_amp)
    ε_lim = ε_limiter(a_up_i, lim_ϵ, lim_amp)
    δ_lim = δ_limiter(a_up_i, lim_ϵ, lim_amp)
    # compute entrainment/detrainment components
    ε_trb =
        2 * a_up_i * entr.c_t * sqrt_tke /
        max((w_up_i * a_up_i * m.turbconv.pressure.H_up), entr.εt_min)
    ε_dyn = λ / w_up_i * (D_ε + M_ε) * ε_lim
    δ_dyn = λ / w_up_i * (D_δ + M_δ) * δ_lim

    ε_dyn = min(max(ε_dyn, FT(0)), FT(1))
    δ_dyn = min(max(δ_dyn, FT(0)), FT(1))
    ε_trb = min(max(ε_trb, FT(0)), FT(1))

    return ε_dyn, δ_dyn, ε_trb
end;

"""
    ε_limiter(a_up::FT, ϵ::FT) where {FT}
Returns the asymptotic value of entrainment
needed to ensure boundedness of the updraft
area fraction between 0 and 1, given
 - `a_up`, the updraft area fraction
 - `ϵ`, a minimum threshold value
"""
ε_limiter(a_up::FT, ϵ::FT, lim_amp::FT) where {FT} =
    FT(1) + lim_amp * exp(-a_up^2 / (2 * ϵ)) - exp(-(FT(1) - a_up)^2 / (2 * ϵ))

"""
    δ_limiter(a_up::FT, ϵ::FT) where {FT}
Returns the asymptotic value of detrainment
needed to ensure boundedness of the updraft
area fraction between 0 and 1, given
 - `a_up`, the updraft area fraction
 - `ϵ`, a minimum threshold value
"""
δ_limiter(a_up::FT, ϵ::FT, lim_amp::FT) where {FT} =
    FT(1) - exp(-a_up^2 / (2 * ϵ)) + lim_amp * exp(-(FT(1) - a_up)^2 / (2 * ϵ))

"""
    εt_limiter(w_up_i::FT, ϵ::FT) where {FT}
Returns the asymptotic value of turbulent
entrainment needed to ensure positiveness
of the updraft vertical velocity, given
 - `w_up_i`, the updraft vertical velocity
 - `ϵ`, a minimum threshold value
"""
εt_limiter(w_up_i::FT, ϵ::FT, lim_amp::FT) where {FT} =
    FT(1) + lim_amp * exp(-w_up_i^2 / (2 * ϵ))
