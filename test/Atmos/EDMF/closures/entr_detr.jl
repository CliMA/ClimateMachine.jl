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
    lim_E = entr.lim_ϵ
    lim_amp = entr.lim_amp
    w_min = entr.w_min
    # precompute vars
    w_up_i = up[i].ρaw / up[i].ρa
    sqrt_tke = sqrt(max(en.ρatke, 0) * ρ_inv / env.a)
    # ensure far from zero
    Δw = filter_w(w_up_i - env.w, w_min)
    w_up_i = filter_w(w_up_i, w_min)
    Δb = up_aux[i].buoyancy - en_aux.buoyancy
    D_E, D_δ, M_δ, M_E =
        nondimensional_exchange_functions(m, entr, state, aux, t, ts, env, i)

    # I am commenting this out for now, to make sure there is no slowdown here
    Λ_w = abs(Δb / Δw)
    Λ_tke = entr.c_λ * abs(Δb / (max(en.ρatke * ρ_inv, 0) + w_min))
    λ = lamb_smooth_minimum(
        SVector(Λ_w, Λ_tke),
        m.turbconv.mix_len.smin_ub,
        m.turbconv.mix_len.smin_rm,
    )

    # compute entrainment/detrainment components
    E_trb =
        2 * up[i].ρa * entr.c_t * sqrt_tke /
        max(m.turbconv.pressure.H_up, entr.H_up_min)
    E_dyn = up[i].ρa * λ * (D_E + M_E)
    Δ_dyn = up[i].ρa * λ * (D_δ + M_δ)

    E_dyn = max(E_dyn, FT(0))
    Δ_dyn = max(Δ_dyn, FT(0))
    E_trb = max(E_trb, FT(0))
    return E_dyn, Δ_dyn, E_trb
end;
