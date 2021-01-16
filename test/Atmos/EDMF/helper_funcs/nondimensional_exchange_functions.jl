"""
    nondimensional_exchange_functions(
        m::AtmosModel{FT},
        entr::EntrainmentDetrainment,
        state::Vars,
        aux::Vars,
        t::Real,
        ts_up,
        ts_en,
        env,
        buoy,
        i,
    ) where {FT}

Returns the nondimensional entrainment and detrainment
functions following Cohen et al. (JAMES, 2020), given:
 - `m`, an `AtmosModel`
 - `entr`, an `EntrainmentDetrainment` model
 - `state`, state variables
 - `aux`, auxiliary variables
 - `ts_up`, updraft thermodynamic states
 - `ts_en`, environment thermodynamic states
 - `env`, NamedTuple of environment variables
 - `buoy`, NamedTuple of environment and updraft buoyancies
 - `i`, the updraft index
"""
function nondimensional_exchange_functions(
    m::AtmosModel{FT},
    entr::EntrainmentDetrainment,
    state::Vars,
    aux::Vars,
    ts_up,
    ts_en,
    env,
    buoy,
    i,
) where {FT}

    # Alias convention:
    gm = state
    up = state.turbconv.updraft
    up_aux = aux.turbconv.updraft
    en_aux = aux.turbconv.environment

    # precompute vars
    w_min = entr.w_min
    N_up = n_updrafts(m.turbconv)
    ρ_inv = 1 / gm.ρ
    a_up_i = fix_void_up(up[i].ρa, up[i].ρa * ρ_inv)
    w_up_i = fix_void_up(up[i].ρa, up[i].ρaw / up[i].ρa, w_min)

    # thermodynamic variables
    RH_up = relative_humidity(ts_up[i])
    RH_en = relative_humidity(ts_en)

    Δw = filter_w(fix_void_up(up[i].ρa, w_up_i - env.w, w_min), w_min)
    Δb = buoy.up[i] - buoy.en

    c_δ = sign(condensate(ts_en) + condensate(ts_up[i])) * entr.c_δ

    # compute dry and moist aux functions

    # -- not working --
    # μ_ij = fix_void_up(up[i].ρa,
    #     (entr.χ - a_up_i / (a_up_i + env.a)) * Δb / Δw,
    #     FT(1))

    # -- working --
    if up[i].ρa*ρ_inv>m.turbconv.subdomains.a_min
        μ_ij = (entr.χ - a_up_i / (a_up_i + env.a)) * Δb / Δw
    else
        μ_ij = FT(1)
    end
    M_ε = c_δ * (max((RH_en^entr.β - RH_up^entr.β), 0))^(1 / entr.β)
    M_δ = c_δ * (max((RH_up^entr.β - RH_en^entr.β), 0))^(1 / entr.β)

    D_ε = entr.c_ε / (1 + exp(-μ_ij / entr.μ_0))
    D_δ = entr.c_ε / (1 + exp(μ_ij / entr.μ_0))
    return D_ε, D_δ, M_δ, M_ε
end;
