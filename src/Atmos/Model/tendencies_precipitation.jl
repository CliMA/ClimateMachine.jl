##### Precipitation tendencies

#####
##### First order fluxes
#####

#####
##### Second order fluxes
#####

#####
##### Sources
#####

function source(
    s::WarmRain_1M{Rain},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = compute_warm_rain_params(m, state, aux, t, ts)
    return state.ρ * nt.S_qr
end

function source(
    s::RainSnow_1M{Rain},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = compute_rain_snow_params(m, state, aux, t, ts)
    return state.ρ * nt.S_qr
end

function source(
    s::RainSnow_1M{Snow},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = compute_rain_snow_params(m, state, aux, t, ts)
    return state.ρ * nt.S_qs
end
