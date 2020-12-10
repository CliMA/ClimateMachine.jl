##### Precipitation tendencies

#####
##### First order fluxes
#####

#####
##### Second order fluxes
#####

function flux(::Diffusion{Rain}, m, state, aux, t, ts, diffusive, hyperdiff)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    d_q_rai = (-D_t) .* diffusive.precipitation.∇q_rai
    return d_q_rai * state.ρ
end

function flux(::Diffusion{Snow}, m, state, aux, t, ts, diffusive, hyperdiff)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    d_q_sno = (-D_t) .* diffusive.precipitation.∇q_sno
    return d_q_sno * state.ρ
end


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
