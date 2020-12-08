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
    nt = warm_rain_sources(m, state, aux, ts)
    return nt.S_ρ_qr
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
    nt = rain_snow_sources(m, state, aux, ts)
    return nt.S_ρ_qr
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
    nt = rain_snow_sources(m, state, aux, ts)
    return nt.S_ρ_qs
end
