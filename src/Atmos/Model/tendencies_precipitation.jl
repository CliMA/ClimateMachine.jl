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

function source(s::WarmRain_1M{Rain}, m, args)
    nt = warm_rain_sources(m, args)
    return nt.S_ρ_qr
end

function source(s::RainSnow_1M{Rain}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_qr
end

function source(s::RainSnow_1M{Snow}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_qs
end
