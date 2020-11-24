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

function source(s::Rain_1M{Rain}, m, state, aux, t, ts, direction, diffusive)
    nt = compute_rain_params(m, state, aux, t, ts)
    return state.ρ * nt.S_qr
end
