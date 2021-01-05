##### Precipitation tendencies

#####
##### First order fluxes
#####

function flux(::PrecipitationFlux{Rain}, atmos, args)
    @unpack state, aux = args
    FT = eltype(state)
    u = state.ρu / state.ρ
    q_rai = state.precipitation.ρq_rai / state.ρ

    v_term_rai::FT = FT(0)
    if q_rai > FT(0)
        v_term_rai = terminal_velocity(
            atmos.param_set,
            atmos.param_set.microphys.rai,
            state.ρ,
            q_rai,
        )
    end

    k̂ = vertical_unit_vector(atmos, aux)
    return state.precipitation.ρq_rai * (u - k̂ * v_term_rai)
end

function flux(::PrecipitationFlux{Snow}, atmos, args)
    @unpack state, aux = args
    FT = eltype(state)
    u = state.ρu / state.ρ
    q_sno = state.precipitation.ρq_sno / state.ρ

    v_term_sno::FT = FT(0)
    if q_sno > FT(0)
        v_term_sno = terminal_velocity(
            atmos.param_set,
            atmos.param_set.microphys.sno,
            state.ρ,
            q_sno,
        )
    end

    k̂ = vertical_unit_vector(atmos, aux)
    return state.precipitation.ρq_sno * (u - k̂ * v_term_sno)
end


#####
##### Second order fluxes
#####

function flux(::Diffusion{Rain}, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_rai = (-D_t) .* diffusive.precipitation.∇q_rai
    return d_q_rai * state.ρ
end

function flux(::Diffusion{Snow}, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_sno = (-D_t) .* diffusive.precipitation.∇q_sno
    return d_q_sno * state.ρ
end


#####
##### Sources
#####

function source(s::WarmRain_1M{Rain}, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qr
end

function source(s::RainSnow_1M{Rain}, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qr
end

function source(s::RainSnow_1M{Snow}, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qs
end
