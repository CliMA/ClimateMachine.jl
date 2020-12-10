##### Mass tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Mass}, m, state, aux, t, ts, direction)
    return state.ρu
end

#####
##### Second order fluxes
#####

struct MoistureDiffusion{PV <: Mass} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(
    ::MoistureDiffusion{Mass},
    m,
    state,
    aux,
    t,
    ts,
    diffusive,
    hyperdiff,
)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    return d_q_tot * state.ρ
end

#####
##### Sources
#####

function source(s::Subsidence{Mass}, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

function source(s::RemovePrecipitation{Mass}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, args)
        return nt.S_ρ_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(s::WarmRain_1M{Mass}, m, args)
    nt = warm_rain_sources(m, args)
    return nt.S_ρ_qt
end

function source(s::RainSnow_1M{Mass}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_qt
end
