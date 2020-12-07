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

function source(s::Subsidence{Mass}, m, state, aux, t, ts, direction, diffusive)
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

function source(
    s::RemovePrecipitation{Mass},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    if has_condensate(ts)
        nt = compute_precip_params(s, aux, ts)
        return state.ρ * nt.S_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(s::Rain_1M{Mass}, m, state, aux, t, ts, direction, diffusive)
    nt = compute_rain_params(m, state, aux, t, ts)
    return state.ρ * nt.S_qt
end
