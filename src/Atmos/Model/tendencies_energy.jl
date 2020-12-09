##### Energy tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Energy}, m, state, aux, t, ts, direction)
    return (state.ρu / state.ρ) * state.ρe
end

function flux(::Pressure{Energy}, m, state, aux, t, ts, direction)
    return state.ρu / state.ρ * air_pressure(ts)
end

#####
##### First order fluxes
#####

struct ViscousFlux{PV <: Energy} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(::ViscousFlux{Energy}, m, state, aux, t, ts, diffusive, hyperdiff)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    return τ * state.ρu
end

struct DiffEnthalpyFlux{PV <: Energy} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(
    ::DiffEnthalpyFlux{Energy},
    m,
    state,
    aux,
    t,
    ts,
    diffusive,
    hyperdiff,
)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    d_h_tot = -D_t .* diffusive.∇h_tot
    return d_h_tot * state.ρ
end

#####
##### Sources
#####

function source(
    s::Subsidence{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.∇h_tot)
end

function source(
    s::RemovePrecipitation{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    if has_condensate(ts)
        nt = compute_remove_precip_params(s, aux, ts)
        @unpack S_qt, λ, I_l, I_i, Φ = nt
        return (λ * I_l + (1 - λ) * I_i + Φ) * state.ρ * S_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(
    s::WarmRain_1M{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = compute_warm_rain_params(m, state, aux, t, ts)
    @unpack S_qt, Φ, I_l = nt
    return state.ρ * S_qt * (Φ + I_l)
end

function source(
    s::RainSnow_1M{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    nt = compute_rain_snow_params(m, state, aux, t, ts)
    return nt.S_ρe
end
