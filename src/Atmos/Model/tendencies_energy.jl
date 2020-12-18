##### Energy tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Energy}, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.ρe
end

function flux(::Pressure{Energy}, atmos, args)
    @unpack state = args
    @unpack ts = args.precomputed
    return state.ρu / state.ρ * air_pressure(ts)
end

function flux(::KinematicModelPressure{Energy}, atmos, args)
    @unpack state, aux = args
    return state.ρu / state.ρ * aux.ref_state.p
end

#####
##### Second order fluxes
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

function source(s::Subsidence{Energy}, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.∇h_tot)
end

function source(s::RemovePrecipitation{Energy}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, args)
        return nt.S_ρ_e
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(s::WarmRain_1M{Energy}, m, args)
    nt = warm_rain_sources(m, args)
    return nt.S_ρ_e
end

function source(s::RainSnow_1M{Energy}, m, args)
    nt = rain_snow_sources(m, args)
    return nt.S_ρ_e
end
