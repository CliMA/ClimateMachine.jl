##### Energy tendencies

#####
##### First order fluxes
#####
export EnergySponge
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

struct EnergySponge{PV <: Energy, FT} <: TendencyDef{Source, PV}
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Relaxation velocity components"
    u_relaxation::SVector{3, FT}
    "Sponge exponent"
    γ::FT
end

EnergySponge(::Type{FT}, args...) where {FT} =
    EnergySponge{Energy, FT}(args...)
function source(
		    s::EnergySponge{Energy},
		        m,
			    state,
			        aux,
				    t,
				        ts,
					    direction,
					        diffusive,
    FT = eltype(aux)						)
    z = altitude(m, aux)
    if z >= s.z_sponge
      r = (z - s.z_sponge) / (s.z_max - s.z_sponge)
        β_sponge = s.α_max * sinpi(r / 2)^s.γ
      FT = eltype(aux)
        _T_0::FT = T_0(m.param_set)
        _cv_d::FT = cv_d(m.param_set)
        T = aux.ref_state.T
        E_int = state.ρ * _cv_d * (T - _T_0)
        e_kin = 0.5 * (state.ρu[1]^2 + state.ρu[2]^2 + state.ρu[3]^2)
        e_int = (state.ρe) - e_kin - state.ρ * gravitational_potential(m.orientation, aux)
        e_diff = β_sponge * (e_int .- E_int)
        return - e_diff * state.ρ
    else
        return FT(0)
    end
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
        nt = remove_precipitation_sources(s, m, state, aux, ts)
        return nt.S_ρ_e
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
    nt = warm_rain_sources(m, state, aux, ts)
    return nt.S_ρ_e
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
    nt = rain_snow_sources(m, state, aux, ts)
    return nt.S_ρ_e
end
