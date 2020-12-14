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

struct ViscousProduction{PV <: Energy} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(
    ::ViscousProduction{Energy},
    m,
    state,
    aux,
    t,
    ts,
    diffusive,
    hyperdiff,
)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    return τ * state.ρu
end

struct EnthalpyProduction{PV <: Energy} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(
    ::EnthalpyProduction{Energy},
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
        nt = compute_precip_params(s, aux, ts)
        @unpack S_qt, λ, I_l, I_i, Φ = nt
        return (λ * I_l + (1 - λ) * I_i + Φ) * state.ρ * S_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(s::Rain_1M{Energy}, m, state, aux, t, ts, direction, diffusive)
    nt = compute_rain_params(m, state, aux, t, ts)
    @unpack S_qt, Φ, I_l = nt
    return state.ρ * S_qt * (Φ + I_l)
end

export TemperatureRelaxation
"""
    TemperatureRelaxation{FT} <: AbstractSource

Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct TemperatureRelaxation{PV <: Energy, FT} <: TendencyDef{Source, PV}
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
TemperatureRelaxation(::Type{FT}, args...) where {FT} =
    TemperatureRelaxation{Energy, FT}(args...)
function source(
    s::TemperatureRelaxation{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    z = altitude(m, aux)

    ρe_int = state.ρ * internal_energy(ts)
    if z >= s.z_sponge
        r = (z - s.z_sponge) / (s.z_max - s.z_sponge)
        β_sponge = s.α_max * sinpi(r / 2)^s.γ
        return -β_sponge * (ρe_int .- aux.ρe₀)
    else
        FT = eltype(state)
        return SVector{3, FT}(0, 0, 0)
    end
end
