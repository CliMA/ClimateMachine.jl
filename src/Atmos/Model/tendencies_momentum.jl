##### Momentum tendencies

using CLIMAParameters.Planet: Omega

#####
##### First order fluxes
#####

function flux(::Advect{Momentum}, m, state, aux, t, ts, direction)
    return state.ρu .* (state.ρu / state.ρ)'
end

function flux(::PressureGradient{Momentum}, m, state, aux, t, ts, direction)
    pad = (state.ρu .* (state.ρu / state.ρ)') * 0
    if m.ref_state isa HydrostaticState
        return pad + (air_pressure(ts) - aux.ref_state.p) * I
    else
        return pad + air_pressure(ts) * I
    end
end

#####
##### Second order fluxes
#####

struct ViscousStress{PV <: Momentum} <: TendencyDef{Flux{SecondOrder}, PV} end
function flux(
    ::ViscousStress{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    diffusive,
    hyperdiff,
)
    ν, D_t, τ = turbulence_tensors(m, state, diffusive, aux, t)
    return τ * state.ρ
end

#####
##### Sources
#####

export Gravity
struct Gravity{PV <: Momentum} <: TendencyDef{Source, PV} end
Gravity() = Gravity{Momentum}()
function source(s::Gravity{Momentum}, m, args)
    @unpack state, aux = args
    if m.ref_state isa HydrostaticState
        return -(state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
    else
        return -state.ρ * aux.orientation.∇Φ
    end
end

export Coriolis
struct Coriolis{PV <: Momentum} <: TendencyDef{Source, PV} end
Coriolis() = Coriolis{Momentum}()
function source(s::Coriolis{Momentum}, m, args)
    @unpack state = args
    FT = eltype(state)
    _Omega::FT = Omega(m.param_set)
    # note: this assumes a SphericalOrientation
    return -SVector(0, 0, 2 * _Omega) × state.ρu
end

export GeostrophicForcing
struct GeostrophicForcing{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    f_coriolis::FT
    u_geostrophic::FT
    v_geostrophic::FT
end
GeostrophicForcing(::Type{FT}, args...) where {FT} =
    GeostrophicForcing{Momentum, FT}(args...)
function source(s::GeostrophicForcing{Momentum}, m, args)
    @unpack state, aux = args
    u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    fkvector = s.f_coriolis * ẑ
    return -fkvector × (state.ρu .- state.ρ * u_geo)
end


export RayleighSponge
"""
    RayleighSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}

Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}
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
RayleighSponge(::Type{FT}, args...) where {FT} =
    RayleighSponge{Momentum, FT}(args...)
function source(s::RayleighSponge{Momentum}, m, args)
    @unpack state, aux = args
    z = altitude(m, aux)
    if z >= s.z_sponge
        r = (z - s.z_sponge) / (s.z_max - s.z_sponge)
        β_sponge = s.α_max * sinpi(r / 2)^s.γ
        return -β_sponge * (state.ρu .- state.ρ * s.u_relaxation)
    else
        FT = eltype(state)
        return SVector{3, FT}(0, 0, 0)
    end
end
