using CLIMAParameters.Planet: Omega
export Source, Gravity, RayleighSponge, Subsidence, GeostrophicForcing, Coriolis

# kept for compatibility
# can be removed if no functions are using this
function atmos_source!(
    f::Function,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    f(atmos, source, state, diffusive, aux, t, direction)
end
function atmos_source!(
    ::Nothing,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) end
# sources are applied additively
@generated function atmos_source!(
    stuple::Tuple,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> atmos_source!(
            stuple[i],
            atmos,
            source,
            state,
            diffusive,
            aux,
            t,
            direction,
        )
        return nothing
    end
end

abstract type Source end

struct Gravity <: Source end
function atmos_source!(
    ::Gravity,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    if atmos.ref_state isa HydrostaticState
        source.ρu -= (state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
    else
        source.ρu -= state.ρ * aux.orientation.∇Φ
    end
end

struct Coriolis <: Source end
function atmos_source!(
    ::Coriolis,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    _Omega::FT = Omega(atmos.param_set)
    # note: this assumes a SphericalOrientation
    source.ρu -= SVector(0, 0, 2 * _Omega) × state.ρu
end

struct Subsidence{FT} <: Source
    D::FT
end

function atmos_source!(
    subsidence::Subsidence,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    z = altitude(atmos, aux)
    w_sub = subsidence_velocity(subsidence, z)
    k̂ = vertical_unit_vector(atmos, aux)

    source.ρe -= ρ * w_sub * dot(k̂, diffusive.∇h_tot)
    source.moisture.ρq_tot -= ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

subsidence_velocity(subsidence::Subsidence{FT}, z::FT) where {FT} =
    -subsidence.D * z


struct GeostrophicForcing{FT} <: Source
    f_coriolis::FT
    u_geostrophic::FT
    v_geostrophic::FT
end
function atmos_source!(
    s::GeostrophicForcing,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
    ẑ = vertical_unit_vector(atmos, aux)
    fkvector = s.f_coriolis * ẑ
    source.ρu -= fkvector × (state.ρu .- state.ρ * u_geo)
end

"""
    RayleighSponge{FT} <: Source

Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{FT} <: Source
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
function atmos_source!(
    s::RayleighSponge,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    z = altitude(atmos, aux)
    if z >= s.z_sponge
        r = (z - s.z_sponge) / (s.z_max - s.z_sponge)
        β_sponge = s.α_max * sinpi(r / 2)^s.γ
        source.ρu -= β_sponge * (state.ρu .- state.ρ * s.u_relaxation)
    end
end
