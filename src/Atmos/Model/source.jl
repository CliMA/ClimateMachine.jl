using CLIMA.PlanetParameters: Omega
export Gravity, RayleighSponge, Subsidence, GeostrophicForcing, Coriolis

# kept for compatibility
# can be removed if no functions are using this
function atmos_source!(f::Function, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
  f(state, aux, t, source)
end
function atmos_source!(::Nothing, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
end
# sources are applied additively
function atmos_source!(stuple::Tuple, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
  map(s -> atmos_source!(s, atmos, state, aux, t, source), stuple)
end

abstract type Source
end

struct Gravity <: Source
end
function atmos_source!(::Gravity, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
  if atmos.ref_state isa HydrostaticState
    source.ρu -= (state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
  else
    source.ρu -= state.ρ * aux.orientation.∇Φ
  end
end

struct Coriolis <: Source
end
function atmos_source!(::Coriolis, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
  # note: this assumes a SphericalOrientation
  source.ρu -= SVector(0, 0, 2*Omega) × state.ρu
end

struct GeostrophicForcing{FT} <: Source
  f_coriolis::FT
  u_geostrophic::FT
  v_geostrophic::FT
end
function atmos_source!(s::GeostrophicForcing, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
  u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
  ẑ = vertical_unit_vector(atmos.orientation, aux)
  fkvector = s.f_coriolis * ẑ
  source.ρu -= fkvector × (state.ρu .- state.ρ*u_geo)
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
  u_relaxation::SVector{3,FT}
  "Sponge exponent"
  γ::FT
end
function atmos_source!(s::RayleighSponge, atmos::AtmosModel, state::Vars, aux::Vars, t::Real, source::Vars)
  z = altitude(atmos.orientation, aux)
  if z >= s.z_sponge
    r = (z - s.z_sponge)/(s.z_max-s.z_sponge)
    β_sponge = s.α_max * sinpi(r/2)^s.γ
    source.ρu -= β_sponge * (state.ρu .- state.ρ*s.u_relaxation)
  end
end
