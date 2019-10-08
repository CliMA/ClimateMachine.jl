export Gravity, RayleighSponge, Subsidence, GeostrophicForcing

# kept for compatibility
# can be removed if no functions are using this
function atmos_source!(f::Function, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  f(source, state, aux, t)
end
function atmos_source!(::Nothing, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
end
# sources are applied additively
function atmos_source!(stuple::Tuple, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  map(s -> atmos_source!(s, m, source, state, aux, t), stuple)
end

abstract type Source
end

struct Gravity <: Source
end
function atmos_source!(::Gravity, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρu -= state.ρ * aux.orientation.∇Φ
end

struct Subsidence <: Source
end
function atmos_source!(::Subsidence, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρu -= state.ρ * m.radiation.D_subsidence
end

struct GeostrophicForcing{DT} <: Source
  f_coriolis::DT
  u_geostrophic::DT
  v_geostrophic::DT
end
function atmos_source!(s::GeostrophicForcing, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  u = state.ρu / state.ρ
  u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
  source.ρu -= state.ρ * s.f_coriolis * (u - u_geo)
end

"""
  RayleighSponge{DT} <: Source
Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{DT} <: Source
  "Domain maximum height [m]"
  zmax::DT
  "Vertical extent at with sponge starts [m]"
  zsponge::DT
  "Sponge Strength 0 ⩽ c_sponge ⩽ 1"
  c_sponge::DT
end
function atmos_source!(s::RayleighSponge, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  DT = eltype(state)
  z = aux.orientation.Φ / grav
  coeff = DT(0)
  if z >= s.zsponge
    coeff_top = s.c_sponge * (sinpi(DT(1/2)*(z - s.zsponge)/(s.zmax-s.zsponge)))^DT(4)
    coeff = min(coeff_top, 1.0)
  end
  source.ρu -= state.ρu * coeff
end
