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

struct Coriolis <: Source
end
function atmos_source!(::Coriolis, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
#  Ω = SVector(0,0,somevalue)
#  source.ρu -= Ω⃗ × u⃗
end


"""
  RayleighSponge{DT} <: Sponge
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
  zmax = s.zmax
  zsponge = s.zsponge
  coeff_top = s.c_sponge * (sinpi(DT(1/2)*(z - zsponge)/(zmax-zsponge)))^DT(4)
  coeff = min(1 + coeff_top, 1.0)
  source.ρu -= state.ρu * coeff
end
