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

struct NoSource <: Source
end
vars_aux(::Source, DT) = @vars()
function atmos_source!(::NoSource, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
end
function init_aux!(m::Source, aux::Vars, geom::LocalGeometry)
end

struct Gravity <: Source
end
function atmos_source!(::Gravity, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρu -= state.ρ * aux.orientation.∇Φ
end
"""
  RayleighSponge{DT} <: Sponge
Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
If sponge type is not specified, Sponge defaults to NoSponge with no modifications to the velocity field at the top
"buffer" zone.
"""
struct RayleighSponge{DT} <: Source
  "Domain maximum height [m]"
  zmax::DT
  "Vertical extent at with sponge starts [m]"
  zsponge::DT
  "Sponge Strength 0 ⩽ c_sponge ⩽ 1"
  c_sponge::DT
end
vars_aux(::RayleighSponge, DT) = @vars(coeff::DT)
function init_aux!(m::RayleighSponge, aux::Vars, geom::LocalGeometry)
  DT = eltype(aux)
  z = aux.orientation.Φ / grav
  zmax = m.zmax
  zsponge = m.zsponge
  coeff_top = m.c_sponge * (sinpi(DT(1/2)*(z - zsponge)/(zmax-zsponge)))^DT(4)
  aux.source.coeff = min(1 + coeff_top, 1.0)
end
function atmos_source!(::RayleighSponge, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρu -= state.ρu * aux.source.coeff
end
