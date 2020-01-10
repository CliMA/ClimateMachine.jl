using CLIMA.PlanetParameters: Omega
export Gravity, RayleighSponge, Subsidence, GeostrophicForcing, Coriolis

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
  if m.ref_state isa HydrostaticState
    source.ρu -= (state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
  else
    source.ρu -= state.ρ * aux.orientation.∇Φ
  end
end

struct Coriolis <: Source
end
function atmos_source!(::Coriolis, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  # note: this assumes a SphericalOrientation
  source.ρu -= SVector(0, 0, 2*Omega) × state.ρu
end

struct GeostrophicForcing{FT} <: Source
  f_coriolis::FT
  u_geostrophic::FT
  v_geostrophic::FT
end
function atmos_source!(s::GeostrophicForcing, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)

  u        = state.ρu/state.ρ
  ug       = SVector(s.u_geostrophic, s.v_geostrophic, 0)
  fkvector = SVector(0, 0, s.f_coriolis)
    
  umome = state.ρ * s.f_coriolis * (u[2] - ug[2])
  vmome = state.ρ * s.f_coriolis * (ug[1] - u[1])
  source.ρu += SVector(umome, vmome, 0)

  #source.ρu -= state.ρ * cross(fkvector, (u - u_geo))
  #source.ρu -= state.ρ * fkvector × (u - u_geo)

end

"""
  RayleighSponge{FT} <: Source
Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{FT} <: Source
  "Domain maximum height [m]"
  zmax::FT
  "Vertical extent at with sponge starts [m]"
  zsponge::FT
  "Sponge Strength 0 ⩽ c_sponge ⩽ 1"
  c_sponge::FT
  #c_sponge::SVector{3,FT}  
  "Relaxtion velocity componens"
  u_relaxation::SVector{3,FT}  
end

function atmos_source!(s::RayleighSponge, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  z = altitude(m.orientation, aux)
  beta = FT(0)
  if z >= s.zsponge
      coeff_top = s.c_sponge * (sinpi(FT(1/2)*(z - s.zsponge)/(s.zmax-s.zsponge)))^FT(4)
      #coeff_top = s.c_sponge * (sinpi(FT(1/2)*(z - s.zsponge)/(s.zmax-s.zsponge)))^FT(2)
      beta = coeff_top
      
      #coeff_top = s.c_sponge * (1 - cos(pi*(z - s.zsponge)/(s.zmax - s.zsponge)));
      #beta = min(coeff_top, FT(1))
  end
      
  u = state.ρu / state.ρ
  source.ρu -= state.ρ * beta * (u - s.u_relaxation)

  # TODO: The following code is in master, and needs to be replaced with the above code:
  # FT = eltype(state)
  # z = altitude(m.orientation, aux)
  # coeff = FT(0)
  # if z >= s.zsponge
  #   coeff_top = s.c_sponge * (sinpi(FT(1/2)*(z - s.zsponge)/(s.zmax-s.zsponge)))^FT(4)
  #   coeff = min(coeff_top, 1.0)
  # end
  # source.ρu -= state.ρu * coeff

end
