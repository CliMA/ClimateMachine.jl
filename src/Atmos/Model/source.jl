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

struct Subsidence <: Source
end
function atmos_source!(::Subsidence, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
    ### SUBDISEDENCE SHOULD BE ADDED HERE
    return


    
#    n = aux.orientation.∇Φ ./ norm(aux.orientation.∇Φ)
#    source.ρu -= m.radiation.D_subsidence * dot(state.ρu, n) * n
#
    ##    source.ρqt
    
end

struct Coriolis <: Source
end
function atmos_source!(::Coriolis, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  # note: this assumes a SphericalOrientation 
  source.ρu -= cross(SVector(0, 0, 2*Omega), state.ρu)
  #source.ρu -= SVector(0, 0, 2*Omega) × state.ρu
end

struct GeostrophicForcing{FT} <: Source
  f_coriolis::FT
  u_geostrophic::FT
  v_geostrophic::FT
end
function atmos_source!(s::GeostrophicForcing, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)

  u          = state.ρu/state.ρ
  u_geo      = SVector(s.u_geostrophic, s.v_geostrophic, 0)
  fkvector   = SVector(0, 0, s.f_coriolis) 
  source.ρu -= state.ρ * cross(fkvector, (u - u_geo))
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
  "Relaxtion velocity components"
  u_relaxation::SVector{3,FT}  
end

function atmos_source!(s::RayleighSponge, m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
    FT = eltype(state)
    z = aux.orientation.Φ / grav
    coeff = FT(0)
    τsponge = FT(6)
    if z >= s.zsponge
        coeff_top = s.c_sponge * (sinpi(FT(1/2)*(z - s.zsponge)/(s.zmax-s.zsponge)))^FT(4)
#        coeff_top = s.c_sponge * (sinpi(FT(1/2)*(z - s.zsponge)/(s.zmax-s.zsponge)))^FT(2)
      coeff = min(coeff_top, FT(1))
    end
        
    u = state.ρu / state.ρ
    source.ρu -= state.ρ * coeff * (u - s.u_relaxation)
    
    ##
    #=
    D = FT(3.75e-6)
    u_ref = s.u_relaxation[1]
    v_ref = s.u_relaxation[2]    
    w_ref = -D*z
    u_relaxation = SVector{3,FT}(u_ref, v_ref, w_ref)
    source.ρu -= state.ρ * coeff * (u - u_relaxation)
    =#
    ##
end
