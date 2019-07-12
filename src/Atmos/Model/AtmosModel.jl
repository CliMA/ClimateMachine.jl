using LinearAlgebra, StaticArrays

import CLIMA.DGmethods: BalanceLaw, dimension, vars_aux, vars_state, vars_transform, vars_diffusive,
  flux!, source!, wavespeed, boundarycondition!, gradtransform!, diffusive!,
  init_aux!, init_state!, preodefun!

struct AtmosModel{T,M,R,F} <: BalanceLaw
  turbulence::T
  moisture::M
  radiation::R
  force::F
end

function vars_state(m::AtmosModel, T)
  NamedTuple{(:ρ, :ρu, :ρe, :turbulence, :moisture, :radiation), 
  Tuple{T, SVector{3,T}, T, vars_state(m.turbulence,T), vars_state(m.moisture,T), vars_state(m.radiation, T)}}
end
function vars_transform(m::AtmosModel, T)
  NamedTuple{(:u, :turbulence, :moisture, :radiation),
  Tuple{SVector{3,T}, vars_transform(m.turbulence,T), vars_transform(m.moisture,T), vars_transform(m.radiation,T)}}
end
function vars_diffusive(m::AtmosModel, T)
  NamedTuple{(:τ, :turbulence, :moisture, :radiation),
  Tuple{SVector{6,T}, vars_diffusive(m.turbulence,T), vars_diffusive(m.moisture,T), vars_diffusive(m.radiation,T)}}
end
function vars_aux(m::AtmosModel, T)
  NamedTuple{(:turbulence, :moisture, :radiation),
  Tuple{vars_aux(m.turbulence,T), vars_aux(m.moisture,T), vars_aux(m.radiation,T)}}
end


# Navier-Stokes flux terms
function flux!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # preflux
  ρinv = 1 / state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  
  p = pressure(m.moisture, state, diffusive, aux, t)

  # invisc terms
  flux.ρ  = ρu
  flux.ρu = ρu .* u' + p*I
  flux.ρe = u * (state.ρe + p)

  flux_diffusive!(m, flux, state, diffusive, aux, t)
end

function flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  u = (1/state.ρ) * state.ρu

  # diffusive
  τ11, τ22, τ33, τ12, τ13, τ23 = diffusive.τ
  ρτ = state.ρ * SMatrix{3,3}(τ11, τ12, τ13,
                              τ12, τ22, τ23,
                              τ13, τ23, τ33)
  flux.ρu += ρτ
  flux.ρe += ρτ*u

  # moisture-based diffusive fluxes
  flux_diffusive!(m.moisture, flux, state, diffusive, aux, t)
end



function source!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source!(m.force, source, state, aux, t)
end

function wavespeed(::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  return abs(dot(nM, u)) + soundspeed(m.moisture, state, aux, t)
end

function gradtransform!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  transform.u = ρinv * state.ρu

  gradtransform!(m.moisture, transform, state, aux, t)
end

function diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  ∇u = ∇transform.u
  
  # strain rate tensor
  # TODO: we use an SVector for this, but should define a "SymmetricSMatrix"? 
  S = SVector(∇u[1,1],
              ∇u[2,2],
              ∇u[3,3],
              (∇u[1,2] + ∇u[2,1])/2,
              (∇u[1,3] + ∇u[3,1])/2,
              (∇u[2,3] + ∇u[3,2])/2)

  # strain rate tensor norm
  # NOTE: factor of 2 scaling
  # normS = norm(2S)
  normS = sqrt(2*(S[1]^2 + S[2]^2 + S[3]^2 + 2*(S[4]^2 + S[5]^2 + S[6]^2))) 

  # kinematic viscosity tensor
  ν = kinematic_viscosity_tensor(m.turbulence, normS)

  # momentum flux tensor
  diffusive.τ = (-2*ν) .* S

  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ν)
end



function preodefun!(m::AtmosModel, aux::Vars, state::Vars, t::Real)
  # map
  preodefun_elem!(m.moisture, aux, state, t)
  preodefun!(m.radiation, aux, state, t)
end


# utility function for specifying the flux terms on tracers
@inline function flux_tracer!(sym::Symbol, flux::Grad, state::Vars)
  # preflux
  ρinv = 1 / state.ρ
  u = ρinv * state.ρu
 
  # invisc terms
  setfield!(flux, sym, u * getfield(state, sym))
end

MMS = AtmosModel(ConstantViscocity(?), DryModel(), NoRadiation())

DYCOMS = AtmosModel(SmagorinskyLilly(coef), 
    EquilMoist(), StevensRadiation())


function boundarycondition!(bl::MMSModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  init_state!(bl, stateP, auxP, (auxM.x, auxM.y, auxM.z), t)
end

function init_auxvars!(::MMSModel, aux::Vars, (x,y,z))
  aux.x = x
  aux.y = y
  aux.z = z

end

function init_state!(bl::MMSModel{dim}, state::Vars, aux::Vars, (x,y,z), t) where {dim}
  state.ρ = ρ_g(t, x, y, z, Val(dim))
  state.ρu = U_g(t, x, y, z, Val(dim))
  state.ρv = V_g(t, x, y, z, Val(dim))
  state.ρw = W_g(t, x, y, z, Val(dim))
  state.ρe = E_g(t, x, y, z, Val(dim))
end


state(m::ConstViscosity, T) = NamedTuple{}
state(m::EquilMoist, T) = NamedTuple{(:ρqt,), Tuple{T}}
state(m::NoRadiation, T) = NamedTuple{}

model = AtmosModel(ConstViscosity(), EquilMoist(), NoRadiation())

st = state(model, Float64)

v = Vars{st}(zeros(MVector{6,Float64}))
g = Grad{st}(zeros(MMatrix{3,6,Float64}))
