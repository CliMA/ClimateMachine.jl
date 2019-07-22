module Atmos

export AtmosModel, 
  ConstantViscosityWithDivergence, 
  DryModel, MoistEquil,
  NoRadiation,
  NoFluxBC, InitStateBC

using LinearAlgebra, StaticArrays
using ..VariableTemplates

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient, vars_diffusive,
  flux!, source!, wavespeed, boundarycondition!, gradvariables!, diffusive!,
  init_aux!, init_state!, update_aux!

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modelling.

# Usage

    AtmosModel(turbulence, moisture, radiation, source, boundarycondition, init_state)

"""
struct AtmosModel{T,M,R,S,BC,IS} <: BalanceLaw
  turbulence::T
  moisture::M
  radiation::R
  # TODO: a better mechanism than functions.
  source::S
  boundarycondition::BC
  init_state::IS
end

function vars_state(m::AtmosModel, T)
  NamedTuple{(:ρ, :ρu, :ρe, :turbulence, :moisture, :radiation), 
  Tuple{T, SVector{3,T}, T, vars_state(m.turbulence,T), vars_state(m.moisture,T), vars_state(m.radiation, T)}}
end
function vars_gradient(m::AtmosModel, T)
  NamedTuple{(:u, :turbulence, :moisture, :radiation),
  Tuple{SVector{3,T}, vars_gradient(m.turbulence,T), vars_gradient(m.moisture,T), vars_gradient(m.radiation,T)}}
end
function vars_diffusive(m::AtmosModel, T)
  NamedTuple{(:ρτ, :turbulence, :moisture, :radiation),
  Tuple{SVector{6,T}, vars_diffusive(m.turbulence,T), vars_diffusive(m.moisture,T), vars_diffusive(m.radiation,T)}}
end
function vars_aux(m::AtmosModel, T)
  NamedTuple{(:coord, :turbulence, :moisture, :radiation),
  Tuple{NamedTuple{(:x,:y,:z),Tuple{T,T,T}}, vars_aux(m.turbulence,T), vars_aux(m.moisture,T), vars_aux(m.radiation,T)}}
end

# Navier-Stokes flux terms
function flux!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  # preflux
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  
  p = pressure(m.moisture, state, aux, t)

  # invisc terms
  flux.ρ  = ρu 
  flux.ρu = ρu .* u' + p*I
  flux.ρe = u * (state.ρe + p)

  flux_diffusive!(m, flux, state, diffusive, aux, t)
end

function flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  u = (1/state.ρ) * state.ρu

  # diffusive
  ρτ11, ρτ22, ρτ33, ρτ12, ρτ13, ρτ23 = diffusive.ρτ
  ρτ = SMatrix{3,3}(ρτ11, ρτ12, ρτ13,
                    ρτ12, ρτ22, ρτ23,
                    ρτ13, ρτ23, ρτ33)
  flux.ρu += ρτ
  flux.ρe += ρτ*u

  # moisture-based diffusive fluxes
  flux_diffusive!(m.moisture, flux, state, diffusive, aux, t)
end

function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  ρu = state.ρu
  u = ρinv * ρu
  return abs(dot(nM, u)) + soundspeed(m.moisture, state, aux, t)
end

function gradvariables!(m::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1 / state.ρ
  transform.u = ρinv * state.ρu

  gradvariables!(m.moisture, transform, state, aux, t)
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

  # kinematic viscosity tensor
  ρν = dynamic_viscosity_tensor(m.turbulence, S, state, aux, t)

  # momentum flux tensor
  diffusive.ρτ = scaled_momentum_flux_tensor(m.turbulence, ρν, S)

  # diffusivity of moisture components
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρν)
end

function update_aux!(m::AtmosModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  update_aux!(m.moisture, state, diffusive, aux, t)
end

include("turbulence.jl")
include("moisture.jl")
include("radiation.jl")

# TODO: figure out a nice way to handle this
function init_aux!(::AtmosModel, aux::Vars, (x,y,z))
  aux.coord.x = x
  aux.coord.y = y
  aux.coord.z = z
end

function source!(bl::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  bl.source(source, state, aux, t)
end


# TODO: figure out a better interface for this.
# at the moment we can just pass a function, but we should do something better
# need to figure out how subcomponents will interact.
function boundarycondition!(bl::AtmosModel, stateP::Vars, diffP::Vars, auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t)
  bl.boundarycondition(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

abstract type BoundaryCondition
end

"""
    NoFluxBC <: BoundaryCondition

Set the momentum at the boundary to be zero.
"""
struct NoFluxBC <: BoundaryCondition
end
function boundarycondition!(bl::AtmosModel{T,M,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars, 
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {T,M,R,S,BC <: NoFluxBC,IS}
  
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * nM  
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is mainly useful for cases where the problem has an explicit solution.
"""
struct InitStateBC <: BoundaryCondition
end
function boundarycondition!(bl::AtmosModel{T,M,R,S,BC,IS}, stateP::Vars, diffP::Vars, auxP::Vars, 
    nM, stateM::Vars, diffM::Vars, auxM::Vars, bctype, t) where {T,M,R,S,BC <: InitStateBC,IS}
  coord = (auxP.coord.x, auxP.coord.y, auxP.coord.z)
  init_state!(bl, stateP, auxP, coord, t)
end

function init_state!(bl::AtmosModel, state::Vars, aux::Vars, coords, t)
  bl.init_state(state, aux, coords, t)
end

end # module
