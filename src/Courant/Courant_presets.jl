module Courant


export Advective_CFL, Diffusive_CFL

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
import ..MoistThermodynamics: internal_energy
using ..SubgridScaleParameters
using GPUifyLoops
using ..MPIStateArrays: MPIStateArray
using CLIMA.Atmos: AtmosModel, soundspeed
import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state,
                        vars_diffusive
using ..Mesh.Topologies
using ..Mesh.Grids
using ..Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection
using Logging
using Printf
using LinearAlgebra
using CLIMA.Atmos



function Advective_CFL(m::AtmosModel, state::Vars, aux::Vars,
                               diffusive::Vars, Δx, Δt, direction=VerticalDirection())
  k̂ = aux.orientation.∇Φ / norm(aux.orientation.∇Φ)
  if direction isa VerticalDirection
    u = dot(state.ρu/state.ρ, k̂) * k̂
          return Δt * (norm(u) + soundspeed(m.moisture, m.orientation, state,aux)) / Δx
  elseif direction isa HorizontalDirection
    u = cross(k̂, cross(state.ρu/state.ρ,k̂))      
    return Δt * (norm(u) + soundspeed(m.moisture, m.orientation, state,
                                            aux)) / Δx
  end
end

function Diffusive_CFL(m::AtmosModel, state::Vars, aux::Vars,
                               diffusive::Vars, Δx, Δt, direction=VerticalDirection())
  ν, τ, sdiff = turbulence_tensors(m.turbulence, state, diffusive, aux, 0)
  S = diffusive.turbulence.S
  diff_horz = sdiff isa Real ? norm(-2 * state.ρ * sdiff * S) : norm(-2 * state.ρ * sdiff[1] * S)
  diff_vert = sdiff isa Real ? norm(-2 * state.ρ * sdiff * S) : norm(-2 * state.ρ * sdiff[2] * S)
  if direction isa VerticalDirection
    s_diff = diff_vert
          return Δt * (s_diff) / Δx^2
  elseif direction isa HorizontalDirection
    s_diff = diff_horz
    return Δt * (s_diff) / Δx^2
  end
end
end
