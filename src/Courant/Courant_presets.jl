module Courant

export advective_courant, diffusive_courant, nondiffusive_courant

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
import ..MoistThermodynamics: internal_energy
using ..SubgridScaleParameters
using GPUifyLoops
using CLIMA.Atmos: AtmosModel, soundspeed
import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state,
                        vars_diffusive
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection
using Logging
using Printf
using LinearAlgebra
using CLIMA.Atmos


function advective_courant(m::BalanceLaw, state::Vars, aux::Vars,
                               diffusive::Vars, Δx, Δt, direction=VerticalDirection())
  k̂ = aux.orientation.∇Φ / norm(aux.orientation.∇Φ)
  if direction isa VerticalDirection
    u = dot(state.ρu/state.ρ, k̂) * k̂
          return Δt * (norm(u)) / Δx
  elseif direction isa HorizontalDirection
    u = cross(k̂, cross(state.ρu/state.ρ,k̂))
    return Δt * (norm(u)) / Δx
  end
end

function nondiffusive_courant(m::BalanceLaw, state::Vars, aux::Vars,
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

function diffusive_courant(m::BalanceLaw, state::Vars, aux::Vars,
                               diffusive::Vars, Δx, Δt, direction=VerticalDirection())
  ν, τ = turbulence_tensors(m.turbulence, state, diffusive, aux, 0)
  FT = eltype(state)
  k̂ = aux.orientation.∇Φ / norm(aux.orientation.∇Φ)
  ν_cour = SVector{3,FT}(ν,ν,ν) 
  ν_horz = SDiagonal(cross(k̂,cross(ν_cour,k̂)))
  ν_vert = SDiagonal(dot(k̂, ν_cour) * k̂)
  S = symmetrize(diffusive.turbulence.∇u)
  diff_horz = norm(-2 * state.ρ * ν_horz * S)
  diff_vert = norm(-2 * state.ρ * ν_vert * S)
  if direction isa VerticalDirection
    s_diff = diff_vert
          return Δt * (s_diff) / Δx^2
  elseif direction isa HorizontalDirection
    s_diff = diff_horz
    return Δt * (s_diff) / Δx^2
  end
end


end

