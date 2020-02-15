module Courant

export advective_courant, diffusive_courant, nondiffusive_courant

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
import ..MoistThermodynamics: internal_energy
using ..SubgridScaleParameters
using GPUifyLoops
using CLIMA.Atmos: AtmosModel, soundspeed, vertical_unit_vector, turbulence_tensors
import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state,
                        vars_diffusive
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection


function advective_courant(m::BalanceLaw, state::Vars, aux::Vars,
                              diffusive::Vars, Δx, Δt, direction=VerticalDirection())
    k̂ = vertical_unit_vector(m.orientation, aux)
    if direction isa VerticalDirection
        u =  abs(dot(state.ρu, k̂))/state.ρ
    elseif direction isa HorizontalDirection
        u = norm( (state.ρu .- dot(state.ρu, k̂)*k̂) / state.ρ )#     cross(k̂, cross(state.ρu/state.ρ,k̂))      
    end
    return Δt * u / Δx
end

function nondiffusive_courant(m::BalanceLaw, state::Vars, aux::Vars,
                              diffusive::Vars, Δx, Δt, direction=VerticalDirection())
    k̂ = vertical_unit_vector(m.orientation, aux)
    if direction isa VerticalDirection
        u =  abs(dot(state.ρu, k̂))/state.ρ
        return Δt * (u + soundspeed(m.moisture, m.orientation, state, aux)) / Δx
    elseif direction isa HorizontalDirection
        u = norm( (state.ρu .- dot(state.ρu, k̂)*k̂) / state.ρ )#     cross(k̂, cross(state.ρu/state.ρ,k̂))      
        return Δt * (u + soundspeed(m.moisture, m.orientation, state, aux)) / Δx
  end
end

function diffusive_courant(m::BalanceLaw, state::Vars, aux::Vars, diffusive::Vars, Δx, Δt, direction=VerticalDirection())
    ν, τ = turbulence_tensors(m.turbulence, state, diffusive, aux, 0)
    FT = eltype(state)

    if ν isa Real    
        return Δt * ν / (Δx*Δx)
    else
        k̂ = vertical_unit_vector(m.orientation, aux)
        ν_vert = ν[1]*k[1] + ν[2]*k[2] + ν[3]*k[3]

        if direction isa VerticalDirection
            return Δt * ν_vert / (Δx*Δx)
        elseif direction isa HorizontalDirection
            ν_horz = MVector{3,FT}(ν,ν,ν) 
            ν_horz[1] -= k[1] * ν_vert 
            ν_horz[2] -= k[2] * ν_vert
            ν_horz[3] -= k[3] * ν_vert

            return Δt * norm(ν_horz) / (Δx*Δx)
        end
    end
end


end

