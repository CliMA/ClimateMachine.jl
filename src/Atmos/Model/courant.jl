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
        norm_u =  abs(dot(state.ρu, k̂))/state.ρ
    elseif direction isa HorizontalDirection
        norm_u = norm( (state.ρu .- dot(state.ρu, k̂).*k̂) / state.ρ )
    else
        norm_u = norm(state.ρu/state.ρ)
    end
    return Δt * norm_u / Δx
end

function nondiffusive_courant(m::BalanceLaw, state::Vars, aux::Vars,
                              diffusive::Vars, Δx, Δt, direction=VerticalDirection())
    k̂ = vertical_unit_vector(m.orientation, aux)
    if direction isa VerticalDirection
        norm_u =  abs(dot(state.ρu, k̂))/state.ρ
    elseif direction isa HorizontalDirection
        norm_u = norm( (state.ρu .- dot(state.ρu, k̂)*k̂) / state.ρ )
    else
        norm_u = norm(state.ρu/state.ρ)
    end
    return Δt * (norm_u + soundspeed(m.moisture, m.orientation, state, aux)) / Δx
end

function diffusive_courant(m::BalanceLaw, state::Vars, aux::Vars, diffusive::Vars, Δx, Δt, direction=VerticalDirection())
    ν, τ = turbulence_tensors(m.turbulence, state, diffusive, aux, 0)
    FT = eltype(state)

    if ν isa Real    
        return Δt * ν / (Δx*Δx)
    else
        k̂ = vertical_unit_vector(m.orientation, aux)
        ν_vert = dot(ν, k)

        if direction isa VerticalDirection
            return Δt * ν_vert / (Δx*Δx)
        elseif direction isa HorizontalDirection
            ν_horz = ν - ν_vert .* k
            return Δt * norm(ν_horz) / (Δx*Δx)
        else
            return Δt * norm(ν) / (Δx*Δx)
        end
    end
end

end

