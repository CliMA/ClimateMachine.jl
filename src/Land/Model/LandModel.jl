module Land

using DocStringExtensions
using LinearAlgebra, StaticArrays

using CLIMAParameters
using CLIMAParameters.Planet:
    ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0, T_freeze, grav

using ..VariableTemplates
using ..MPIStateArrays
using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    nodal_update_auxiliary_state!,
    wavespeed
using ..DGMethods: LocalGeometry, DGModel
export LandModel


"""
    LandModel{PS, S, LBC, SRC, IS} <: BalanceLaw

A BalanceLaw for land modeling.
Users may over-ride prescribed default values for each field.

# Usage

    LandModel(
        param_set,
        soil;
        boundary_conditions,
        source,
        init_state_prognostic
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LandModel{PS, S, R, LBC, SRC, IS} <: BalanceLaw
    "Parameter set"
    param_set::PS
    "Soil model"
    soil::S
    "River model"
    river::R
    "struct of boundary conditions"
    boundary_conditions::LBC
    "Source Terms (Problem specific source terms)"
    source::SRC
    "Initial Condition (Function to assign initial values of state variables)"
    init_state_prognostic::IS
end

"""
    LandModel(
        param_set::AbstractParameterSet,
        soil::BalanceLaw;
        boundary_conditions::LBC = (),
        source::SRC = (),
        init_state_prognostic::IS = nothing
    ) where {SRC, IS, LBC}

Constructor for the LandModel structure.
"""
function LandModel(
    param_set::AbstractParameterSet,
    soil::BalanceLaw;
    river::BalanceLaw = NoRiverModel(),
    boundary_conditions::LBC = LandDomainBC(),
    source::SRC = (),
    init_state_prognostic::IS = nothing,
) where {SRC, IS, LBC}
    @assert init_state_prognostic ≠ nothing
    land = (param_set, soil, river, boundary_conditions, source, init_state_prognostic)
    return LandModel{typeof.(land)...}(land...)
end


function vars_state(land::LandModel, st::Prognostic, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
        river::vars_state(land.river, st, FT)
    end
end


function vars_state(land::LandModel, st::Auxiliary, FT)
    @vars begin
        x::FT
        y::FT
        z::FT
        soil::vars_state(land.soil, st, FT)
        river::vars_state(land.river, st, FT)
    end
end

function vars_state(land::LandModel, st::Gradient, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
        river::vars_state(land.river, st, FT)
    end
end

function vars_state(land::LandModel, st::GradientFlux, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
        river::vars_state(land.river, st, FT)
    end
end

function nodal_init_state_auxiliary!(
    land::LandModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.x = geom.coord[1]
    aux.y = geom.coord[2]
    aux.z = geom.coord[3]
    land_init_aux!(land, land.soil, aux, geom)
    land_init_aux!(land, land.river, aux, geom)
end

function flux_first_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    directions,
) 
   flux_first_order!(land, land.river, flux, state, aux, t, directions) 
end


function compute_gradient_argument!(
    land::LandModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_argument!(land, land.soil, transform, state, aux, t)
    #compute_gradient_argument!(land, land.river, transform, state, aux, t)
end

function compute_gradient_flux!(
    land::LandModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_flux!(
        land,
        land.soil,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

end

function flux_second_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(
        land,
        land.soil,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )

end

function nodal_update_auxiliary_state!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, land.soil, state, aux, t)
    land_nodal_update_auxiliary_state!(land, land.river, state, aux, t)
end


function source!(
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    land_source!(land.source, land, source, state, diffusive, aux, t, direction)
end


function init_state_prognostic!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    land.init_state_prognostic(land, state, aux, coords, t, args...)
end


include("SoilWaterParameterizations.jl")
using .SoilWaterParameterizations
include("SoilHeatParameterizations.jl")
using .SoilHeatParameterizations
include("soil_model.jl")
include("soil_water.jl")
include("soil_heat.jl")
include("Runoff.jl")
using .Runoff
include("land_bc.jl")
include("soil_bc.jl")
include("source.jl")
include("River.jl")
using .River

function wavespeed(
    m::LandModel,
    n⁻,
    state::Vars,
    aux::Vars,
    t::Real,
    direction
)
    FT = eltype(state)
    g = FT(9.8)
    width = m.river.width(aux.x,aux.y)
    area = max(eltype(state)(0.0), state.river.area)
    height = area ./ width
    v = calculate_velocity(m.river, aux.x, aux.y, height)
    speed = abs(norm(v))
    return speed#+sqrt(g*height)
end

end # Module
