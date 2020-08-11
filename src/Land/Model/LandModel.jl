module Land

using CLIMAParameters
using DocStringExtensions
using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MPIStateArrays

using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!

using ..DGMethods: LocalGeometry, DGModel, nodal_init_state_auxiliary!

export LandModel

"""
    LandModel{PS, S, SRC, IS} <: BalanceLaw

A BalanceLaw for land modeling.
Users may over-ride prescribed default values for each field.

# Usage

    LandModel(
        param_set,
        soil,
        source
        init_state_prognostic
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LandModel{PS, S, SRC, IS} <: BalanceLaw
    "Parameter set"
    param_set::PS
    "Soil model"
    soil::S
    "Source Terms (Problem specific source terms)"
    source::SRC
    "Initial Condition (Function to assign initial values of state variables)"
    init_state_prognostic::IS
end

"""
    LandModel(
        param_set::AbstractParameterSet,
        soil::BalanceLaw;
        source::SRC = (),
        init_state_prognostic::IS = nothing
    ) where {SRC, IS}

Constructor for the LandModel structure. 
"""
function LandModel(
    param_set::AbstractParameterSet,
    soil::BalanceLaw;
    source::SRC = (),
    init_state_prognostic::IS = nothing,
) where {SRC, IS}
    @assert init_state_prognostic ≠ nothing
    land = (param_set, soil, source, init_state_prognostic)
    return LandModel{typeof.(land)...}(land...)
end


function vars_state(land::LandModel, st::Prognostic, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end


function vars_state(land::LandModel, st::Auxiliary, FT)
    @vars begin
        z::FT
        soil::vars_state(land.soil, st, FT)
    end
end

function vars_state(land::LandModel, st::Gradient, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

function vars_state(land::LandModel, st::GradientFlux, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

function init_state_auxiliary!(
    m::LandModel,
    state_auxiliary::MPIStateArray,
    grid,
)

    nodal_init_state_auxiliary!(
        m,
        (m, aux, tmp, geom) -> land_nodal_init_state_auxiliary!(m, aux, geom),
        state_auxiliary,
        grid,
    )
end

function land_nodal_init_state_auxiliary!(
    land::LandModel,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    land_init_aux!(land, land.soil, aux, geom)
end

function flux_first_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    directions,
) end


function compute_gradient_argument!(
    land::LandModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_argument!(land, land.soil, transform, state, aux, t)
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


function update_auxiliary_state!(
    dg::DGModel,
    land::LandModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(
        land_nodal_update_auxiliary_state!,
        dg,
        land,
        Q,
        t,
        elems,
    )
end


function land_nodal_update_auxiliary_state!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, land.soil, state, aux, t)
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

include("land_bc.jl")
include("SoilWaterParameterizations.jl")
using .SoilWaterParameterizations
include("soil_model.jl")
include("soil_water.jl")
include("soil_heat.jl")
include("soil_bc.jl")
include("source.jl")
end # Module
