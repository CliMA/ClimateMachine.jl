module Land

using DocStringExtensions
using UnPack
using DispatchedTuples
using LinearAlgebra, StaticArrays

using CLIMAParameters
using CLIMAParameters.Planet:
    ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0, T_freeze, grav

using ..VariableTemplates
using ..MPIStateArrays
using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    prognostic_vars,
    get_prog_state,
    flux,
    source,
    eq_tends,
    precompute,
    vars_state,
    boundary_conditions,
    parameter_set,
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
    LandModel{PS, S, SF, LBC, SRC, SRCDT, IS} <: BalanceLaw

A BalanceLaw for land modeling.
Users may over-ride prescribed default values for each field.

# Usage

    LandModel(
        param_set,
        soil;
        surface,
        boundary_conditions,
        source,
        source_dt,
        init_state_prognostic
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LandModel{PS, S, SF, LBC, SRC, SRCDT, IS} <: BalanceLaw
    "Parameter set"
    param_set::PS
    "Soil model"
    soil::S
    "Surface Flow model"
    surface::SF
    "struct of boundary conditions"
    boundary_conditions::LBC
    "Source Terms (Problem specific source terms)"
    source::SRC
    "DispatchedTuple of sources"
    source_dt::SRCDT
    "Initial Condition (Function to assign initial values of state variables)"
    init_state_prognostic::IS
end

parameter_set(m::LandModel) = m.param_set

"""
    LandModel(
        param_set::AbstractParameterSet,
        soil::BalanceLaw;
        surface::BalanceLaw = NoSurfaceFlowModel(),
        boundary_conditions::LBC = (),
        source::SRC = (),
        init_state_prognostic::IS = nothing
    ) where {SRC, IS, LBC}

Constructor for the LandModel structure.
"""
function LandModel(
    param_set::AbstractParameterSet,
    soil::BalanceLaw;
    surface::BalanceLaw = NoSurfaceFlowModel(),
    boundary_conditions::LBC = LandDomainBC(),
    source::SRC = (),
    init_state_prognostic::IS = nothing,
) where {SRC, IS, LBC}
    @assert init_state_prognostic ≠ nothing
    source_dt = prognostic_var_source_map(source)
    land = (
        param_set,
        soil,
        surface,
        boundary_conditions,
        source,
        source_dt,
        init_state_prognostic,
    )
    return LandModel{typeof.(land)...}(land...)
end


function vars_state(land::LandModel, st::Prognostic, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
        surface::vars_state(land.surface, st, FT)
    end
end


function vars_state(land::LandModel, st::Auxiliary, FT)
    @vars begin
        x::FT
        y::FT
        z::FT
        soil::vars_state(land.soil, st, FT)
        surface::vars_state(land.surface, st, FT)
    end
end

function vars_state(land::LandModel, st::Gradient, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
        surface::vars_state(land.surface, st, FT)
    end
end

function vars_state(land::LandModel, st::GradientFlux, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
        surface::vars_state(land.surface, st, FT)
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
    land_init_aux!(land, land.surface, aux, geom)
end

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

function nodal_update_auxiliary_state!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, land.soil, state, aux, t)
    land_nodal_update_auxiliary_state!(land, land.surface, state, aux, t)
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

include("prog_types.jl")
include("RadiativeEnergyFlux.jl")
using .RadiativeEnergyFlux
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
include("SurfaceFlow.jl")
using .SurfaceFlow
include("soil_bc.jl")
include("prognostic_vars.jl")
include("land_tendencies.jl")
include("source.jl")

function wavespeed(m::LandModel, n⁻, state::Vars, aux::Vars, t::Real, direction)
    FT = eltype(state)
    h = max(state.surface.height, FT(0.0))
    v = calculate_velocity(m.surface, aux.x, aux.y, h)
    speed = norm(v)
    return speed
end

end # Module
