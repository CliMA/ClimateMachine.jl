using MPI
using OrderedCollections
using StaticArrays
using OrdinaryDiffEq
using DiffEqBase

#  - load CLIMAParameters and set up to use it:

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates

# not boiler plate
using ClimateMachine.SingleStackUtils

#  - import necessary ClimateMachine modules: (`import`ing enables us to
#  provide implementations of these structs/methods)
import ClimateMachine.BalanceLaws:
    vars_state,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_update_auxiliary_state!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    boundary_state!

import ClimateMachine.DGMethods: calculate_dt


## Mappings that need to be performed

# this part
#=
T   = Field(Dict("field_type" => "prognostic"))    # Prognostic quantity
α∇T = Field(...)    # Diagnostic/auxiliary quantity

Base.@kwdef struct HeatModel{FT, APS} <: BalanceLaw
    "Parameters"
    param_set::APS
    "Thermal diffusivity"
    α::FT = 0.01
    "Initial conditions for temperature"
    initialT::FT = 295.15
end

pde_equation = [
    α∇T   == α * ∇(T), ## auxiliary equation
    ## RHS argument is gradient state
    ## LHS is gradient flux state
    ∂t(T) == -∇⋅(α∇T), ##  Actual PDE / "Balance law"
]
# Specify auxiliary variables for `HeatModel`
vars_state(::HeatModel, ::Auxiliary, FT) = @vars();
# Specify prognostic variables, the variables solved for in the PDEs, for
# `HeatModel`
vars_state(::HeatModel, ::Prognostic, FT) = @vars(T::FT);
# Specify state variables whose gradients are needed for `HeatModel`
vars_state(::HeatModel, ::Gradient, FT) = @vars(∇T::FT);
# Specify gradient variables for `HeatModel`
vars_state(::HeatModel, ::GradientFlux, FT) = @vars(α∇T::SVector{3, FT});

# this part
    α∇T   == α * ∇(T)
# maps to 
function compute_gradient_argument!(
    m::HeatModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.T = state.T
end;

function compute_gradient_flux!(
    m::HeatModel,
    diffusive::Vars,
    ∇G::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.α∇T = -m.α * ∇G.T
end;

# this 
∂t(T) == ∇⋅(α∇T), ##  Actual PDE / "Balance law"

# needs to map to 
function flux_second_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.T += diffusive.α∇T
end;

function source!(m::HeatModel, _...) end;
function flux_first_order!(m::HeatModel, _...) end;
function nodal_update_auxiliary_state!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end;

##
# diffusive courant boiler plate
# Next, we'll define our implementation of `diffusive_courant`:
function diffusive_courant(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    return Δt * m.α / (Δx * Δx)
end


function calculate_dt(dg, model::HeatModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end


function to_balance_law(model, pde_system, FT)
    # does this work? or does it have to be a macro?
    auxi = get_terms(pde_system, Dict("field_type" => ::Auxiliary))
    vars_state(model, ::Auxiliary, FT) = @vars(auxi...);
    progi = get_terms(pde_system, Dict("field_type" => ::Prognostic))
    vars_state(model, ::Prognostic, FT) = @vars(progi...);
    gradi = get_terms(pde_system, Dict("field_type" => ::Gradient))
    vars_state(model, ::Gradient, FT) = @vars(gradi...);
    gradient_fluxi = get_terms(pde_sytem, Dict("field_type" => ::GradientFlux))
    vars_state(model, ::GradientFlux, FT) = @vars(gradient_fluxi...);
    return vars_state
end

=#