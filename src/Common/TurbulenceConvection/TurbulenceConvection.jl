"""
    TurbulenceConvection

Turbulence convection models, for example
the Eddy-Diffusivity Mass-Flux model
"""
module TurbulenceConvection

using ..BalanceLaws: BalanceLaw, AbstractStateType
using ..VariableTemplates: @vars, Vars, Grad

export TurbulenceConvectionModel, NoTurbConv

export init_aux_turbconv!, turbconv_nodal_update_auxiliary_state!

import ..BalanceLaws:
    vars_state,
    init_state_auxiliary!,
    update_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!

using ..MPIStateArrays: MPIStateArray
using ..DGMethods: DGModel, LocalGeometry

abstract type TurbulenceConvectionModel end

"""
    NoTurbConv <: TurbulenceConvectionModel

A "no model" type, which results in kernels that
pass through and do nothing.
"""
struct NoTurbConv <: TurbulenceConvectionModel end

vars_state(m::TurbulenceConvectionModel, ::AbstractStateType, FT) = @vars()

function init_aux_turbconv!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    aux::Vars,
    geom::LocalGeometry,
)
    return nothing
end

function update_auxiliary_state!(
    dg::DGModel,
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return nothing
end

function turbconv_nodal_update_auxiliary_state!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end

function flux_first_order!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end

function compute_gradient_argument!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end

function compute_gradient_flux!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end

function flux_second_order!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
)
    return nothing
end

function integral_load_auxiliary_state!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    return nothing
end

function integral_set_auxiliary_state!(
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    aux::Vars,
    integ::Vars,
)
    return nothing
end

include("boundary_conditions.jl")
include("source.jl")

end
