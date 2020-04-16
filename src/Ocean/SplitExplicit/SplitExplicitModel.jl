module SplitExplicit

export OceanDGModel,
    OceanModel,
    HorizontalModel,
    BarotropicModel,
    LinearVerticalModel,
    AbstractOceanProblem

using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using ..VariableTemplates
using ..MPIStateArrays
using ..DGmethods: init_ode_state
using CLIMAParameters.Planet: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids:
    polynomialorder, VerticalDirection, HorizontalDirection, min_node_distance

using ..DGmethods.NumericalFluxes:
    Rusanov,
    CentralNumericalFluxGradient,
    CentralNumericalFluxDiffusive,
    CentralNumericalFluxNonDiffusive

import ..DGmethods.NumericalFluxes:
    update_penalty!, numerical_flux_diffusive!, NumericalFluxNonDiffusive

import ..DGmethods:
    BalanceLaw,
    vars_aux,
    vars_state,
    vars_gradient,
    vars_diffusive,
    flux_nondiffusive!,
    flux_diffusive!,
    source!,
    wavespeed,
    boundary_state!,
    update_aux!,
    update_aux_diffusive!,
    gradvariables!,
    init_aux!,
    init_state!,
    LocalGeometry,
    DGModel,
    nodal_update_aux!,
    diffusive!,
    copy_stack_field_down!,
    create_state,
    calculate_dt,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_aux!,
    integral_set_aux!,
    reverse_integral_load_aux!,
    reverse_integral_set_aux!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

abstract type AbstractOceanModel <: BalanceLaw end
abstract type AbstractOceanProblem end

function ocean_init_aux! end
function ocean_init_state! end

include("OceanModel.jl")
include("TendencyIntegralModel.jl")
include("HorizontalModel.jl")
include("BarotropicModel.jl")
include("LinearVerticalModel.jl")
include("Communication.jl")
include("BoundaryConditions.jl")

end
