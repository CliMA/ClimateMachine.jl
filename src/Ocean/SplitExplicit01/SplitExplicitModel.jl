module SplitExplicit01

export OceanDGModel,
    OceanModel,
    Continuity3dModel,
    HorizontalModel,
    BarotropicModel,
#   basic_grid_info,
    AbstractOceanProblem

#using Printf
using StaticArrays
using LinearAlgebra: I, dot, Diagonal
#using CLIMAParameters.Planet: grav
using ..VariableTemplates
using ..MPIStateArrays
using ..DGMethods: init_ode_state
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids:
    polynomialorder, dimensionality, dofs_per_element,
    VerticalDirection, HorizontalDirection, min_node_distance

using ..BalanceLaws: BalanceLaw
import ..BalanceLaws: nodal_update_auxiliary_state!

using ..DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxFirstOrder

import ..DGMethods.NumericalFluxes:
    update_penalty!, numerical_flux_second_order!, NumericalFluxFirstOrder

import ..DGMethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    compute_gradient_argument!,
    init_state_auxiliary!,
    init_state_conservative!,
    LocalGeometry,
    DGModel,
    compute_gradient_flux!,
    copy_stack_field_down!,
#   create_state,
    calculate_dt,
    vars_integrals,
    vars_reverse_integrals,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

abstract type AbstractOceanModel <: BalanceLaw end
abstract type AbstractOceanProblem end

function ocean_init_aux! end
function ocean_init_state! end

function basic_grid_info(dg::DGModel)
    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)

    Nq = N + 1
    Nqk = dim == 2 ? 1 : Nq
    Nfp = Nq * Nqk
    Np = dofs_per_element(grid)

    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)
    nrealelem = length(topology.realelems)
    nhorzrealelem = div(nrealelem, nvertelem)

    return (Nq, Nqk, Nfp, Np, nvertelem, nhorzelem, nhorzrealelem, nrealelem)
end

include("OceanModel.jl")
include("Continuity3dModel.jl")
include("VerticalIntegralModel.jl")
include("BarotropicModel.jl")
include("Communication.jl")
include("OceanBoundaryConditions.jl")

end
