module SplitExplicit01

export OceanDGModel,
    OceanModel,
    Continuity3dModel,
    HorizontalModel,
    BarotropicModel,
    AbstractOceanProblem

#using Printf
using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using CLIMAParameters.Planet: grav

using ...VariableTemplates
using ...MPIStateArrays
using ...DGMethods: init_ode_state, basic_grid_info
using ...Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ...Mesh.Grids:
    polynomialorder,
    dimensionality,
    dofs_per_element,
    VerticalDirection,
    HorizontalDirection,
    min_node_distance

using ...BalanceLaws
#import ...BalanceLaws: nodal_update_auxiliary_state!

using ...DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxFirstOrder

using ..Ocean

import ..Ocean: ocean_init_aux!, ocean_init_state!, ocean_boundary_state!

import ...DGMethods.NumericalFluxes:
    update_penalty!, numerical_flux_second_order!, NumericalFluxFirstOrder

import ...DGMethods:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    compute_gradient_argument!,
    init_state_auxiliary!,
    init_state_prognostic!,
    LocalGeometry,
    DGModel,
    compute_gradient_flux!,
    calculate_dt,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ...SystemSolvers: BatchedGeneralizedMinimalResidual, linearsolve!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

function initialize_fast_state! end
function initialize_adjustment! end

include("SplitExplicitLSRK2nMethod.jl")
include("OceanModel.jl")
include("Continuity3dModel.jl")
include("VerticalIntegralModel.jl")
include("BarotropicModel.jl")
include("IVDCModel.jl")
include("Communication.jl")
include("OceanBoundaryConditions.jl")

end
