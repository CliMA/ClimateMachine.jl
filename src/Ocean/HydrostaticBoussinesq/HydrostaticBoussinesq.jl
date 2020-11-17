module HydrostaticBoussinesq

export HydrostaticBoussinesqModel, Forcing

using StaticArrays
using LinearAlgebra: dot, Diagonal
using CLIMAParameters.Planet: grav

using ..Ocean
using ...VariableTemplates
using ...MPIStateArrays
using ...Mesh.Filters: apply!
using ...Mesh.Grids: VerticalDirection
using ...Mesh.Geometry
using ...DGMethods
using ...DGMethods: init_state_auxiliary!
using ...DGMethods.NumericalFluxes
using ...DGMethods.NumericalFluxes: RusanovNumericalFlux
using ...BalanceLaws

import ..Ocean: coriolis_parameter
import ...DGMethods.NumericalFluxes: update_penalty!
import ...BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!
import ..Ocean: ocean_init_state!, ocean_init_aux!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

include("hydrostatic_boussinesq_model.jl")
include("bc_velocity.jl")
include("bc_temperature.jl")
include("LinearHBModel.jl")
include("Courant.jl")

end
