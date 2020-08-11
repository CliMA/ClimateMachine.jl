module DGMethods

using MPI
using StaticArrays
using DocStringExtensions
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using ..MPIStateArrays
using ..Mesh.Grids
using ..Mesh.Topologies
using ..VariableTemplates
using ..Courant
using ..BalanceLaws:
    BalanceLaw,
    AbstractStateType,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux,
    GradientLaplacian,
    Hyperdiffusive,
    UpwardIntegrals,
    DownwardIntegrals,
    vars_state,
    number_states

import ..BalanceLaws:
    BalanceLaw,
    init_state_prognostic!,
    init_state_auxiliary!,
    flux_first_order!,
    flux_second_order!,
    compute_gradient_flux!,
    compute_gradient_argument!,
    source!,
    transform_post_gradient_laplacian!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    nodal_update_auxiliary_state!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

export DGModel,
    init_ode_state,
    restart_ode_state,
    restart_auxiliary_state,
    basic_grid_info,
    nodal_init_state_auxiliary!,
    contiguous_field_gradient!

include("NumericalFluxes.jl")
include("DGModel.jl")
include("DGModel_kernels.jl")
include("create_states.jl")

"""
    calculate_dt(dg, model, Q, Courant_number, direction, t)

For a given model, compute a time step satisying the nondiffusive Courant number
`Courant_number`
"""
function calculate_dt(dg, model, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = courant(nondiffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end

end
