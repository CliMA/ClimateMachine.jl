module BalanceLaws

using ..VariableTemplates

export BalanceLaw,
    vars_state,
    number_states,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_flux!,
    compute_gradient_argument!,
    transform_post_gradient_laplacian!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    nodal_init_state_auxiliary!,
    nodal_update_auxiliary_state!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

include("state_types.jl")
include("interface.jl")

end
