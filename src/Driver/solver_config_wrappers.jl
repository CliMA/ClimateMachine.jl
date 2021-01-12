import ..SingleStackUtils: single_stack_diagnostics, NodalStack
using ..ODESolvers

# Convenience wrapper
single_stack_diagnostics(solver_config; kwargs...) = single_stack_diagnostics(
    solver_config.dg.grid,
    solver_config.dg.balance_law,
    gettime(solver_config.solver),
    solver_config.dg.direction;
    prognostic = solver_config.Q,
    auxiliary = solver_config.dg.state_auxiliary,
    diffusive = solver_config.dg.state_gradient_flux,
    hyperdiffusive = solver_config.dg.states_higher_order[2],
    kwargs...,
)

# Convenience wrapper
NodalStack(solver_config; kwargs...) = NodalStack(
    solver_config.dg.balance_law,
    solver_config.dg.grid;
    prognostic = solver_config.Q,
    auxiliary = solver_config.dg.state_auxiliary,
    diffusive = solver_config.dg.state_gradient_flux,
    hyperdiffusive = solver_config.dg.states_higher_order[2],
    kwargs...,
)
