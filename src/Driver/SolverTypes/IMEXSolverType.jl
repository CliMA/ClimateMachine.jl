
export IMEXSolverType

"""
"""
struct IMEXSolverType <: AbstractSolverType
    linear_model::Type
    linear_solver::Type
    linear_solver_adjustable::Bool
    solver_method::Function
    solver_storage_variant::Type

    function IMEXSolverType(;
        linear_model = AtmosAcousticGravityLinearModel,
        linear_solver = ManyColumnLU,
        linear_solver_adjustable = false,
        solver_method = ARK2GiraldoKellyConstantinescu,
        solver_storage_variant = LowStorageVariant(),
    )

        return new(
            linear_model,
            linear_solver,
            linear_solver_adjustable,
            solver_method,
            solver_storage_variant,
        )
    end
end

"""
"""
function solversetup(
    ode_solver::IMEXSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    vdg = DGModel(
        ode_solver.linear_model(dg.balance_law),
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = VerticalDirection(),
    )

    solver = ode_solver.solver_method(
        dg,
        vdg,
        LinearBackwardEulerSolver(
            ode_solver.linear_solver();
            isadjustable = ode_solver.linear_solver_adjustable,
        ),
        Q;
        dt = dt,
        t0 = t0,
        split_explicit_implicit = false,
        variant = ode_solver.solver_storage_variant,
    )
    return solver
end