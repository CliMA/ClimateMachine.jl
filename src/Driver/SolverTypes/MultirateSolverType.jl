
export MultirateSolverType

"""
"""
struct MultirateSolverType{DS} <: AbstractSolverType
    splitting_type::DS
    linear_model::Type
    implicit_solver::Type
    implicit_solver_adjustable::Bool
    slow_method::Function
    fast_method::Function
    timestep_ratio::Int

    function MultirateSolverType(;
        splitting_type = SlowFastSplitting(),
        linear_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        implicit_solver_adjustable = false,
        slow_method = LSRK54CarpenterKennedy,
        fast_method = LSRK54CarpenterKennedy,
        timestep_ratio = 100,
    )

        DS = typeof(splitting_type)

        return new{DS}(
            splitting_type,
            linear_model,
            implicit_solver,
            implicit_solver_adjustable,
            slow_method,
            fast_method,
            timestep_ratio,
        )
    end
end

"""
"""
function solversetup(
    ode_solver::MultirateSolverType{DS},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) where {DS <: SlowFastSplitting}

    linmodel = ode_solver.linear_model(dg.balance_law)

    fast_dg = DGModel(
        linmodel,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
    )

    slow_model = RemainderModel(dg.balance_law, (linmodel,))

    slow_dg = DGModel(
        slow_model,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        diffusion_direction = diffusion_direction,
    )

    slow_solver = ode_solver.slow_method(slow_dg, Q; dt = dt)
    fast_dt = dt / ode_solver.timestep_ratio
    fast_solver = ode_solver.fast_method(fast_dg, Q; dt = fast_dt)

    solver = MultirateRungeKutta(
        (slow_solver, fast_solver),
        t0 = t0
    )

    return solver
end

"""
"""
function solversetup(
    ode_solver::MultirateSolverType{DS},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) where {DS <: HEVISplitting}

    linmodel = ode_solver.linear_model(dg.balance_law)

    acoustic_dg_full = DGModel(
        linmodel,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = EveryDirection(),
    )

    acoustic_dg_vert = DGModel(
        linmodel,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = VerticalDirection(),
    )

    fast_dt = dt / ode_solver.timestep_ratio

    fast_solver = ode_solver.fast_method(
        acoustic_dg_full,
        acoustic_dg_vert,
        LinearBackwardEulerSolver(
            ode_solver.implicit_solver();
            isadjustable = ode_solver.implicit_solver_adjustable,
        ),
        Q;
        dt = fast_dt,
        t0 = t0,
        split_explicit_implicit = false,
        variant = LowStorageVariant(),
    )

    slow_model = RemainderModel(dg.balance_law, (linmodel,))

    slow_dg = DGModel(
        slow_model,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        diffusion_direction = diffusion_direction,
    )

    slow_solver = ode_solver.slow_method(slow_dg, Q; dt = dt, t0 = t0)

    solver = MultirateRungeKutta(
        (slow_solver, fast_solver),
        t0 = t0,
    )

    return solver
end
