
export MultirateSolverType

LSRK2N = LowStorageRungeKutta2N
ARK = AdditiveRungeKutta

"""
"""
struct MultirateSolverType{DS, SS, FS} <: AbstractSolverType
    splitting_type::DS
    linear_model::Type
    slow_method::SS
    fast_method::FS
    timestep_ratio::Int

    function MultirateSolverType(;
        splitting_type = SlowFastSplitting(),
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK54CarpenterKennedy,
        fast_method = LSRK54CarpenterKennedy,
        timestep_ratio = 100,
    )
        DS = typeof(splitting_type)
        SS = typeof(slow_method)
        FS = typeof(fast_method)

        return new{DS, SS, FS}(
            splitting_type,
            linear_model,
            slow_method,
            fast_method,
            timestep_ratio,
        )
    end
end

"""
"""
function solversetup(
    ode_solver::MultirateSolverType{DS, SS, FS},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) where {DS, SS, FS}
    solversetup(
        ode_solver.splitting_type,
        ode_solver,
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )
end

"""
"""
function solversetup(
    ::SlowFastSplitting,
    ode_solver::MultirateSolverType{DS, SS, FS},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) where {DS, SS, FS}

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