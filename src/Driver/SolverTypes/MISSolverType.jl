
export MISSolverType

"""
"""
struct MISSolverType{DS} <: AbstractSolverType
    splitting_type::DS
    linear_model::Type
    mis_method::Function
    fast_method::Function
    nsubsteps::Int

    function MISSolverType(;
        splitting_type = SlowFastSplitting(),
        linear_model = AtmosAcousticGravityLinearModel,
        mis_method = MIS2,
        fast_method = LSRK54CarpenterKennedy,
        nsubsteps = 50,
    )

        DS = typeof(splitting_type)

        return new{DS}(
            splitting_type,
            linear_model,
            mis_method,
            fast_method,
            nsubsteps,
        )
    end
end

"""
"""
function solversetup(
    ode_solver::MISSolverType{DS},
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
        direction = EveryDirection(),
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

    solver = ode_solver.mis_method(
        slow_dg,
        fast_dg,
        ode_solver.fast_method,
        ode_solver.nsubsteps,
        Q;
        dt = dt,
        t0 = t0,
    )

    return solver
end
