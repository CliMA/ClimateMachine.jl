
export SplitExplicitSolverType

"""
# Description
    SplitExplicitSolverType

This solver type constructs an ODE solver using the SplitExplicitLSRK2nSolver.
# Arguments
- `dt_slow` (AbstractFloat): Time step for the slow solver
- `dt_fast` (AbstractFloat): Time step for the fast solver
- `slow_method` (Function): Function defining the explicit
    Runge-Kutta solver for the slow model.
    Default: `LSRK54CarpenterKennedy`
- `fast_method` (Function): Function defining the explicit
    Runge-Kutta solver for the fast model.
    Default: `LSRK54CarpenterKennedy`
"""
struct SplitExplicitSolverType{SM, FM, FT} <: AbstractSolverType
    # Function for an explicit Runge-Kutta method
    slow_method::SM
    # Function for an explicit Runge-Kutta method
    fast_method::FM
    # time step for slow method
    dt_slow::FT
    # time step for fast method
    dt_fast::FT
    # parameter for super timestepping 
    add_fast_steps::FT
    # number of implicit solves per step
    numImplSteps::FT

    function SplitExplicitSolverType{FT}(
        dt_slow,
        dt_fast;
        add_fast_steps = 2,
        numImplSteps = 5,
        slow_method = LSRK54CarpenterKennedy,
        fast_method = LSRK54CarpenterKennedy,
    ) where {FT <: AbstractFloat}
        SM = typeof(slow_method)
        FM = typeof(fast_method)

        return new{SM, FM, FT}(
            slow_method,
            fast_method,
            dt_slow,
            dt_fast,
            add_fast_steps,
            numImplSteps,
        )
    end
end

"""
    getdtmodel(ode_solver::AbstractSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(::SplitExplicitSolverType, bl)
    # For explicit methods, the entire model itself
    # contributes to the total stability of the time-integrator
    return bl
end

"""
# Description
    function solversetup(
        ode_solver::ExplicitSolverType,
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

Creates an explicit ODE solver.
"""
function solversetup(ode_solver::SplitExplicitSolverType, dg_3D, Q_3D, _, t0, _)
    dg_2D = dg_3D.modeldata.dg_2D
    Q_2D = dg_3D.modeldata.Q_2D
    ivdc_model = dg_3D.modeldata.ivdc_dg.balance_law

    fast_solver =
        ode_solver.fast_method(dg_2D, Q_2D, dt = ode_solver.dt_fast, t0 = t0)
    slow_solver =
        ode_solver.slow_method(dg_3D, Q_3D, dt = ode_solver.dt_slow, t0 = t0)

    solver = SplitExplicitLSRK2nSolver(
        slow_solver,
        fast_solver;
        add_fast_steps = ode_solver.add_fast_steps,
        numImplSteps = ode_solver.numImplSteps,
        ivdc_dt = ivdc_model.ivdc_dt,
    )

    return solver
end
