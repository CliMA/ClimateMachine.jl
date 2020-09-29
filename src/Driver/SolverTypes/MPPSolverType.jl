export MPPSolverType

struct MPPSolverType{F, PV} <: AbstractSolverType
    dg_solver::F
    positive_variables::PV

    MPPSolverType(dg_solver::F, positive_variables::PV) where {F, PV} =
        new{F, PV}(dg_solver, positive_variables)
end

function getdtmodel(::MPPSolverType, bl)
    # For explicit methods, the entire model itself
    # contributes to the total stability of the time-integrator
    return bl
end

function solversetup(
    ode_solver::MPPSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    solver = MPPSolver(
        ode_solver.positive_variables,
        ode_solver.dg_solver,
        dg,
        Q;
        dt = dt,
        t0 = t0,
    )

    return solver
end
