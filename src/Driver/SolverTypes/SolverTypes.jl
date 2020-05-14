
export AbstractSolverType

"""
    AbstractSolverType

This is an abstract type representing a generic solver. By
a "solver," we mean an ODE solver together with any potential
implicit solver (linear solvers).
"""
abstract type AbstractSolverType end

"""
    solversetup(
        ode_solver::AbstractSolverType,
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

TODO: Fill out
"""
solversetup(
    ode_solver::AbstractSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) = throw(
    MethodError(
        solversetup,
        (ode_solver, dg, Q, dt, t0, diffusion_direction),
    )
)

include("ExplicitSolverType.jl")
include("IMEXSolverType.jl")
include("MultirateSolverType.jl")

DefaultSolverType = IMEXSolverType
export DefaultSolverType
