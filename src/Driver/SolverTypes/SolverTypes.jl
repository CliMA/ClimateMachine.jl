
export AbstractSolverType
export DiscreteSplittingType, HEVISplitting

"""
    AbstractSolverType

This is an abstract type representing a generic solver. By
a "solver," we mean an ODE solver together with any potential
implicit solver (linear solvers).
"""
abstract type AbstractSolverType end

"""
    DiscreteSplittingType

This is an abstract type representing a temporal splitting
in the discrete equations. For example, HEVI
(horizontally explicit, vertically implicit) type methods.
"""
abstract type DiscreteSplittingType end

"""
    SlowFastSplitting

The tendency is treated using a standard slow-fast splitting,
where the fast processes are purely related to acoustic/gravity
waves (ideal for a typical multirate or MIS method).
This splitting does not take into account the geometric
direction (vertical vs horizontal).
"""
struct SlowFastSplitting <: DiscreteSplittingType end

"""
    HEVISplitting

HEVI (horizontally explicit, vertically implicit) type method,
where vertical acoustic waves are treated implicitly. All other
dynamics are treated explicitly.

Note: Can potentially imagine several different types of
HEVI splittings (for example, include vertical momentum and/or
diffusion)
"""
struct HEVISplitting <: DiscreteSplittingType end

"""
    getdtmodel(ode_solver::AbstractSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
getdtmodel(ode_solver::AbstractSolverType, bl) =
    throw(MethodError(solversetup, (ode_solver, bl)),)

"""
    solversetup(
        ode_solver::AbstractSolverType,
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

A function which returns a configured ODE solver.
"""
solversetup(
    ode_solver::AbstractSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) = throw(MethodError(
    solversetup,
    (ode_solver, dg, Q, dt, t0, diffusion_direction),
))

include("ExplicitSolverType.jl")
include("IMEXSolverType.jl")
include("MultirateSolverType.jl")
include("MISSolverType.jl")

DefaultSolverType = IMEXSolverType
export DefaultSolverType
