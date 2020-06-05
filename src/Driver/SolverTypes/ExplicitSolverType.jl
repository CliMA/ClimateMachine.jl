
export ExplicitSolverType

"""
# Description
    ExplicitSolverType(;
        solver_method = LSRK54CarpenterKennedy,
    )

This solver type constructs an ODE solver using an explicit
Runge-Kutta method.

# Arguments
- `solver_method` (Function): Function defining the explicit
    Runge-Kutta solver.
    Default: `LSRK54CarpenterKennedy`
"""
struct ExplicitSolverType <: AbstractSolverType
    # Function for an explicit Runge-Kutta method
    solver_method::Function

    function ExplicitSolverType(; solver_method = LSRK54CarpenterKennedy)

        return new(solver_method)
    end
end

"""
    getdtmodel(ode_solver::AbstractSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(::ExplicitSolverType, bl)
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
function solversetup(
    ode_solver::ExplicitSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    solver = ode_solver.solver_method(dg, Q; dt = dt, t0 = t0)

    return solver
end
