
export ImplicitSolverType

"""
# Description
    ImplicitSolverType(;
        solver_method = KenCarp4,
    )

This solver type constructs an ODE solver using a _fully implicit_
method.

# Arguments
- `solver_method` (Function): Function defining the implicit
    solver.
    Default: `KenCarp4`
"""
struct ImplicitSolverType <: AbstractSolverType
    # Function for an implicit method
    solver_method::Function

    function ImplicitSolverType(
        alg;
        solver_method = ODESolvers.DiffEqJLConstructor(alg),
    )

        return new(solver_method)
    end
end

"""
    getdtmodel(ode_solver::AbstractSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(ode_solver::ImplicitSolverType, bl)
    # For implicit methods, the entire model itself
    # contributes to the total stability of the time-integrator
    return bl
end


"""
# Description
    function solversetup(
        ode_solver::ImplicitSolverType,
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

Creates an implicit ODE solver.
"""
function solversetup(
    ode_solver::ImplicitSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    solver = ode_solver.solver_method(dg, Q; dt = dt, t0 = t0)

    return solver
end
