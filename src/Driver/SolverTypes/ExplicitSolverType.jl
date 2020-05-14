
export ExplicitSolverType

"""
"""
struct ExplicitSolverType <: AbstractSolverType
    solver_method::Function
    linear_model::Type

    function ExplicitSolverType(;
        solver_method = LSRK54CarpenterKennedy,
        linear_model = nothing,
    )

        return new(solver_method, linear_model)
    end
end

"""
"""
function solversetup(
    ode_solver::ExplicitSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    solver = ode_solver.solver_method(
        dg,
        Q;
        dt = dt,
        t0 = t0,
    )
    return solver
end
