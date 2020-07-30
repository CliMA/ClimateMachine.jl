
"""
    AbstractLinearSolver

This is an abstract type representing a generic iterative
linear solver.
"""
abstract type AbstractLinearSolver <: AbstractSystemSolver end

abstract type AbstractLinearSolverCache end
abstract type AbstractKrylovMethod end

mutable struct LinearSolver{
    AT,
    OP,
    krylovType <: AbstractKrylovMethod,
    pcType <: AbstractPreconditioner,
    fType,
    lscType <: AbstractLinearSolverCache,
} <: AbstractLinearSolver
    x::AT
    linop!::OP
    krylov_alg::krylovType
    pc::pcType
    rtol::fType
    atol::fType
    cache::lscType
end

function LScreate(
    linearoperator!,
    krylov_alg::AbstractKrylovMethod,
    pc_alg::AbstractPreconditioner,
    sol::AT;
    max_iter = min(30, eltype(sol)),
    rtol = √eps(eltype(AT)),
    atol = eps(eltype(AT)),
) where {AT}

    cache = cache(krylov_alg, sol, max_iter)
    pc = PCcreate(pc_alg, linearoperator!, sol)

    return LinearSolver(
        sol,
        linearoperator!,
        krylov_alg,
        pc,
        rtol,
        atol,
        cache,
    )
end

function LSinitialize!(
    linsolver::LinearSolver,
    args...,
)

    PCinitialize!(linsolver.pc, args...)

end
