
"""
    AbstractLinearSolver

This is an abstract type representing a generic iterative
linear solver.
"""
abstract type AbstractLinearSolver <: AbstractSystemSolver end

abstract type AbstractLinearSolverCache end
abstract type AbstractKrylovMethod end

mutable struct LinearSolver{
    OP,
    krylovType <: AbstractKrylovMethod,
    pcType <: AbstractPreconditioner,
    fType,
    lscType <: AbstractLinearSolverCache,
} <: AbstractLinearSolver
    linop!::OP
    krylov_alg::krylovType
    pc::pcType
    rtol::fType
    atol::fType
    cache::lscType
end

function LinearSolver(
    linearoperator!,
    krylov_alg::AbstractKrylovMethod,
    pc_alg::AbstractPreconditionerType,
    sol::AT;
    max_iter = min(30, eltype(sol)),
    max_restart_iter = 10,
    rtol = √eps(eltype(AT)),
    atol = eps(eltype(AT)),
) where {AT}

    cache = cache(
        krylov_alg,
        sol,
        max_iter,
        max_restart_iter,
    )
    pc = Preconditioner(pc_alg, linearoperator!, sol)

    return LinearSolver(
        linearoperator!,
        krylov_alg,
        pc,
        rtol,
        atol,
        cache,
    )
end

function linearsolve!(
    linearsolver::LinearSolver,
    Q,
    Qrhs,
    args...;
    restart_iter = 30,
    cvg = Ref{Bool}(),
)
    iters = 0
    krylov_alg = linearsolver.krylov_alg
    linearoperator! = linearsolver.linop!

    converged, threshold = LSinitialize!(
        krylov_alg,
        linearsolver,
        linearoperator!,
        Q,
        Qrhs,
        args...,
    )

    converged && return iters

    while !converged && iters < max_iters
        converged, inner_iters, residual_norm =
            doiteration!(linearoperator!, Q, Qrhs, solver, threshold, args...)

        iters += inner_iters

        if !isfinite(residual_norm)
            error("norm of residual is not finite after $iters iterations of `doiteration!`")
        end

        achieved_tolerance = residual_norm / threshold * solver.rtol
    end

    converged || @warn "Linear solver did not attain convergence after $iters iterations"
    cvg[] = converged

    iters
end

@kernel function linearcombination!(Q, cs, Xs, increment::Bool)
    i = @index(Global, Linear)
    if !increment
        @inbounds Q[i] = -zero(eltype(Q))
    end
    @inbounds for j in 1:length(cs)
        Q[i] += cs[j] * Xs[j][i]
    end
end
