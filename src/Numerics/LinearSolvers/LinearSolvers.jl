module LinearSolvers

using ..MPIStateArrays

using StaticArrays, LinearAlgebra

using KernelAbstractions

# just for testing LinearSolvers
LinearAlgebra.norm(A::MVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::MVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::MVector, B::MVector, weighted) = dot(A, B)
LinearAlgebra.norm(A::AbstractVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::AbstractVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::AbstractVector, B::AbstractVector, weighted) = dot(A, B)

export linearsolve!, settolerance!, prefactorize
export AbstractLinearSolver, AbstractIterativeLinearSolver

"""
    AbstractLinearSolver

This is an abstract type representing a generic linear solver.
"""
abstract type AbstractLinearSolver end

"""
    AbstractIterativeLinearSolver

This is an abstract type representing a generic iterative
linear solver.

The available concrete implementations are:

  - [`GeneralizedConjugateResidual`](@ref)
  - [`GeneralizedMinimalResidual`](@ref)
"""
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

"""
    settolerance!(solver::AbstractIterativeLinearSolver, tolerance, relative)

Sets the relative or absolute tolerance of the iterative linear solver
`solver` to `tolerance`.
"""
settolerance!(
    solver::AbstractIterativeLinearSolver,
    tolerance,
    relative = true,
) = (relative ? (solver.rtol = tolerance) : (solver.atol = tolerance))

doiteration!(
    linearoperator!,
    Q,
    Qrhs,
    solver::AbstractIterativeLinearSolver,
    tolerance,
    args...,
) = throw(MethodError(
    doiteration!,
    (linearoperator!, Q, Qrhs, solver, tolerance, args...),
))

initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::AbstractIterativeLinearSolver,
    args...,
) = throw(MethodError(initialize!, (linearoperator!, Q, Qrhs, solver, args...)))

"""
    prefactorize(linop!, linearsolver, args...)

Prefactorize the in-place linear operator `linop!` for use with `linearsolver`.
"""
prefactorize(linop!, linearsolver::AbstractIterativeLinearSolver, args...) =
    linop!

"""
    linearsolve!(linearoperator!, solver::AbstractIterativeLinearSolver, Q, Qrhs, args...)

Solves a linear problem defined by the `linearoperator!` function and the state
`Qrhs`, i.e,

```math
L(Q) = Q_{rhs}
```

using the `solver` and the initial guess `Q`. After the call `Q` contains the
solution.  The arguments `args` is passed to `linearoperator!` when it is
called.
"""
function linearsolve!(
    linearoperator!,
    solver::AbstractIterativeLinearSolver,
    Q,
    Qrhs,
    args...;
    max_iters = length(Q),
    cvg = Ref{Bool}(),
)
    converged = false
    iters = 0

    converged, threshold =
        initialize!(linearoperator!, Q, Qrhs, solver, args...)
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

    converged || @warn "Solver did not attain convergence after $iters iterations"
    cvg[] = converged

    @show iters
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

end
