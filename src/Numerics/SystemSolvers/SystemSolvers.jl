module SystemSolvers

using ..MPIStateArrays
using ..MPIStateArrays: array_device, realview

using ..Mesh.Grids
import ..Mesh.Grids: polynomialorders, dimensionality
using ..Mesh.Topologies
using ..DGMethods
using ..DGMethods: DGModel
using ..BalanceLaws

using Adapt
using CUDA
using LinearAlgebra
using LazyArrays
using StaticArrays
using KernelAbstractions

const weighted_norm = false

# just for testing SystemSolvers
LinearAlgebra.norm(A::MVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::MVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::MVector, B::MVector, weighted) = dot(A, B)
LinearAlgebra.norm(A::AbstractVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::AbstractVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::AbstractVector, B::AbstractVector, weighted) = dot(A, B)

export linearsolve!,
    settolerance!, prefactorize, construct_preconditioner, preconditioner_solve!
export AbstractSystemSolver,
    AbstractIterativeSystemSolver, AbstractNonlinearSolver
export nonlinearsolve!

"""
    AbstractSystemSolver

This is an abstract type representing a generic linear solver.
"""
abstract type AbstractSystemSolver end

"""
    atol(::AbstractSystemSolver)
    rtol(::AbstractSystemSolver)
    maxiters(::AbstractSystemSolver)
"""
function atol end
function rtol end
function maxiters end

"""
    AbstractNonlinearSolver

This is an abstract type representing a generic nonlinear solver.
"""
abstract type AbstractNonlinearSolver <: AbstractSystemSolver end

"""
    initialize!(::AbstractNonlinearSolver, args...; kwargs...)
    dononlineariteration!(::AbstractNonlinearSolver, args...; kwargs...)
"""
function initialize! end
function dononlineariteration! end

function check_convergence(residual_norm, threshold, iters)
    if !isfinite(residual_norm)
        error("norm of residual is not finite after $iters iterations")
    end

    # Check residual_norm / norm(R0)
    # Comment: Should we check "correction" magitude?
    # ||Delta Q|| / ||Q|| ?
    return residual_norm < threshold
end

"""
    function nonlinearsolve!(
        solver::AbstractNonlinearSolver,
        rhs!,
        Q::AT,
        Qrhs,
        args...;
        kwargs...
    ) where {AT}

Solving rhs!(Q) = Qrhs via a nonlinear solver.
"""
function nonlinearsolve!(
    solver::AbstractNonlinearSolver,
    rhs!,
    Q::AT,
    Qrhs,
    args...;
    kwargs...
) where {AT}
    FT = eltype(Q)
    iters = 0

    # Initialize NLSolver, compute the threshold
    initial_residual_norm = initialize!(rhs!, Q, Qrhs, solver, args...)
    initial_residual_norm < atol(solver) && return iters
    threshold = max(atol(solver), rtol(solver) * initial_residual_norm)

    converged = false
    m = maxiters(solver)
    while !converged && iters < m
        converged = dononlineariteration!(
            solver,
            rhs!,
            Q,
            Qrhs,
            threshold,
            iters,
            args...;
            kwargs...
        )
        iters += 1
    end

    converged ||
        @warn "Nonlinear solver did not converge after $iters iterations"

    iters
end

"""
    AbstractIterativeSystemSolver

This is an abstract type representing a generic iterative
linear solver.

The available concrete implementations are:

  - [`GeneralizedConjugateResidual`](@ref)
  - [`GeneralizedMinimalResidual`](@ref)
"""
abstract type AbstractIterativeSystemSolver <: AbstractSystemSolver end

"""
    settolerance!(solver::AbstractIterativeSystemSolver, tolerance, relative)

Sets the relative or absolute tolerance of the iterative linear solver
`solver` to `tolerance`.
"""
settolerance!(
    solver::AbstractIterativeSystemSolver,
    tolerance,
    relative = true,
) = (relative ? (solver.rtol = tolerance) : (solver.atol = tolerance))

doiteration!(
    linearoperator!,
    preconditioner,
    Q,
    Qrhs,
    solver::AbstractIterativeSystemSolver,
    threshold,
    args...,
) = throw(MethodError(
    doiteration!,
    (linearoperator!, preconditioner, Q, Qrhs, solver, tolerance, args...),
))

initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::AbstractIterativeSystemSolver,
    args...,
) = throw(MethodError(initialize!, (linearoperator!, Q, Qrhs, solver, args...)))

"""
    prefactorize(linop!, linearsolver, args...)

Prefactorize the in-place linear operator `linop!` for use with `linearsolver`.
"""
function prefactorize(
    linop!,
    linearsolver::AbstractIterativeSystemSolver,
    args...,
)
    return nothing
end

"""
    linearsolve!(linearoperator!, solver::AbstractIterativeSystemSolver, Q, Qrhs, args...)

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
    preconditioner,
    solver::AbstractIterativeSystemSolver,
    Q,
    Qrhs,
    args...;
    max_iters = length(Q),
    cvg = Ref{Bool}(),
)
    converged = false
    iters = 0

    if preconditioner === nothing
        preconditioner = NoPreconditioner()
    end

    converged, threshold =
        initialize!(linearoperator!, Q, Qrhs, solver, args...)
    converged && return iters

    while !converged && iters < max_iters
        converged, inner_iters, residual_norm = doiteration!(
            linearoperator!,
            preconditioner,
            Q,
            Qrhs,
            solver,
            threshold,
            args...,
        )

        iters += inner_iters

        if !isfinite(residual_norm)
            error("norm of residual is not finite after $iters iterations of `doiteration!`")
        end

        achieved_tolerance = residual_norm / threshold * solver.rtol
    end

    converged ||
        @warn "Solver did not attain convergence after $iters iterations"
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

include("generalized_minimal_residual_solver.jl")
include("generalized_conjugate_residual_solver.jl")
include("conjugate_gradient_solver.jl")
include("columnwise_lu_solver.jl")
include("preconditioners.jl")
include("batched_generalized_minimal_residual_solver.jl")
include("jacobian_free_newton_krylov_solver.jl")
include("accelerators.jl")
include("picard_solver.jl")

end
