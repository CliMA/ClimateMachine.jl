module SystemSolvers

using ..MPIStateArrays
using ..MPIStateArrays: array_device, realview

using ..Mesh.Grids
import ..Mesh.Grids: polynomialorders, dimensionality
using ..Mesh.Topologies
using ..DGMethods
using ..DGMethods: DGModel, DGFVModel, SpaceDiscretization
import ..DGMethods.FVReconstructions: width
using ..BalanceLaws

using Adapt
using CUDA
using LinearAlgebra
using LazyArrays
using StaticArrays
using KernelAbstractions
using CUDAKernels

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
    AbstractNonlinearSolver

This is an abstract type representing a generic nonlinear solver.
"""
abstract type AbstractNonlinearSolver <: AbstractSystemSolver end

"""
    LSOnly

Only applies the linear solver (no Newton solver)
"""
struct LSOnly <: AbstractNonlinearSolver
    linearsolver::Any
end

function donewtoniteration!(
    rhs!,
    linearoperator!,
    preconditioner,
    Q,
    Qrhs,
    solver::LSOnly,
    args...,
)
    @info "donewtoniteration! linearsolve!", args...
    linearsolve!(
        linearoperator!,
        preconditioner,
        solver.linearsolver,
        Q,
        Qrhs,
        args...;
        max_iters = getmaxiterations(solver.linearsolver),
    )
end


"""

Solving rhs!(Q) = Qrhs via Newton,

where `F = rhs!(Q) - Qrhs`

dF/dQ(Q^n) ΔQ ≈ jvp!(ΔQ;  Q^n, F(Q^n))

preconditioner ≈ dF/dQ(Q)

"""
function nonlinearsolve!(
    rhs!,
    jvp!,
    preconditioner,
    solver::AbstractNonlinearSolver,
    Q::AT,
    Qrhs,
    args...;
    max_newton_iters = 10,
    cvg = Ref{Bool}(),
) where {AT}

    FT = eltype(Q)
    tol = solver.tol
    converged = false
    iters = 0

    if preconditioner === nothing
        preconditioner = NoPreconditioner()
    end

    # Initialize NLSolver, compute initial residual
    initial_residual_norm = initialize!(rhs!, Q, Qrhs, solver, args...)
    if initial_residual_norm < tol
        converged = true
    end
    converged && return iters


    while !converged && iters < max_newton_iters

        # dF/dQ(Q^n) ΔQ ≈ jvp!(ΔQ;  Q^n, F(Q^n)), update Q^n in jvp!
        update_Q!(jvp!, Q, args...)

        # update preconditioner based on finite difference, with jvp!
        preconditioner_update!(jvp!, rhs!.f!, preconditioner, args...)

        # do newton iteration with Q^{n+1} = Q^{n} - dF/dQ(Q^n)⁻¹ (rhs!(Q) - Qrhs)
        residual_norm, linear_iterations = donewtoniteration!(
            rhs!,
            jvp!,
            preconditioner,
            Q,
            Qrhs,
            solver,
            args...,
        )
        # @info "Linear solver converged in $linear_iterations iterations"
        iters += 1

        preconditioner_counter_update!(preconditioner)


        if !isfinite(residual_norm)
            error("norm of residual is not finite after $iters iterations of `donewtoniteration!`")
        end

        # Check residual_norm / norm(R0)
        # Comment: Should we check "correction" magitude?
        # ||Delta Q|| / ||Q|| ?
        relresidual = residual_norm / initial_residual_norm
        if relresidual < tol || residual_norm < tol
            # @info "Newton converged in $iters iterations!"
            converged = true
        end
    end

    converged ||
        @warn "Nonlinear solver did not converge after $iters iterations"
    cvg[] = converged

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

end
