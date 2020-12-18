export IterativeSolver, solve!

"""
    IterativeAlgorithm

Abstract type for an iterative algorithm.
"""
abstract type IterativeAlgorithm end

"""
    IterativeSolver

Abstract type for an iterative solver.
"""
abstract type IterativeSolver end

"""
    IterativeSolver(
        algorithm::IterativeAlgorithm,
        problem::Problem
    )::IterativeSolver

Constructs the solver associated with the combination of `algorithm` and
`problem`.
"""
function IterativeSolver(
    algorithm::IterativeAlgorithm,
    problem::Problem,
)::IterativeSolver end

"""
    atol(solver::IterativeSolver)::AbstractFloat

Returns the absolute tolerance of `solver`.
"""
function atol(solver::IterativeSolver)::AbstractFloat end

"""
    rtol(solver::IterativeSolver)::AbstractFloat

Returns the relative tolerance of `solver`.
"""
function rtol(solver::IterativeSolver)::AbstractFloat end

"""
    maxiters(solver::IterativeSolver)::Int

Returns the maximum number of iterations that `solver` can take.
"""
function maxiters(solver::IterativeSolver)::Int end

"""
    initialize!(
        solver::IterativeSolver,
        problem::Problem,
        args...,
    )::AbstractFloat

Initializes `solver` and returns the norm of the residual.
"""
function initialize!(
    solver::IterativeSolver,
    problem::Problem,
    args...,
)::AbstractFloat end

"""
    doiteration!(
        solver::IterativeSolver,
        problem::Problem,
        threshold,
        iters,
        args...,
    )::AbstractFloat

Performs an iteration of `solver`, updates the solution vector in `problem`,
and returns whether the solver converged. Uses `threshold` and `iters` to check
for convergence.
"""
function doiteration!(
    solver::IterativeSolver,
    problem::Problem,
    threshold,
    iters,
    args...,
)::AbstractFloat end

"""
    solve!(solver::IterativeSolver, problem::Problem, args...)::Int

Iteratively solves `problem` with `solver` by repeatedly calling the function
`doiteration!`. Returns the number of iterations taken by `solver`.
"""
function solve!(solver::IterativeSolver, problem::Problem, args...)
    iters = 0

    initial_residual_norm = initialize!(solver, problem, args...)
    check_convergence(initial_residual_norm, atol(solver), iters) && return iters
    threshold = max(atol(solver), rtol(solver) * initial_residual_norm)

    has_converged = false
    m = maxiters(solver)
    while !has_converged && iters < m
        has_converged = doiteration!(solver, problem, threshold, iters, args...)
        println("$(typeof(solver).name), iteration $iters, $(problem.Q)")
        iters += 1
    end

    has_converged ||
        @warn "$(typeof(solver).name) did not converge after $iters iterations"
    return iters
end

# Utility function used by solve!() and doiteration!() that checks whether the
# solver has converged.
function check_convergence(residual_norm, threshold, iters)
    isfinite(residual_norm) ||
        error("Norm of residual is not finite after $iters iterations")
    return residual_norm < threshold
end

#= TODO: Uncomment when SystemSolvers.jl is removed.
# Kernel abstraction used by some iterative linear solvers that sets
# Q = Σ_i c_i X_i.
@kernel function linearcombination!(Q, cs, Xs, increment::Bool)
    i = @index(Global, Linear)
    if !increment
        @inbounds Q[i] = -zero(eltype(Q))
    end
    @inbounds for j in 1:length(cs)
        Q[i] += cs[j] * Xs[j][i]
    end
end
=#

include("GeneralizedMinimalResidualAlgorithm.jl")
include("JacobianFreeNewtonKrylovAlgorithm.jl")

# TODO:
#   - Figure out what solve!() needs to return
#       - Previously, nonlinear algorithms returned number of doiteration() calls, while linear algorithms returned number of linear operator evaluations
#       - Perhaps the latter is what we actually care about, so we should just return that.
#       - Discussion result: return (fcalls, iters)
#   - Consider where EulerOperator needs to be
#       - It should be explicitly dealt with in Preconditioners.jl and enable_duals.jl, but BackwardEulerSolvers.jl is included after those files.
#       - Has been commented out in BackwardEulerSolvers.jl and moved to Problem.jl as a temporary workaround.
#   - Consider eliminating Problem and passing its components as function arguments
#       - Since the user needs to pass different arrays Q, they get no benefit from constructing an immutable Problem around Q instead of passing Q directly.
#       - While function calls appear simpler with one Problem argument instead of three arguments, those functions will still be called with those three arguments, just wrapped in a Problem.
#       - So, really, introducing an immutable Problem just introduces clutter without any benefit.
#       - On the other hand, if Problem is made mutable, then the user could store both Q and args... in there, swapping them out when necessary. I don't think this makes things any simpler, though.
#       - Discussion result: remove Problem
#   - Deal with EulerOperator in Preconditioners.jl
#   - Get a reference for stepsize() computation in JacobianFreeNewtonKrylovSolver
#   - Check whether weighted_norm needs to be passed around everywhere
#   - Pass α for EulerOperator in args... to solve!()
#   - Rename JaCobIanfrEEneWtONKryLovSoLVeR with proper capitalization after removing jacobian_free_newton_krylov_solver.jl