export IterativeSolver, solve!

"""
    IterativeAlgorithm

Abstract type for an iterative algorithm.
"""
abstract type IterativeAlgorithm end

"""
    KrylovAlgorithm

Abstract type for a Krylov subspace method.
"""
abstract type KrylovAlgorithm <: IterativeAlgorithm end

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
        threshold,
        iters,
        problem::Problem,
        args...,
    )::AbstractFloat

Initializes `solver` and returns the norm of the residual, whether the solver
converged, and the number of times `f` was evaluated. Uses `threshold` and
`iters` to check for convergence by calling `check_convergence`.
"""
function initialize!(
    solver::IterativeSolver,
    problem::Problem,
    args...,
)::AbstractFloat end

"""
    doiteration!(
        solver::IterativeSolver,
        threshold,
        iters,
        problem::Problem,
        args...,
    )::AbstractFloat

Performs an iteration of `solver`, updates the solution vector in `problem`,
and returns whether the solver converged and the number of times `f` was
evaluated. Uses `threshold` and `iters` to check for convergence by calling
`check_convergence`.
"""
function doiteration!(
    solver::IterativeSolver,
    threshold,
    iters,
    problem::Problem,
    args...,
)::AbstractFloat end

"""
    solve!(solver::IterativeSolver, problem::Problem, args...)::Int

Iteratively solves `problem` with `solver` by repeatedly calling the function
`doiteration!`. Returns the number of iterations taken by `solver` and the
number of times `solver` called `f`.
"""
function solve!(solver::IterativeSolver, problem::Problem, args...)
    iters = 0

    initial_residual_norm, has_converged, fcalls =
        initialize!(solver, atol(solver), iters, problem, args...)
    has_converged && return (iters, fcalls)
    threshold = max(atol(solver), rtol(solver) * initial_residual_norm)

    while !has_converged && iters < maxiters(solver)
        has_converged, newfcalls =
            doiteration!(solver, threshold, iters, problem, args...)
        iters += 1
        fcalls += newfcalls
    end

    has_converged ||
        @warn "$(typeof(solver).name) did not converge after $iters iterations"
    return (iters, fcalls)
end

# Function used by solve!() and doiteration!() that checks whether the solver
# has converged.
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

# Macro used by algorithm constructors that checks whether the keyword
# arguments specified by the user are all positive.
macro check_positive(args...)
    n = length(args)
    block = Expr(:block)
    block.args = Array{Any}(undef, n)
    for i in 1:n
        arg = args[i]
        message = "Keyword argument $arg must be positive, but it was set to "
        block.args[i] = quote
            if !isnothing($arg) && !($arg > 0)
                throw(AssertionError(string($message, $arg)))
            end
        end
    end
    return block
end

# Macro used by algorithm constructors that checks whether the keyword
# arguments specified by the user are all finite.
macro check_finite(args...)
    n = length(args)
    block = Expr(:block)
    block.args = Array{Any}(undef, n)
    for i in 1:n
        arg = args[i]
        message = "Keyword argument $arg must be finite, but it was set to "
        block.args[i] = quote
            if !isnothing($arg) && !isfinite($arg)
                throw(AssertionError(string($message, $arg)))
            end
        end
    end
    return block
end

include("GeneralizedMinimalResidualAlgorithm.jl")
include("BatchedGeneralizedMinimalResidualAlgorithm.jl")
include("JacobianFreeNewtonKrylovAlgorithm.jl")

# TODO:
#   - Our stopping condition is incorrect!
#       - By taking the max of atol and rtol * initial_residual_norm, we are using the looser condition, ensuring that the tighter condition is not satisfied upon termination; we should use min instead.
#           - See the last paragraph in http://www.math.uakron.edu/~kreider/num1/stopcrit.pdf.
#           - If we want to ensure that rtol * initial_residual_norm is greater than eps(FT), we should take the max of eps(FT) and rtol * initial_residual_norm, but that is not the job of atol.
#           - Rather, atol specifies the maximum absolute error that the result can have. In Mathematica, the only way to disable atol is to set it to infinity, but in our code we could also set it to 0.
#           - Should we even use eps(FT) = eps(one(FT)) = nextfloat(one(FT)) - one(FT)? If Q has norm n, maybe we want to use eps(n)? Or, if Q has minimum (in magnitude) element m, we could use eps(m)?
#       - We should also consider using residual_norm instead of initial_residual norm for the relative tolerance condition, like some other authors do.
#       - We are also planning to add a new condition based on Q instead of f(Q), so this part of the code will have to change anyway.
#       - There was also a bug in the original GMRES initialize!() related to the stopping criteria, which has now been fixed.
#           - It occasionally caused GMRES to terminate before reaching the stopping criteria, putting Newton's method into an infinite loop (until it reached max_newton_iters).
#   - Consider where EulerOperator needs to be
#       - It should be explicitly dealt with in Preconditioners.jl and enable_duals.jl, but BackwardEulerSolvers.jl is included after those files.
#       - Has been commented out in BackwardEulerSolvers.jl and moved to Problem.jl as a temporary workaround.
#   - Consider eliminating Problem and passing its components as function arguments
#       - Since the user needs to pass different arrays Q, they get no benefit from constructing an immutable Problem around Q instead of passing Q directly.
#       - While function calls appear simpler with one Problem argument instead of three arguments, those functions will still be called with those three arguments, just wrapped in a Problem.
#       - So, really, introducing an immutable Problem just introduces clutter without any benefit.
#       - On the other hand, if Problem is made mutable, then the user could store both Q and args... in there, swapping them out when necessary. I don't think this makes things any simpler, though.
#       - Discussion result: remove Problem; maybe ask Simon first
#   - Get a reference for stepsize() computation in JacobianFreeNewtonKrylovSolver
#   - Check whether weighted_norm needs to be passed around everywhere
#   - Pass α for EulerOperator in args... to solve!()
#   - Rename JaCobIanfrEEneWtONKryLovSoLVeR with proper capitalization after removing jacobian_free_newton_krylov_solver.jl
#   - If we want other linear solvers and preconditioners, we should check out https://arxiv.org/pdf/1607.00351.pdf (may help solve stiffer nonlinear problems)
#       - "For symmetric systems, conjugate gradient (CG) and MINRES are widely recognized as the best [Krylov] methods. However, the situation is far less clear for nonsymmetric systems."
#       - Detailed comparison of Krylov methods (GMRES, TFQMR, BiCGSTAB and QMRCGSTAB) and general preconditioners (Gauss-Seidel, incomplete LU factorization, and algebraic multigrid).
#       - "GMRES tends to deliver better performance when coupled with an effective multigrid preconditioner, but it is less competitive with an ineffective preconditioner due to restarts."
#       - "Right preconditioning is, in general, more reliable than left preconditioning for large-scale systems."