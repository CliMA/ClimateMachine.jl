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
    IterativeSolver(algorithm::IterativeAlgorithm, args...)

Constructs the solver associated with `algorithm`.

`args...` specifies the format of the problem to be solved,
dependent on `algorithm`.
"""
function IterativeSolver(algorithm::IterativeAlgorithm, args...) end

"""
    atol(solver::IterativeSolver)

Returns the absolute tolerance of `solver`.
"""
function atol(solver::IterativeSolver) end

"""
    rtol(solver::IterativeSolver)

Returns the relative tolerance of `solver`.
"""
function rtol(solver::IterativeSolver) end

"""
    maxiters(solver::IterativeSolver)

Returns the maximum number of iterations that `solver` can take.
"""
function maxiters(solver::IterativeSolver) end

"""
    residual!(
        solver::IterativeSolver,
        threshold,
        iters,
        args...,
    )

Returns the norm of the residual, whether the solver converged, and the number
of times `f` was evaluated.

Uses `threshold` and `iters` to check for convergence by calling
`check_convergence`.
"""
function residual!(
    solver::IterativeSolver,
    threshold,
    iters,
    args...,
) end

"""
    initialize!(
        solver::IterativeSolver,
        threshold,
        iters,
        args...,
    )

Initializes `solver`, returning the output of `residual!`.
"""
function initialize!(
    solver::IterativeSolver,
    threshold,
    iters,
    args...,
) end

"""
    doiteration!(
        solver::IterativeSolver,
        threshold,
        iters,
        args...,
    )

Performs an iteration of `solver` and updates the solution vector.

Returns whether the solver converged, the number of times `f` was evaluated,
and the number of inner solver iterations (for restarted methods).
"""
function doiteration!(
    solver::IterativeSolver,
    threshold,
    iters,
    args...,
) end

"""
    solve!(solver::IterativeSolver, args...)::Int

Iteratively solves a (non)linear system equations with a (non)linear `solver`.

`args...` contains the problem to be solved in the format specified by the
solver algorithm being used. Returns the number of iterations taken by `solver`
and the number of times `solver` called `f`.
"""
function solve!(solver::IterativeSolver, args...)
    iters = 0 # total iterations of solver

    initial_residual_norm, has_converged, fcalls =
        initialize!(solver, atol(solver), iters, args...)
    has_converged && return (iters, fcalls)
    threshold = max(atol(solver), rtol(solver) * initial_residual_norm) # TODO: make this a min after comparison testing.

    while !has_converged && iters < maxiters(solver)
        has_converged, newfcalls, newiters =
            doiteration!(solver, threshold, iters, args...)
        iters += newiters # newiters = 1 unless using a restarted algorithm
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

# Macro used by algorithm constructors that checks whether the arguments
# specified by the user all make the function `check` return `true`. The first
# argument that makes it return `false` causes a `DomainError` to be thrown.
macro checkargs(string, check, args...)
    n = length(args)
    block = Expr(:block)
    block.args = Array{Any}(undef, n)
    for i in 1:n
        arg = args[i]
        error_message = "$arg must $string, but it was set to "
        arg = esc(arg)
        block.args[i] = :(
            if !isnothing($arg) && !$check($arg)
                throw(DomainError(string($error_message, $arg)))
            end
        )
    end
    return block
end

include("GeneralizedMinimalResidualAlgorithm.jl")
include("BatchedGeneralizedMinimalResidualAlgorithm.jl")
include("JacobianFreeNewtonKrylovAlgorithm.jl")
include("StandardPicardAlgorithm.jl")
include("AccelerationAlgorithm.jl")
include("GeneralizedConjugateResidualAlgorithm.jl")
include("ConjugateGradientAlgorithm.jl")

# TODO:
#   - Make GeneralizedMinimalResidualAlgorithm look more like BatchedGeneralizedMinimalResidualAlgorithm
#       - Check if the explicit computations in BatchedGeneralizedMinimalResidualAlgorithm are more efficient than the abstractions in GeneralizedMinimalResidualAlgorithm.
#       - Fix the preconditioner implementation in GeneralizedMinimalResidualAlgorithm.
#           - The solution taken from Solvent.jl looks wrong; use the solution from BatchedGeneralizedMinimalResidualAlgorithm, which has a seperate ΔQ vector to which the preconditioner is applied.
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
#   - Get a reference for stepsize() computation in JacobianFreeNewtonKrylovSolver
#   - [DONE] Check whether weighted_norm needs to be passed around everywhere
#   - Pass α for EulerOperator in args... to solve!()
#   - Rename JaCobIanfrEEneWtONKryLovSoLVeR with proper capitalization after removing jacobian_free_newton_krylov_solver.jl
#   - If we want other linear solvers and preconditioners, we should check out https://arxiv.org/pdf/1607.00351.pdf (may help solve stiffer nonlinear problems)
#       - "For symmetric systems, conjugate gradient (CG) and MINRES are widely recognized as the best [Krylov] methods. However, the situation is far less clear for nonsymmetric systems."
#       - Detailed comparison of Krylov methods (GMRES, TFQMR, BiCGSTAB and QMRCGSTAB) and general preconditioners (Gauss-Seidel, incomplete LU factorization, and algebraic multigrid).
#       - "GMRES tends to deliver better performance when coupled with an effective multigrid preconditioner, but it is less competitive with an ineffective preconditioner due to restarts."
#       - "Right preconditioning is, in general, more reliable than left preconditioning for large-scale systems."
#   - Testing of the IterativeAlgorithm Interface
#       - [DONE] Convert CG and GCR to new interface
#       - [DONE] Comparison testing with old code
#       - [DONE] More thorough tests following example of test/Numerics/SystemSolvers/iterativesolvers.jl
#           - No iterations if initial value is the solution
#           - Test expected number of iterations
#           - Convert these tests to use the new interface
#       - Test all params throw correct errors if invalid input used
#       - Test Accelerator has <= number of iterations as non-accelerated version
#       - Correctness testing with large and small problems
#       - Organize test directory
#       - Remove `sarrays` param after benchmarking
#       - Some solvers break in tests when switching to min threshold;
#           this seems to be when atol and rtol are related to eps(FT), as defaulted.