export StandardPicardAlgorithm

struct StandardPicardAlgorithm <: IterativeAlgorithm
    atol
    rtol
    maxiters
end

"""
    function StandardPicardAlgorithm(
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxiters::Union{Int, Nothing} = nothing,
    )

Constructor for the `StandardPicardAlgorithm`, which solves `f(Q) = Q`.

The Picard method solves a fixed point problem by repeatedly setting new
estimate `Q^{k+1} = f(Q^k)`. `f` must be a contractive function.

# Keyword Arguments
- `atol`: absolute tolerance; defaults to `1e-6`
- `rtol`: relative tolerance; defaults to `1e-6`
- `maxiters`: maximum number of iterations; defaults to 20
"""
function StandardPicardAlgorithm(;
    atol::Union{Real, Nothing} = nothing,
    rtol::Union{Real, Nothing} = nothing,
    maxiters::Union{Int, Nothing} = nothing,
)
    @check_positive(atol, rtol, maxiters)
    return StandardPicardAlgorithm(
        atol,
        rtol,
        maxiters,
    )
end

struct StandardPicardSolver{AT, FT} <: IterativeSolver
    fQ::AT      # container for Q^{k+1} = f(Q^k)
    residual::AT       # container for residual Q^k - f(Q^k) 
    atol::FT
    rtol::FT
    maxiters::Int
end

function IterativeSolver(
    algorithm::StandardPicardAlgorithm,
    f!,
    Q,
    args...;
)
    FT = eltype(Q)

    atol = isnothing(algorithm.atol) ? FT(1e-6) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? FT(1e-6) : FT(algorithm.rtol)
    maxiters = isnothing(algorithm.maxiters) ? 20 : algorithm.maxiters
    
    return StandardPicardSolver(
        similar(Q),
        similar(Q),
        atol,
        rtol,
        maxiters,
    )
end

atol(solver::StandardPicardSolver) = solver.atol
rtol(solver::StandardPicardSolver) = solver.rtol
maxiters(solver::StandardPicardSolver) = solver.maxiters

function initialize!(
    solver::StandardPicardSolver,
    threshold,
    iters,
    f!,
    Q,
    args...,
)
    fQ = solver.fQ
    R = solver.residual
    f!(fQ, Q, args...)
    R .= Q .- fQ
    residual_norm  = norm(R, weighted_norm)
    has_converged = check_convergence(residual_norm, threshold, iters)
    return residual_norm, has_converged, 1
end

function doiteration!(
    solver::StandardPicardSolver,
    threshold,
    iters,
    f!,
    Q,
    args...,
)
    R = solver.residual
    fQ = solver.fQ
    Q .= fQ

    # Compute residual norm and residual for next step
    f!(fQ, Q, args...)
    R .= fQ .- Q
    residual_norm  = norm(R, weighted_norm)
    has_converged = check_convergence(residual_norm, threshold, iters)
    @info Q
    return has_converged, 1
end