export StandardPicardAlgorithm, StandardPicardSolver # TODO: Remove solver export.

"""
    StandardPicardAlgorithm(
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxiters::Union{Int, Nothing} = nothing,
    )

Constructor for a `StandardPicardAlgorithm`, which solves an equation of the
form `f(Q) = Q`, where `f` is assumed to be a contraction mapping of `Q`.

This algorithm uses the standard Picard method, which sets `Q^{k+1} = f(Q^k)`,
where `Q^k` and `Q^{k+1}` are the values of `Q` on the `k`-th and `k+1`-th
iterations of the algorithm.

# Keyword Arguments
- `atol`: absolute tolerance; defaults to `1e-6`
- `rtol`: relative tolerance; defaults to `1e-6`
- `maxiters`: maximum number of iterations; defaults to 10
"""
struct StandardPicardAlgorithm <: FixedPointIterativeAlgorithm
    atol
    rtol
    maxiters
    function StandardPicardAlgorithm(;
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxiters::Union{Int, Nothing} = nothing,
    )
        @checkargs("be positive", arg -> arg > 0, atol, rtol, maxiters)
        return new(atol, rtol, maxiters)
    end
end


struct StandardPicardSolver{AT, FT} <: IterativeSolver
    fQ::AT        # container for f(Q^k)
    residual::AT  # container for residual f(Q^k) - Q^k
    atol::FT      # absolute tolerance
    rtol::FT      # relative tolerance
    maxiters::Int # maximum number of iterations
end

function IterativeSolver(
    algorithm::StandardPicardAlgorithm,
    Q,
    f!,
)
    FT = eltype(Q)

    atol = isnothing(algorithm.atol) ? FT(1e-6) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? FT(1e-6) : FT(algorithm.rtol)
    maxiters = isnothing(algorithm.maxiters) ? 10 : algorithm.maxiters
    
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

function residual!(solver::StandardPicardSolver,
    threshold,
    iters,
    Q,
    f!,
    args...,
)
    fQ = solver.fQ
    residual = solver.residual

    f!(fQ, Q, args...)
    residual .= fQ .- Q

    residual_norm  = norm(residual, weighted_norm)
    has_converged = check_convergence(residual_norm, threshold, iters)
    return residual_norm, has_converged
end

function initialize!(
    solver::StandardPicardSolver,
    threshold,
    iters,
    args...,
)
    return residual!(solver, threshold, iters, args...)
end

function picardcount(solver::StandardPicardSolver, bool::Bool) return nothing end

function doiteration!(
    solver::StandardPicardSolver,
    threshold,
    iters,
    Q,
    f!,
    args...,
)
picardcount(solver, true)
    Q .= solver.fQ
    residual_norm, has_converged = residual!(solver, threshold, iters, Q, f!, args...)
    return has_converged, 1
end