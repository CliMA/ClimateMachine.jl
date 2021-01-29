export AndersonAccelerationAlgorithm, AccelerationSolver

abstract type AccelerationSolver <: IterativeSolver end

atol(solver::AccelerationSolver) = atol(solver.iterativesolver)
rtol(solver::AccelerationSolver) = rtol(solver.iterativesolver)
maxiters(solver::AccelerationSolver) = maxiters(solver.iterativesolver)

function residual!(
    solver::AccelerationSolver,
    threshold,
    iters,
    args...;
)
    return residual!(solver.iterativesolver, threshold, iters, args...)
end

function initialize!(
    solver::AccelerationSolver,
    threshold,
    iters,
    args...;
)
    return initialize!(solver.iterativesolver, threshold, iters, args...)
end

function doiteration!(
    solver::AccelerationSolver,
    threshold,
    iters,
    Q,
    args...;
)
    has_converged, inneriters = doiteration!(
        solver.iterativesolver,
        threshold,
        iters,
        Q,
        args...;
    )
    @assert inneriters == 1 "Cannot accelerate when inneriters ≂̸ 1."
    if !has_converged
        doacceleration!(solver, Q, iters)
        _, has_converged = residual!(solver.iterativesolver, threshold, iters, Q, args...)
    end
    return has_converged, inneriters
end

struct AndersonAccelerationAlgorithm <: IterativeAlgorithm
    iterativealgorithm
    depth
    ω
end

"""
    AndersonAccelerationAlgorithm(
        iterativealgorithm::IterativeAlgorithm;
        depth::Union{Int, Nothing} = nothing,
        ω::Union{Real, Nothing} = nothing,
    )

Constructor for a `AndersonAccelerationAlgorithm`, which accelerates the solution
of an equation by `iterativealgorithm`.

For up to `depth` previous solution estimates, the norm of the weighted sum of their
residuals is minimized at each iteration. The optimal weights are then used to
construct a new solution estimate from the previous estimates.

# Arguments
- `iterativealgorithm`: algorithm used to solve the equation

# Keyword Arguments
- `depth`: accelerator window size; defaults to `1`
- `ω`: relaxation parameter; defaults to `1.0`
"""
function AndersonAccelerationAlgorithm(
    iterativealgorithm::IterativeAlgorithm;
    depth::Union{Int, Nothing} = nothing,
    ω::Union{Real, Nothing} = nothing,
)
    @checkargs("be positive", arg -> arg > 0, depth, ω)
    @checkargs("be at most 1", arg -> arg <= 1, ω)
    return AndersonAccelerationAlgorithm(
        iterativealgorithm,
        depth,
        ω,
    )
end

struct AndersonAccelerationSolver{IST, AT1, AT2, FT} <: AccelerationSolver
    iterativesolver::IST
    depth::Int                    # accelerator window size
    ω::FT                         # relaxation param
    β::AT1                        # β_k, linear combination weights
    x::AT1                        # x_k
    xprev::AT1                    # x_{k-1}
    g::AT1                        # g_k
    gcopy::AT1
    gprev::AT1                    # g_{k-1}
    Xβ::AT1
    Gβ::AT1
    X::AT2                        # (x_k - x_{k-1}, ..., x_{k-m_k+1} - x_{k-m_k})
    G::AT2                        # (g_k - g_{k-1}, ..., g_{k-m_k+1} - g_{k-m_k})
    Gcopy::AT2
end

function IterativeSolver(
    algorithm::AndersonAccelerationAlgorithm,
    Q,
    args...;
)
    FT = eltype(Q)
    
    depth = isnothing(algorithm.depth) ? 1 : algorithm.depth
    ω = isnothing(algorithm.ω) ? FT(1) : FT(algorithm.ω)    
    
    β = similar(vec(Q), depth)
    x = similar(vec(Q))
    X = similar(x, length(x), depth)

    return AndersonAccelerationSolver(
        IterativeSolver(algorithm.iterativealgorithm, Q, args...),
        depth, ω, β,
        x, similar(x), similar(x), similar(x), similar(x), similar(x), similar(x),
        X, similar(X), similar(X),
    )
end

function doacceleration!(solver::AndersonAccelerationSolver, Q, k)
    depth = solver.depth
    ω = solver.ω
    x = solver.x
    xprev = solver.xprev
    g = solver.g
    gcopy = solver.gcopy
    gprev = solver.gprev
    X = solver.X
    G = solver.G
    Gcopy = solver.Gcopy
    β = solver.β
    Xβ = solver.Xβ
    Gβ = solver.Gβ

    fx = vec(Q)

    if k == 0
        xprev .= x                     # x_0
        gprev .= fx .- x               # g_0 = f(x_0) - x_0
        x .= ω .* fx .+ (1 - ω) .* x   # x_1 = ω f(x_0) + (1 - ω) x_0
    else
        mk = min(depth, k)
        g .= fx .- x                   # g_k = f(x_k) - x_k
        X[:, 2:mk] .= X[:, 1:mk - 1]   # X_k = (x_k - x_{k-1}, ...,
        X[:, 1] .= x .- xprev          #        x_{k-m_k+1} - x_{k-m_k})
        G[:, 2:mk] .= G[:, 1:mk - 1]   # G_k = (g_k - g_{k-1}, ...,
        G[:, 1] .= g .- gprev          #        g_{k-m_k+1} - g_{k-m_k})

        βview = view(β, 1:mk)
        Xview = view(X, :, 1:mk)
        Gview = view(G, :, 1:mk)
        Gcopyview = view(Gcopy, :, 1:mk)
        
        Gcopyview .= Gview
        qr = qr!(Gcopyview, Val(true))
        
        # β_k = argmin_β(g_k - G_k β)
        # Optimized version of ldiv!(βview, qr, g)
        l = length(g)
        if l > mk
            gcopy .= g
            ldiv!(qr, gcopy)
            βview .= view(gcopy, 1:mk)
        else
            # Can't be a broadcast because mk != l.
            # Should we replace all the broadcasts with copyto!s?
            copyto!(βview, view(g, 1:l))
            ldiv!(qr, βview)
        end

        xprev .= x                     # x_k        
        mul!(Xβ, Xview, βview)
        mul!(Gβ, Gview, βview)
        x .= x .- Xβ .+ ω .* (g .- Gβ) # x_{k+1} = x_k - X_k β_k + ω (g_k - G_k β_k)
        gprev .= g                     # g_k
    end

    fx .= x
end