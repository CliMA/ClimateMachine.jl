export AbstractAccelerator,
    AndersonAccelerator,
    NGMRESAccelerator

"""
    AbstractAccelerator

This is an abstract type representing a generic accelerator that wraps another nonlinear solver.
"""
abstract type AbstractAccelerator <: AbstractNonlinearSolver end
function internalsolver(::AbstractAccelerator) end
function set_initial_residual_norm(::AbstractAccelerator, ::Real) end
function get_initial_residual_norm(::AbstractAccelerator) end
function doacceleration!(::AbstractAccelerator, ::Any) end

function initialize!(rhs!, Q, Qrhs, solver::AbstractAccelerator, args...)
    initial_residual_norm = initialize!(rhs!, Q, Qrhs, internalsolver(solver), args...)
    set_initial_residual_norm(solver, initial_residual_norm)
end
function dononlineariteration!(
    rhs!,
    jvp!,
    preconditioner,
    Q,
    Qrhs,
    solver::AbstractAccelerator,
    iters,
    args...,
)
    nlsolver = internalsolver(solver)
    residual_norm, linear_iterations = dononlineariteration!(
        rhs!,
        jvp!,
        preconditioner,
        Q,
        Qrhs,
        nlsolver,
        iters,
        args...,
    )
    converged = checkconverged(get_initial_residual_norm(solver), residual_norm)
    if !converged
        doacceleration!(solver, Q, iters)
        R = nlsolver.residual
        rhs!(R, Q, args...)
        R .-= Qrhs
        residual_norm = norm(R, weighted_norm)
    end
    return residual_norm, linear_iterations
end

"""
struct AndersonAccelerator{AT}
    A::DGColumnBandedMatrix
    Q::AT
    PQ::AT
    counter::Int
    update_freq::Int
end

...
# Arguments
- `A`: the lu factor of the precondition (approximated Jacobian), in the DGColumnBandedMatrix format
- `Q`: MPIArray container, used to update A
- `PQ`: MPIArray container, used to update A
- `counter`: count the number of Newton, when counter > update_freq or counter < 0, update precondition
- `update_freq`: preconditioner update frequency
...
"""
# TODO: Might want to get rid of mutable + k, and pass k to dononlineariteration
mutable struct AndersonAccelerator{M, FT, AT1, AT2, NLS} <: AbstractAccelerator
    tol::FT # TODO: REMOVE LATER
    initial_residual_norm::FT
    ω::FT                         # relaxation parameter
    β::AT1                        # β_k, linear combination weights
    x::AT1                        # x_k
    xprev::AT1                    # x_{k-1}
    g::AT1                        # g_k
    gcopy::AT1                    # Only needed when length(vec(Q)) > M.
    gprev::AT1                    # g_{k-1}
    Xβ::AT1
    Gβ::AT1
    X::AT2                        # (x_k - x_{k-1}, ..., x_{k-m_k+1} - x_{k-m_k})
    G::AT2                        # (g_k - g_{k-1}, ..., g_{k-m_k+1} - g_{k-m_k})
    Gcopy::AT2
    nlsolver::NLS
end

function AndersonAccelerator(Q::AT, nlsolver::NLS; M::Int = 1, ω::FT = 1.) where {AT, NLS, FT}
    β = similar(vec(Q), M) # Could also be @MArray zeros(M), but ldiv! is not defined for MArrays.
    x = similar(vec(Q))
    X = similar(x, length(x), M)
    AndersonAccelerator{M, FT, typeof(x), typeof(X), NLS}(
        nlsolver.tol, zero(FT), ω, β,
        x, similar(x), similar(x), similar(x), similar(x), similar(x), similar(x),
        X, similar(X), similar(X),
        nlsolver
    )
end

internalsolver(a::AndersonAccelerator) = a.nlsolver
set_initial_residual_norm(a::AndersonAccelerator, n::Real) = (a.initial_residual_norm = n)
get_initial_residual_norm(a::AndersonAccelerator) = a.initial_residual_norm

function doacceleration!(a::AndersonAccelerator{M}, Q, k) where {M}
    ω = a.ω
    x = a.x
    xprev = a.xprev
    g = a.g
    gcopy = a.gcopy
    gprev = a.gprev
    X = a.X
    G = a.G
    Gcopy = a.Gcopy
    β = a.β
    Xβ = a.Xβ
    Gβ = a.Gβ

    fx = vec(Q)

    if k == 0
        xprev .= x                     # x_0
        gprev .= fx .- x               # g_0 = f(x_0) - x_0
        x .= ω .* fx .+ (1 - ω) .* x   # x_1 = ω f(x_0) + (1 - ω) x_0
    else
        mk = min(M, k)
        g .= fx .- x                   # g_k = f(x_k) - x_k
        X[:, 2:mk] .= X[:, 1:mk - 1]   # X_k = (x_k - x_{k-1}, ...,
        X[:, 1] .= x .- xprev          #        x_{k-m_k+1} - x_{k-m_k})
        G[:, 2:mk] .= G[:, 1:mk - 1]   # G_k = (g_k - g_{k-1}, ...,
        G[:, 1] .= g .- gprev          #        g_{k-m_k+1} - g_{k-m_k})
        Gcopy[:, 1:mk] .= G[:, 1:mk]
        βview = view(β, 1:mk)
        qr = qr!(view(Gcopy, :, 1:mk), Val(true))
        
        # Optimized version of ldiv!(βview, qr, g)
        # β_k = argmin_β(g_k - G_k β)
        l = size(qr, 1)
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
        @views mul!(Xβ, X[:, 1:mk], β[1:mk])
        @views mul!(Gβ, G[:, 1:mk], β[1:mk])
        x .= x .- Xβ .+ ω .* (g .- Gβ) # x_{k+1} = x_k - X_k β_k + ω (g_k - G_k β_k)
        gprev .= g                     # g_k
    end

    fx .= x
end