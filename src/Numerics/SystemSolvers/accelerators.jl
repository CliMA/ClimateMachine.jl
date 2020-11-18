export AbstractAccelerator,
    NoAccelerator,
    AndersonAccelerator,
    NGMRESAccelerator,
    doacceleration

"""
    AbstractAccelerator

This is an abstract type representing an accelerator such as NGMRES.
"""
abstract type AbstractAccelerator end

"""
    NoAccelerator

Do nothing
"""
struct NoAccelerator <: AbstractAccelerator end

"""
    do_acceleration!(::NoAccelerator, x)

Do nothing, when there is no accelerator, accelerator = Nothing
"""
function do_acceleration!(::NoAccelerator, x, k::Int) end

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
struct AndersonAccelerator{M, FT, AT1, AT2} <: AbstractAccelerator
    ω::FT                         # relaxation parameter
    β::MArray{Tuple{M}, FT, 1, M} # β_k
    x::AT1                        # x_k
    xprev::AT1                    # x_{k-1}
    g::AT1                        # g_k
    gprev::AT1                    # g_{k-1}
    Xβ::AT1
    Gβ::AT1
    X::AT2                        # (x_k - x_{k-1}, x_{k-1} - x_{k-2}, ..., x_{k-m_k+1} - x_{k-m_k})
    G::AT2                        # (g_k - g_{k-1}, g_{k-1} - g_{k-2}, ..., g_{k-m_k+1} - g_{k-m_k})
    Gcopy::AT2
end

function AndersonAccelerator(Q::AT; M::Int = 1, ω::FT = 1.) where {AT, FT}
    x = similar(vec(Q))
    X = similar(x, length(x), M)
    AndersonAccelerator{M, FT, typeof(x), typeof(X)}(
        ω,
        @MArray zeros(M),
        x, similar(x), similar(x), similar(x), similar(x), similar(x),
        X, similar(X), similar(X)
    )
end

function doacceleration!(a::AndersonAccelerator{M}, Q, k::Int) where {M}
    ω = a.ω
    x = a.x
    xprev = a.xprev
    g = a.g
    gprev = a.gprev
    X = a.x
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
        @views ldiv!(
            β[1:mk],
            qr!(Gcopy[:, 1:mk], Val(true)),
            g
        )                              # β_k = argmin_β(g_k - G_k β)
        xprev .= x                     # x_k        
        @views mul!(Xβ, X[:, 1:mk], β[1:mk])
        @views mul!(Gβ, G[:, 1:mk], β[1:mk])
        x .= x .- Xβ .+ ω .* (g .- Gβ) # x_{k+1} = x_k - X_k β_k + ω (g_k - G_k β_k)
        gprev .= g                     # g_k
    end

    fx .= x
end
