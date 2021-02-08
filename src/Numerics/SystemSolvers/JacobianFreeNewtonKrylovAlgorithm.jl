export JacobianFreeNewtonKrylovAlgorithm, JaCobIanfrEEneWtONKryLovSoLVeR # TODO: Remove solver export.

include("enable_duals.jl")

abstract type JacobianVectorProduct end

mutable struct JacobianVectorProductFD{F!T, AT, FT} <: JacobianVectorProduct
    f!::F!T   # nonlinear operator f
    Q::AT     # solution vector Q
    QdQ::AT   # container for Q + e ΔQ
    fQ::AT    # container for f(Q)
    fQdQ::AT  # container for f(Q + e ΔQ)
    β::FT     # parameter used to compute e
end

mutable struct JacobianVectorProductAD{F!T, AT} <: JacobianVectorProduct
    f!::F!T   # nonlinear operator f that can operate on dual numbers
    QdQ::AT   # container for Q + ε ΔQ that can store dual numbers
    fQdQ::AT  # container for f(Q + ε ΔQ) that can store dual numbers
end

"""
    JacobianVectorProduct(Q, f!, autodiff, β)
    
Constructor for the `JacobianVectorProduct`.
    
If `autodiff == false`, the Jacobian vector product is approximated with the
finite difference method as

              ∂f         f(Q + e ΔQ) - f(Q)
    J(Q) ΔQ = --(Q) ΔQ ≈ ------------------, where e = `stepsize(ΔQ, Q, β)`.
              ∂Q                 e

If `autodiff == true`, the Jacobian vector product is determined exactly with
automatic differentiation. This is achieved by computing `f(Q + ε ΔQ)`, where
`ε` is an infinitesimal with the property that `ε² = 0`. Since `f` is analytic
at `Q`, its Taylor series expansion around `Q` gives the equation

                           dim(Q)  ∂f
    f(Q + ε ΔQ) = f(Q) + ε   Σ    ----(Q) ΔQ_i + O(ε²) =
                           i = 1  ∂Q_i
                           ∂f
                = f(Q) + ε --(Q) ΔQ + 0 = f(Q) + ε J(Q) ΔQ.
                           ∂Q

# Arguments
- `f!`: nonlinear operator `f`; used for type information only
- `Q`: array that specifies the point at which the Jacobian of `f` needs to be
    computed; used for type information only
- `autodiff`: whether to use automatic differentiation for calculating the
    Jacobian vector product; when `autodiff == false`, the finite difference
    method is used
- `β`: parameter for the finite difference method; ignored when
    `autodiff == true`
"""
function JacobianVectorProduct(f!, Q, autodiff, β)
    if autodiff
        return JacobianVectorProductAD(
            enable_duals(f!),
            enable_duals(Q),
            enable_duals(similar(Q)),
        )
    else
        return JacobianVectorProductFD(
            f!,
            Q,
            similar(Q),
            similar(Q),
            similar(Q),
            β,
        )
    end
end

function stepsize(ΔQ, Q, β::FT) where {FT}
    n = FT(length(ΔQ))
    normΔQ = norm(ΔQ, weighted_norm)
    factor = normΔQ > β^2 ? n * normΔQ : n
    return one(FT) / factor * β * norm(Q, 1, false) + β # TODO: Simplify this after testing is done.
end

function (jvp!::JacobianVectorProductFD)(JΔQ, ΔQ, args...)
    Q = jvp!.Q
    QdQ = jvp!.QdQ
    fQ = jvp!.fQ
    fQdQ = jvp!.fQdQ

    e = stepsize(ΔQ, Q, jvp!.β)
    QdQ .= Q .+ e .* ΔQ
    jvp!.f!(fQdQ, QdQ, args...)
    JΔQ .= (fQdQ .- fQ) ./ e
    return nothing
end

function (jvp!::JacobianVectorProductAD)(JΔQ, ΔQ, args...)
    QdQ = jvp!.QdQ
    fQdQ = jvp!.fQdQ

    partial(QdQ) .= ΔQ
    jvp!.f!(fQdQ, QdQ, args...)
    JΔQ .= partial(fQdQ)
    return nothing
end

function initializejvp!(jvp!::JacobianVectorProductFD, f!, Q)
    jvp!.f! = f!
    jvp!.Q = Q
    return nothing
end
function initializejvp!(jvp!::JacobianVectorProductAD, f!, Q)
    jvp!.f! = update_duals!(jvp!.f!, f!)
    jvp!.QdQ = update_duals!(jvp!.QdQ, Q)
    return nothing
end

function updatejvp!(jvp!::JacobianVectorProductFD, args...)
    jvp!.f!(jvp!.fQ, jvp!.Q, args...)
    return nothing
end
function updatejvp!(jvp!::JacobianVectorProductAD, args...)
    return nothing
end

"""
    JacobianFreeNewtonKrylovAlgorithm(
        krylovalgorithm::KrylovAlgorithm;
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxiters::Union{Int, Nothing} = nothing,
        autodiff::Union{Bool, Nothing} = nothing,
        β::Union{Real, Nothing} = nothing,
    )

Constructor for the `JacobianFreeNewtonKrylovAlgorithm`, which solves an
equation of the form `f(Q) = rhs`.

Suppose that the value of `Q` on the `k`-th iteration of the algorithm is
`Q^k`, where `f(Q^k) ≠ rhs`. Since `f` is analytic at `Q^k`, its Taylor series
expansion around `Q^k` gives the approximation

                           dim(Q)  ∂f
    f(Q^k + ΔQ) = f(Q^k) +   Σ    ----(Q^k) ΔQ_i + O(ΔQ²) ≈
                           i = 1  ∂Q_i
                           ∂f
                ≈ f(Q^k) + --(Q^k) ΔQ + 0 = f(Q^k) + J(Q^k) ΔQ.
                           ∂Q

Setting this equal to `rhs` gives the equation

    J(Q^k) ΔQ = rhs - f(Q^k).

To get `Q^{k+1}`, the algorithm solves this linear equation for `ΔQ` and sets

    Q^{k+1} = Q^k + ΔQ.

A standard Newton algorithm would solve the linear equation by explicitly
constructing the Jacobian `J(Q^k)` and finding its inverse. The Newton-Krylov
algorithm avoids this by using a Krylov subspace method, iteratively evaluating
the function `jvp(ΔQ) ≈ J(Q^k) ΔQ` to solve the equation.

# Arguments
- `krylovalgorithm`: algorithm used to solve the linear equation generated on
    each iteration

# Keyword Arguments
- `atol`: absolute tolerance; defaults to `1e-6`
- `rtol`: relative tolerance; defaults to `1e-6`
- `maxiters`: maximum number of iterations; defaults to `10`
- `autodiff`: whether to use automatic differentiation for calculating the
    Jacobian vector product; when `autodiff == false`, the finite difference
    method is used; defaults to `false`
- `β`: parameter for the finite difference method; ignored when
    `autodiff == true`; defaults to `1e-4`
"""
struct JacobianFreeNewtonKrylovAlgorithm <: GenericIterativeAlgorithm
    krylovalgorithm
    atol
    rtol
    maxiters
    autodiff
    β
    function JacobianFreeNewtonKrylovAlgorithm(
        krylovalgorithm::KrylovAlgorithm;
        atol::Union{Real, Nothing} = nothing,
        rtol::Union{Real, Nothing} = nothing,
        maxiters::Union{Int, Nothing} = nothing,
        autodiff::Union{Bool, Nothing} = nothing,
        β::Union{Real, Nothing} = nothing,
    )
        @checkargs("be positive", arg -> arg > 0, atol, rtol, maxiters, β)
        @checkargs("be finite", arg -> isfinite(arg), β)
        return new(krylovalgorithm, atol, rtol, maxiters, autodiff, β)
    end
end

struct JaCobIanfrEEneWtONKryLovSoLVeR{ILST, JVPT, AT, FT} <:
        IterativeSolver
    krylovsolver::ILST # solver used to solve the linear problem
    jvp!::JVPT         # Jacobian vector product jvp(ΔQ) ≈ J(Q^k) ΔQ
    ΔQ::AT             # container for Q^{k+1} - Q^k
    Δf::AT             # container for rhs - f(Q^k)
    atol::FT           # absolute tolerance
    rtol::FT           # relative tolerance
    maxiters::Int      # maximum number of iterations
end

function IterativeSolver(
    algorithm::JacobianFreeNewtonKrylovAlgorithm,
    Q,
    f!,
    rhs,
)
    FT = eltype(Q)

    atol = isnothing(algorithm.atol) ? FT(1e-6) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? FT(1e-6) : FT(algorithm.rtol)
    maxiters = isnothing(algorithm.maxiters) ? 10 : algorithm.maxiters
    autodiff = isnothing(algorithm.autodiff) ? false : algorithm.autodiff
    β = isnothing(algorithm.β) ? FT(1e-4) : FT(algorithm.β)

    jvp! = JacobianVectorProduct(f!, Q, autodiff, β)
    ΔQ = similar(Q)
    Δf = similar(Q)
    return JaCobIanfrEEneWtONKryLovSoLVeR(
        IterativeSolver(algorithm.krylovalgorithm, ΔQ, jvp!, Δf),
        jvp!,
        ΔQ,
        Δf,
        atol,
        rtol,
        maxiters,
    )
end

atol(solver::JaCobIanfrEEneWtONKryLovSoLVeR) = solver.atol
rtol(solver::JaCobIanfrEEneWtONKryLovSoLVeR) = solver.rtol
maxiters(solver::JaCobIanfrEEneWtONKryLovSoLVeR) = solver.maxiters

function residual!(
    solver::JaCobIanfrEEneWtONKryLovSoLVeR,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    Δf = solver.Δf

    f!(Δf, Q, args...)
    Δf .= rhs .- Δf

    residual_norm = norm(Δf, weighted_norm)
    has_converged = check_convergence(residual_norm, threshold, iters)
    return residual_norm, has_converged
end

function initialize!(
    solver::JaCobIanfrEEneWtONKryLovSoLVeR,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    initializejvp!(solver.jvp!, f!, Q)
    return residual!(solver, threshold, iters, Q, f!, rhs, args...)
end

function doiteration!(
    solver::JaCobIanfrEEneWtONKryLovSoLVeR,
    threshold,
    iters,
    Q,
    f!,
    rhs,
    args...;
)
    ΔQ = solver.ΔQ
    jvp! = solver.jvp!
    krylovsolver = solver.krylovsolver
    preconditioner = krylovsolver.preconditioner

    ΔQ .= zero(eltype(ΔQ))
    updatejvp!(jvp!, args...)

    preconditioner_update!(jvp!, f!, preconditioner, args...)
    krylovsolver(ΔQ, jvp!, solver.Δf, args...)
    preconditioner_counter_update!(preconditioner)

    Q .+= ΔQ

    residual_norm, has_converged = residual!(solver, threshold, iters, Q, f!, rhs, args...)
    return has_converged, 1
end
