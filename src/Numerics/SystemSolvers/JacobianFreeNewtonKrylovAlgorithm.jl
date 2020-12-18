export JacobianFreeNewtonKrylovAlgorithm

include("enable_duals.jl")

abstract type JacobianVectorProduct end

mutable struct JacobianVectorProductFD{F!T, AT1, AT2, FT} <: JacobianVectorProduct
    f!::F!T   # nonlinear operator f
    Q::AT1    # pointer to the solution vector Q
    QdQ::AT1  # container for Q + e ΔQ
    fQ::AT2   # cache for f(Q)
    fQdQ::AT2 # container for f(Q + e ΔQ)
    β::FT     # parameter used to compute e
end

mutable struct JacobianVectorProductAD{F!T, AT1, AT2} <: JacobianVectorProduct
    f!::F!T   # nonlinear operator f that can operate on dual numbers
    QdQ::AT1  # container for Q + ε ΔQ that can store dual numbers
    fQdQ::AT2 # container for f(Q + ε ΔQ) that can store dual numbers
end

"""
    JacobianVectorProduct(f!, Q, fQ, autodiff, β)
    
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
...
# Arguments
- `f!`: nonlinear operator `f`
- `Q`: the point at which the Jacobian of `f` must be computed
- `fQ`: an array with the same shape and element type as `f(Q)`
- `autodiff`: whether to use automatic differentiation for calculating the
    Jacobian vector product (if `false`, use the finite difference method)
- `β`: parameter for the finite difference method (used if `autodiff == false`)
...
"""
function JacobianVectorProduct(f!, Q, fQ, autodiff, β)
    if autodiff
        return JacobianVectorProductAD(
            enable_duals(f!),
            enable_duals(Q),
            enable_duals(similar(fQ)),
        )
    else
        return JacobianVectorProductFD(
            f!,
            Q,
            similar(Q),
            similar(fQ),
            similar(fQ),
            β,
        )
    end
end

function stepsize(ΔQ, Q, β::FT) where {FT}
    n = FT(length(ΔQ))
    normΔQ = norm(ΔQ, weighted_norm)
    factor = normΔQ > β^2 ? n * normΔQ : n
    return β * norm(Q, 1, false) / factor + β
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

function initializejvp!(jvp!::JacobianVectorProductFD, Q)
    jvp!.Q = Q
    return nothing
end
function initializejvp!(jvp!::JacobianVectorProductAD, Q)
    jvp!.QdQ = setvalue!(jvp.QdQ, Q)
    return nothing
end

function updatejvp!(jvp!::JacobianVectorProductFD, args...)
    jvp!.f!(jvp!.fQ, jvp!.Q, args...)
    return nothing
end
function updatejvp!(jvp!::JacobianVectorProductAD, args...)
    return nothing
end

struct JacobianFreeNewtonKrylovAlgorithm <: IterativeAlgorithm
    iterativelinearalgorithm
    atol
    rtol
    maxiters
    autodiff
    β
end

"""
    JacobianFreeNewtonKrylovAlgorithm(
        iterativelinearalgorithm;
        atol::Union{AbstractFloat, Nothing} = nothing,
        rtol::Union{AbstractFloat, Nothing} = nothing,
        maxiters::Union{Int, Nothing} = nothing,
        autodiff::Union{Bool, Nothing} = nothing,
        β::Union{AbstractFloat, Nothing} = nothing,
    )

Constructor for the `JacobianFreeNewtonKrylovAlgorithm`, which solves a
`StandardProblem` that represents the equation `f(Q) = frhs`.

Suppose that the value of `Q` on the `k`-th iteration of the algorithm is
`Q^k`, where `f(Q^k) ≠ frhs`. Since `f` is analytic at `Q^k`, its Taylor series
expansion around `Q^k` gives the approximation

                           dim(Q)  ∂f
    f(Q^k + ΔQ) = f(Q^k) +   Σ    ----(Q^k) ΔQ_i + O(ΔQ²) ≈
                           i = 1  ∂Q_i
                           ∂f
                ≈ f(Q^k) + --(Q^k) ΔQ + 0 = f(Q^k) + J(Q^k) ΔQ.
                           ∂Q

Setting this equal to `frhs` gives the equation

    J(Q^k) ΔQ = frhs - f(Q^k).

To get `Q^{k+1}`, the algorithm solves this linear equation for `ΔQ` and sets

    Q^{k+1} = Q^k + ΔQ.

A standard Newton algorithm would solve the linear equation by explicitly
constructing the Jacobian `J(Q^k)` and finding its inverse. The Newton-Krylov
algorithm avoids this by using a Krylov subspace method to solve the equation.

...
# Arguments
- `iterativelinearalgorithm`: algorithm used to solve the linear equation
    generated on each iteration
- `atol`: absolute tolerance; defaults to `1e-6`
- `rtol`: relative tolerance; defaults to `1e-6`
- `maxiters`: maximum number of iterations; defaults to 10
- `autodiff`: whether to use automatic differentiation for calculating the
    Jacobian vector product (if `false`, use the finite difference method);
    defaults to `false`
- `β`: parameter for the finite difference method (for `autodiff == false`);
    defaults to `1e-4`
...
"""
function JacobianFreeNewtonKrylovAlgorithm(
    iterativelinearalgorithm;
    atol::Union{AbstractFloat, Nothing} = nothing,
    rtol::Union{AbstractFloat, Nothing} = nothing,
    maxiters::Union{Int, Nothing} = nothing,
    autodiff::Union{Bool, Nothing} = nothing,
    β::Union{AbstractFloat, Nothing} = nothing,
)
    return JacobianFreeNewtonKrylovAlgorithm(
        iterativelinearalgorithm,
        atol,
        rtol,
        maxiters,
        autodiff,
        β,
    )
end

struct JaCobIanfrEEneWtONKryLovSoLVeR{ILST, LPT, JVPT, AT1, AT2, FT} <:
        IterativeSolver
    iterativelinearsolver::ILST # solver used to solve the linear problem
    linearproblem::LPT          # linear problem jvp(ΔQ) = Δf
    jvp!::JVPT                  # Jacobian vector product jvp(ΔQ) ≈ J(Q^k) ΔQ
    ΔQ::AT1                     # container for Q^{k+1} - Q^k
    Δf::AT2                     # container for frhs - f(Q^k)
    atol::FT                    # absolute tolerance
    rtol::FT                    # relative tolerance
    maxiters::Int               # maximum number of iterations
end

function IterativeSolver(
    algorithm::JacobianFreeNewtonKrylovAlgorithm,
    problem::StandardProblem
)
    Q = problem.Q
    FT = eltype(Q)

    atol = isnothing(algorithm.atol) ? FT(1e-6) : FT(algorithm.atol)
    rtol = isnothing(algorithm.rtol) ? FT(1e-6) : FT(algorithm.rtol)
    maxiters = isnothing(algorithm.maxiters) ? 10 : algorithm.maxiters
    autodiff = isnothing(algorithm.autodiff) ? false : algorithm.autodiff
    β = isnothing(algorithm.β) ? FT(1e-4) : FT(algorithm.β)

    jvp! = JacobianVectorProduct(problem.f!, Q, problem.frhs, autodiff, β)
    ΔQ = similar(Q)
    Δf = similar(Q)
    linearproblem = StandardProblem(jvp!, ΔQ, Δf)

    return JaCobIanfrEEneWtONKryLovSoLVeR(
        IterativeSolver(algorithm.iterativelinearalgorithm, linearproblem),
        linearproblem,
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
    problem::StandardProblem,
    args...,
)
    Δf = solver.Δf

    problem.f!(Δf, problem.Q, args...)
    Δf .= problem.frhs .- Δf
    return norm(Δf, weighted_norm)
end

function initialize!(
    solver::JaCobIanfrEEneWtONKryLovSoLVeR,
    problem::StandardProblem,
    args...,
)
    initializejvp!(solver.jvp!, problem.Q)
    return residual!(solver, problem, args...)
end

function doiteration!(
    solver::JaCobIanfrEEneWtONKryLovSoLVeR,
    problem::StandardProblem,
    threshold,
    iters,
    args...,
)
    ΔQ = solver.ΔQ
    jvp! = solver.jvp!
    iterativelinearsolver = solver.iterativelinearsolver
    preconditioner = iterativelinearsolver.preconditioner
    f! = problem.f!

    ΔQ .= zero(eltype(ΔQ))
    updatejvp!(jvp!, args...)

    # TODO: Abstract this away in Preconditioner.jl
    if isa(problem.f!, EulerOperator)
        preconditioner_update!(jvp!, f!.f!, preconditioner, args...)
    elseif isa(problem.f!, DGModel)
        preconditioner_update!(jvp!, f!, preconditioner, args...)
    end
    solve!(iterativelinearsolver, solver.linearproblem, args...)
    preconditioner_counter_update!(preconditioner)

    problem.Q .+= ΔQ
    residual_norm = residual!(solver, problem, args...)
    return check_convergence(residual_norm, threshold, iters)
end
