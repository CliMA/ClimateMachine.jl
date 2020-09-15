
export JacobianFreeNewtonKrylovSolver, JacobianAction

"""
mutable struct JacobianAction{FT, AT}
    rhs!
    ϵ::FT
    Q::AT
    Qdq::AT
    Fq::AT
    Fqdq::AT
end

Solve for Frhs = F(q), the Jacobian action is computed

    ∂F(Q)      F(Q + eΔQ) - F(Q)
    ---- ΔQ ≈ -------------------
     ∂Q                e


...
# Arguments     
- `rhs!`           : nonlinear operator F(Q)
- `ϵ::FT`          : ϵ used for finite difference, e = e(Q, ϵ)
- `Q::AT`          : cache for Q
- `Qdq::AT`        : container for Q + ϵΔQ
- `Fq::AT`         : cache for F(Q)
- `Fqdq::AT`       : container for F(Q + ϵΔQ)
...
"""
mutable struct JacobianAction{FT, AT}
    rhs!
    ϵ::FT
    Q::AT
    Qdq::AT
    Fq::AT
    Fqdq::AT
end

function JacobianAction(rhs!, Q, ϵ)
    return JacobianAction(
        rhs!,
        ϵ,
        similar(Q),
        similar(Q),
        similar(Q),
        similar(Q),
    )
end

"""
Approximates the action of the Jacobian of a nonlinear
form on a vector `ΔQ` using the difference quotient:

      ∂F(Q)      F(Q + e ΔQ) - F(Q)
JΔQ = ---- ΔQ ≈ -------------------
       ∂Q                e


Compute  JΔQ with cached Q and F(Q), and the direction  dQ
"""
function (op::JacobianAction)(JΔQ, dQ, args...)
    rhs! = op.rhs!
    Q = op.Q
    Qdq = op.Qdq
    ϵ = op.ϵ
    Fq = op.Fq
    Fqdq = op.Fqdq

    FT = eltype(dQ)
    n = length(dQ)
    normdQ = norm(dQ, weighted_norm)

    if normdQ > ϵ
        factor = FT(1 / (n * normdQ))
    else
        # initial newton step, ΔQ = 0
        factor = FT(1 / n)
    end

    β = √ϵ
    e = factor * β * norm(Q, 1, false) + β

    Qdq .= Q .+ e .* dQ

    rhs!(Fqdq, Qdq, args...)

    JΔQ .= (Fqdq .- Fq) ./ e

end

"""
update cached Q and F(Q) before each Newton iteration
"""
function update_Q!(op::JacobianAction, Q, args...)
    op.Q .= Q
    Fq = op.Fq

    op.rhs!(Fq, Q, args...)
end

"""
Solve for Frhs = F(Q), by finite difference

    ∂F(Q)      F(Q + eΔQ) - F(Q)
    ---- ΔQ ≈ -------------------
     ∂Q                e

     Q^n+1 = Q^n - dF/dQ(Q^{n})⁻¹ (F(Q^n) - Frhs)

     set ΔQ = F(Q^n) - Frhs
"""
mutable struct JacobianFreeNewtonKrylovSolver{FT, AT} <: AbstractNonlinearSolver
    # small number used for finite difference
    ϵ::FT
    # tolerances for convergence
    tol::FT
    # Max number of Newton iterations
    M::Int
    # Linear solver for the Jacobian system
    linearsolver
    # container for unknows ΔQ, which is updated for the linear solver
    ΔQ::AT
    # contrainer for F(Q)
    residual::AT
end

"""
JacobianFreeNewtonKrylovSolver constructor
"""
function JacobianFreeNewtonKrylovSolver(
    Q,
    linearsolver;
    ϵ = 1.e-8,
    tol = 1.e-6,
    M = 30,
)
    FT = eltype(Q)
    residual = similar(Q)
    ΔQ = similar(Q)
    return JacobianFreeNewtonKrylovSolver(
        FT(ϵ),
        FT(tol),
        M,
        linearsolver,
        ΔQ,
        residual,
    )
end

"""
JacobianFreeNewtonKrylovSolver initialize the residual
"""
function initialize!(
    rhs!,
    Q,
    Qrhs,
    solver::JacobianFreeNewtonKrylovSolver,
    args...,
)
    # where R = Qrhs - F(Q)
    R = solver.residual
    # Computes F(Q) and stores in R
    rhs!(R, Q, args...)
    # Computes R = R - Qrhs
    R .-= Qrhs
    return norm(R, weighted_norm)
end

"""
Solve for Frhs = F(Q), by finite difference

Q^n+1 = Q^n - dF/dQ(Q^{n})⁻¹ (F(Q^n) - Frhs)

set ΔQ = F(Q^n) - Frhs

...
# Arguments 
- `rhs!`:  functor rhs!(Q) =  F(Q)
- `jvp!`:  Jacobian action jvp!(ΔQ)  = dF/dQ(Q) ⋅ ΔQ
- `preconditioner`: approximation of dF/dQ(Q)
- `Q` : Q^n
- `Qrhs` : Frhs
- `solver`: linear solver
...
"""
function donewtoniteration!(
    rhs!,
    jvp!,
    preconditioner::AbstractPreconditioner,
    Q,
    Qrhs,
    solver::JacobianFreeNewtonKrylovSolver,
    args...,
)

    FT = eltype(Q)
    ΔQ = solver.ΔQ
    ΔQ .= FT(0.0)

    # R(Q) == 0, R = F(Q) - Qrhs, where F = rhs!
    # Compute right-hand side for Jacobian system:
    # J(Q)ΔQ = -R
    # where R = Qrhs - F(Q), which is computed at the end of last step or in the initialize function
    R = solver.residual

    # R = F(Q^n) - Frhs
    # ΔQ = dF/dQ(Q^{n})⁻¹ (Frhs - F(Q^n)) = -dF/dQ(Q^{n})⁻¹ R
    iters =
        linearsolve!(jvp!, preconditioner, solver.linearsolver, ΔQ, -R, args...)

    # Newton correction Q^{n+1} = Q^n + dF/dQ(Q^{n})⁻¹ (Frhs - F(Q^n))
    Q .+= ΔQ

    # Compute residual norm and residual for next step
    rhs!(R, Q, args...)
    R .-= Qrhs
    resnorm = norm(R, weighted_norm)

    return resnorm, iters
end
