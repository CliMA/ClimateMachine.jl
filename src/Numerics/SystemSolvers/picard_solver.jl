
export PicardStandardSolver

"""
Solve for Frhs = F(Q), by finite difference

    ∂F(Q)      F(Q + eΔQ) - F(Q)
    ---- ΔQ ≈ -------------------
     ∂Q                e

     Q^n+1 = Q^n - dF/dQ(Q^{n})⁻¹ (F(Q^n) - Frhs)

     set ΔQ = F(Q^n) - Frhs
"""
mutable struct PicardStandardSolver{FT, AT} <: AbstractNonlinearSolver
    # small number used for finite difference
    ϵ::FT
    # tolerances for convergence
    tol::FT
    # Max number of Newton iterations
    M::Int
    # Linear solver for the Jacobian system
    linearsolver::Any
    # container for unknows ΔQ, which is updated for the linear solver
    ΔQ::AT
    # contrainer for F(Q)
    residual::AT
end

function PicardStandardSolver(
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
PicardStandardSolver initialize the residual
"""
function initialize!(
    rhs!,
    Q,
    Qrhs,
    solver::PicardStandardSolver,
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
    dononlineariteration!(
        rhs!,
        jvp!,
        preconditioner::AbstractPreconditioner,
        Q,
        Qrhs,
        solver::PicardStandardSolver,
        iters,
        args...,
    )

Solve for Frhs = F(Q), by finite difference

Q^n+1 = Q^n - dF/dQ(Q^{n})⁻¹ (F(Q^n) - Frhs)

Newton
Q^n+1 - Q^n = dF/dQ(Q^{n})⁻¹ (Frhs - F(Q^n))
Frhs - F(Q^n) = dF/dQ(Q^{n}) * (Q^n+1 - Q^n)

Backward Euler
    (Q^{n+1} - Q^n)/Δt = f(Q^{n+1}, t^{n+1})
Newton
    Q^{n+1} - Δt f(Q^{n+1}, t^{n+1}) = Q^n
    F(Q) = Q - Δt f(Q, t^{n+1})
    Frhs = Q^n
    w.t.s. F(Q^{n+1}) = Frhs

    r(Q^{n+1}) = F(Q^{n+1}) - Frhs = Q^{n+1} - Δt f(Q^{n+1}, t^{n+1}) - Q^n

    F(Q^{n+1,k}) + ∂F/∂Q(Q^{n+1,k}) * (Q^{n+1,k+1} - Q^{n+1,k}) = Frhs
    Q^{n+1,k+1} = Q^{n+1,k} - ∂F/∂Q(Q^{n+1,k})⁻¹(F(Q^{n+1,k}) - Frhs)
"Standard" Picard
    Q^{n+1} = Δt f(Q^{n+1}, t^{n+1}) + Q^n
    F(Q) = Δt f(Q, t^{n+1}) + Q^n
    w.t.s. F(Q^{n+1}) = Q^{n+1}

    r(Q^{n+1}) = F(Q^{n+1}) - Q^{n+1} = Δt f(Q^{n+1}, t^{n+1}) + Q^n - Q^{n+1}

    Q^{n+1,k+1} = F(Q^{n+1,k})


F(Q) = Q - Δt f(Q, t^{n+1})
Qrhs = Q^n
r(Q) = F(Q) - Qrhs = Q - Δt f(Q, t^{n+1}) - Q^n
Q = -r(Q) + Q = Δt f(Q, t^{n+1}) + Q^n

**** Picard Fixed Point Idea ****
Solve F(Q) = Qrhs
F(Q) - Qrhs = 0 (root finding formulation)
    r(Q) = F(Q) - Qrhs (residual)
so we want to solve:
r(Q) = 0 <==> -r(Q) = 0
formulated as a fixed point method:
Q = -r(Q) + Q
Q_{k+1} = r(Q_k) + Q_k = Q_k + ΔQ
    so ΔQ = r(Q) = rhs!(Q) - Qrhs

Note: we already computed the residual for Q_k during the previous step/initialize

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
function dononlineariteration!(
    rhs!,
    jvp!,
    preconditioner::AbstractPreconditioner,
    Q,
    Qrhs,
    solver::PicardStandardSolver,
    iters,
    args...,
)
    R = solver.residual

    # Picard Update
    Q .-= R

    # Compute residual norm and residual for next step
    rhs!(R, Q, args...)
    R .-= Qrhs
    resnorm = norm(R, weighted_norm)

    return resnorm, iters
end
