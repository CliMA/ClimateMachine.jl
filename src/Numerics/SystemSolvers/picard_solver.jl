
export PicardStandardSolver

"""
Solve for F(Q) = Qrhs by Picard fixed point.
"""
struct PicardStandardSolver{FT, AT} <: AbstractNonlinearSolver
    # Absolute tolerance
    atol::FT
    # Relative tolerance
    rtol::FT
    # Max newton iterations
    maxiters::Int
    # container for F(Q)
    FQ::AT
    # container for F(Q) - Q
    residual::AT
end

function PicardStandardSolver(
    Q;
    atol = 1.e-6,
    rtol = 1.e-6,
    maxiters = 30,
)
    FT = eltype(Q)
    return PicardStandardSolver(
        FT(atol),
        FT(rtol),
        maxiters,
        similar(Q),
        similar(Q),
    )
end

atol(solver::PicardStandardSolver) = solver.atol
rtol(solver::PicardStandardSolver) = solver.rtol
maxiters(solver::PicardStandardSolver) = solver.maxiters

"""
PicardStandardSolver initialize the residual
"""
function initialize!(
    rhs!,
    Q,
    Qrhs, # unused
    solver::PicardStandardSolver,
    args...,
)
    FQ = solver.FQ
    R = solver.residual
    
    rhs!(FQ, Q, args...)
    R .= FQ .- Q
    return norm(R, weighted_norm)
end

"""
    dononlineariteration!(
        rhs!,
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
    F_n(Q) = Q - Δt f(Q, t^{n+1})
    Frhs = Q^n
    w.t.s. F_n(Q^{n+1}) = Frhs

    r_n(Q^{n+1}) = F_n(Q^{n+1}) - Frhs = Q^{n+1} - Δt f(Q^{n+1}, t^{n+1}) - Q^n

    F_n(Q^{n+1,k}) + ∂F_n/∂Q(Q^{n+1,k}) * (Q^{n+1,k+1} - Q^{n+1,k}) = Frhs
    Q^{n+1,k+1} = Q^{n+1,k} - ∂F_n/∂Q(Q^{n+1,k})⁻¹(F_n(Q^{n+1,k}) - Frhs)
"Standard" Picard
    Q^{n+1} = Δt f(Q^{n+1}, t^{n+1}) + Q^n
    F_p(Q) = Δt f(Q, t^{n+1}) + Q^n = Q + Q^n - F_n(Q)
    w.t.s. F_p(Q^{n+1}) = Q^{n+1}

    r_p(Q^{n+1}) = F_p(Q^{n+1}) - Q^{n+1} = Δt f(Q^{n+1}, t^{n+1}) + Q^n - Q^{n+1} = -r_n(Q^{n+1})

    Q^{n+1,k+1} = F_p(Q^{n+1,k}) = Q^{n+1,k} - (F_n(Q^{n+1,k}) - Q^n)


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
- `Q`: Q^n
- `Qrhs`: unused, included to fit general nonlinearsolve! interface
- `solver`: linear solver
...
"""
function dononlineariteration!(
    solver::PicardStandardSolver,
    rhs!,
    Q,
    Qrhs, # unused
    threshold,
    iters,
    args...,
)
    R = solver.residual
    FQ = solver.FQ
    Q .= FQ

    # Compute residual norm and residual for next step
    rhs!(FQ, Q, args...)
    R .= FQ .- Q
    resnorm = norm(R, weighted_norm)
    converged = check_convergence(resnorm, threshold, iters)
    return converged
end
