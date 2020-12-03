export AbstractAccelerator,
    AndersonAccelerator,
    NGMRESAccelerator

"""
    AbstractAccelerator

This is an abstract type representing a generic accelerator that wraps another nonlinear solver.
"""
abstract type AbstractAccelerator <: AbstractNonlinearSolver end

"""
    internalsolver(::AbstractAccelerator)
    doacceleration!(::AbstractAccelerator, Q, k)
"""
function internalsolver end
function doacceleration! end

function initialize!(rhs!, Q, Qrhs, solver::AbstractAccelerator, args...)
    return initialize!(rhs!, Q, Qrhs, internalsolver(solver), args...)
end
atol(solver::AbstractAccelerator) = atol(internalsolver(solver))
rtol(solver::AbstractAccelerator) = rtol(internalsolver(solver))
maxiters(solver::AbstractAccelerator) = maxiters(internalsolver(solver))

function dononlineariteration!(
    solver::AbstractAccelerator,
    rhs!,
    Q,
    Qrhs,
    threshold,
    iters,
    args...;
    kwargs...
)
    nlsolver = internalsolver(solver)
    converged = dononlineariteration!(
        nlsolver,
        rhs!,
        Q,
        Qrhs,
        threshold,
        iters,
        args...;
        kwargs...
    )
    if !converged
        doacceleration!(solver, Q, iters)
        R = nlsolver.residual
        rhs!(R, Q, args...)
        R .-= Qrhs
        residual_norm = norm(R, weighted_norm)
        converged = check_convergence(residual_norm, threshold, iters)
    end
    return converged
end

"""
    struct AndersonAccelerator{Depth, FT, AT1, AT2, NLS}
        initial_residual_norm::FT
        ω::FT
        β::AT1
        x::AT1
        xprev::AT1
        g::AT1
        gcopy::AT1
        gprev::AT1
        Xβ::AT1
        Gβ::AT1
        X::AT2
        G::AT2
        Gcopy::AT2
        nlsolver::NLS
    end

...
# Arguments
- `Q`: MPIArray container
- `nlsolver`: AbstractNonlinearSolver implementation
- `depth`: the accelerator window size
- `ω`: relaxation parameter
...
"""
struct AndersonAccelerator{FT, AT1, AT2, NLS} <: AbstractAccelerator
    depth::Int                    # window size
    ω::FT                         # relaxation parameter
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
    nlsolver::NLS
end

function AndersonAccelerator(Q::AT, nlsolver::NLS; depth::Int = 1, ω = 1.) where {AT, NLS}
    FT = eltype(Q)

    # β could also be @MArray zeros(depth), but ldiv! is not defined for MArrays.
    @show typeof(vec(Q))
    β = similar(vec(Q), depth)
    x = similar(vec(Q))
    X = similar(x, length(x), depth)

    AndersonAccelerator(
        depth, FT(ω), β,
        x, similar(x), similar(x), similar(x), similar(x), similar(x), similar(x),
        X, similar(X), similar(X),
        nlsolver
    )
end

internalsolver(a::AndersonAccelerator) = a.nlsolver

function doacceleration!(a::AndersonAccelerator, Q, k)
    depth = a.depth
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

#=
Simon's Notes:

BackwardEulerProblem(f,alpha,Q_star): find Q such that
Q - α * f(Q) = Q_star
approximate as f(Q) = A(Q)*Q + b(Q)
Q - α * A(Q)*Q + α * b(Q) = Q_star
(I - α * A(Q)) * Q + α * b(Q) = Q_star
(I - α * A(Q)) * Q = Q_star - α * b(Q)
Picard General: (I - α * A(Q_k)) * Q_{k+1} = Q_star - α * b(Q_k)
Picard Standard: A(Q_k) = 0

- if f is linear, we can do a linear solver (either direct or iterative)
- if we can compute jacobian action `J(f,Q0)(ΔQ)` (Jacobian of `f` at `Q0`, with tangent `ΔQ`)
  then at each iteration you're solving a linear BackwardEulerProblem(J(f,Q0),alpha, Q_star - Q0)
- BasicPicard() / Fixed point iterations 
  solve via: Q_k+1 = α * f(Q_k) + Q_star
- SplitPicard(linearsolver): BackwardEulerProblem(ApproximateJacobian(A, b), alpha, Q_star)

f(Q) =A(Q_k, Q_star, α, f)*Q_k+1 = b(Q_k)

F_n(Q) = Q - Δt f(Q, t^{n+1})    
F_p(Q) = Δt f(Q, t^{n+1}) + Q^n

linear solver: find Qp such that 
  Qp - α * f(Qp,t) = Q 

nonlinear prob `f` is no longer linear, at each iteration
  (Q0 + ΔQ) - α J_f(Q0,t)*ΔQ = Q_star
  ΔQ - α J_f(Q0,t)*ΔQ = Q_star - Q0


AbstractSolverAlgorithm
- AbstractNonlinearSolverAlgorithm
  - NewtonKrylov(linearsolver, jacobianaction) where jacobianaction = 

    at each iteration, we're using `linearsolver` to solve `jacobianaction(nonlinearprob, v)`

- AbstractLinearSolverAlgorithm
  - DirectLinearSolverAlgorithm
    - ColumnwiseLU
  - IterativeLinearSolverAlgorithm
    - GMRES
    - CG
=#

#=
Dennis's Notes:

Perhaps "Basic" and "Split" should be replaced with "ZerothOrder" and "FirstOrder"?
Perhaps "Matrix" and "Vector" should be replaced with "Linear" and "Constant"?

The SystemSolver contains basic algorithm information (atol, maxiters, etc.), as well as cache of arrays for intermediate computations, and all of these things are immutable.
The Problem contains information specific to the current backward euler step, so either it has to be mutable, or it has to be constructed anew for each backward euler step.
  Specifically, different backward euler steps require different values of Δt, which means that they require different functions f!, fMatrix!, and fVector!.
  It is possible to also make the Problem immutable, but this would require passing around Δt in addition to the Problem and calling, e.g., f! as f!(fQ, Q, Δt, args...).
Should the SystemSolver be split into seperate Algorithm and SolverCache structs?
  Aside from generalizability, what would be the benefit of doing so?
  If we did this, we would have to pass around three objects (Algorithm, SolverCache, and Problem), while only needing to specialize methods on one of them (Algorithm).
Since the SystemSolver type determines the Problem type, perhaps we should also store the problem information in the SystemSolver?
  This would let us get away with passing around just a single object.
  On the other hand, if we did this, the SystemSolver would probably have to be mutable. Is there any downside to this?
Does the solution array Q belong in the SystemSolver or in the Problem? The notes below assume that it is in the Problem.
Do cached arrays like fQ, fMatrixQ, and R belong in the SystemSolver or in the Problem? The notes below assume that they are in the SystemSolver.

Problem
- StandardProblem(Q, f!, frhs::Array)               # Find Q such that f(Q) = frhs, where f!(fQ, Q, args...) sets fQ = f(Q)
- BasicFixedPointProblem(Q, f!)                     # Find Q such that Q = f(Q), where f!(fQ, Q, args...) sets fQ = f(Q)
- ModifiedFixedPointProblem(Q, fMatrix!, fVector!)  # Find Q such that fMatrix(Q) * Q = fVector(Q), where fMatrix!(fMatrixQ, Q, args...) sets fMatrixQ = fMatrix(Q) and ...
                                                    # BasicFixedPointProblem is equivalent to SplitFixedPointProblem with fMatrix(Q) = I and fVector(Q) = f(Q)
                                                    # For Richard's Equation, the standard SplitFixedPointProblem involves a tri-diagonal fMatrix.

ODESolver
- BackwardEulerSolver: (Q^{n+1} - Q^n)/Δt = ∂Q∂t(Q^{n+1}, t^{n+1}, args...) or (Q^{n+1} - Q^n)/Δt = ∂Q∂tMatrix(Q^{n+1}, t^{n+1}, args...) * Q^{n+1} + ∂Q∂tVector(Q^{n+1}, t^{n+1}, args...)
  * StandardProblem: Q^{n+1} - Δt * ∂Q∂t(Q^{n+1}, t^{n+1}, args...) = Q^n,
                     so f(Q) = Q - Δt * ∂Q∂t(Q, t^{n+1}, args...) and frhs = Q^n
  * BasicFixedPointProblem: Q^{n+1} = Δt * ∂Q∂t(Q^{n+1}, t^{n+1}, args...) + Q^n,
                           so f(Q) = Δt * ∂Q∂t(Q, t^{n+1}, args...) + Q^n
  * SplitFixedPointProblem: (I - Δt * ∂Q∂tMatrix(Q^{n+1}, t^{n+1}, args...)) * Q^{n+1} = Δt * ∂Q∂tVector(Q^{n+1}, t^{n+1}, args...) + Q^n,
                           so fMatrix(Q) = I - Δt * ∂Q∂tMatrix(Q, t^{n+1}, args...) and fVector(Q) = Δt * ∂Q∂tVector(Q, t^{n+1}, args...) + Q^n
  # BackwardEulerSolver chooses Problem type based on IterativeSolver type and whether it was provided with ∂Q∂t or both ∂Q∂tMatrix and ∂Q∂tVector

ImplicitTendencySolver(::SystemSolver)

SystemSolver
- DirectSolver
  - DirectLinearSolver
    - ColumnLUSolver
      - SingleColumnLUSolver
      - ManyColumnLUSolver
- IterativeSolver
  - IterativeLinearSolver(preconditioner::Preconditioner, atol::AbstractFloat, rtol::AbstractFloat, maxiters::Integer) # Only solves StandardProblem
    - ConjugateGradientSolver
    - GeneralizedConjugateResidualSolver
    - GeneralizedMinimalResidualSolver
    - BatchedGeneralizedMinimalResidualSolver
  - IterativeNonlinearSolver(atol::AbstractFloat, rtol::AbstractFloat, maxiters::Integer)
    - BasicPicardSolver                                                                              # Only solves BasicFixedPointProblem:    Q^{k+1} = f(Q^k)
    - SplitPicardSolver(linearsolver::IterativeLinearSolver)                                         # Only solves ModifiedFixedPointProblem: Q^{k+1} = fMatrix(Q^k)⁻¹ * fVector(Q^k)
    - JacobianFreeNewtonKrylovSolver(linearsolver::IterativeLinearSolver, finite_or_auto_diff::Bool) # Only solves StandardProblem:           Q^{k+1} = Q^k - df/dQ(Q^k)⁻¹ * (f(Q^k) - frhs)
  - Accelerator(nonlinearsolver::IterativeNonlinearSolver, depth::Integer)
    - AndersonAccelerator
    - NonlinearGeneralizedMinimalResidualSolver

  Methods:
  * function solve!(iterativesolver::IterativeSolver, problem::Problem, args...)
        iters = 0
        initial_residual_norm = initialize!(iterativesolver, problem, args...)
        initial_residual_norm < atol(solver) && return iters
        threshold = max(atol(solver), rtol(solver) * initial_residual_norm)
        m = maxiters(solver)
        has_converged = false
        while !has_converged && iters < m
            residual_norm = doiteration!(iterativesolver, problem, args...)
            converged = check_convergence(residual_norm, threshold, iters)
            iters += 1
        end
        has_converged || @warn "$(typeof(iterativesolver)) did not converge after $iters iterations"
        return iters
    end
  
  * function check_convergence(residual_norm, threshold, iters)
        isfinite(residual_norm) || error("Norm of residual is not finite after $iters iterations")
        return residual_norm < threshold
    end

  Interface:
  * function atol(::IterativeSolver)::AbstractFloat end
  * function rtol(::IterativeSolver)::AbstractFloat end
  * function maxiters(::IterativeSolver)::Integer end
  * function residual!(::IterativeSolver, ::Problem, args...)::AbstractFloat end # Computes residual and returns its norm
  * function initialize!(::IterativeSolver, ::Problem, args...)::AbstractFloat end # Initializes IterativeSolver's cache and returns residual!
  * function doiteration!(::IterativeSolver, ::Problem, threshold::AbstractFloat, iters::Integer, args...)::AbstractFloat end # Updates IterativeSolver's cache and Problem's Q, and returns residual!

  Implementation for Accelerator:
  * atol(s::Accelerator) = atol(s.nonlinearsolver)
  * rtol(s::Accelerator) = rtol(s.nonlinearsolver)
  * maxiters(s::Accelerator) = maxiters(s.nonlinearsolver)
  * residual!(s::Accelerator, p::Problem, args...) = residual!(s.nonlinearsolver, p, args...)
  * initialize!(s::Accelerator, p::Problem, args...) = initialize!(s.nonlinearsolver, p, args...)
  * function doiteration!(s::Accelerator, p::Problem, threshold::AbstractFloat, iters::Integer, args...)
        nonlinearsolver = s.nonlinearsolver
        residual_norm = doiteration!(nonlinearsolver, p, threshold, iters, args...)
        if !check_convergence(residual_norm, threshold, iters)
            accelerate!(s, p, threshold, iters, args...)
            residual_norm = residual!(nonlinearsolver, p, args...)
        end
        return residual_norm
    end
  * function accelerate!(::Accelerator, ::Problem, threshold::AbstractFloat, iters::Integer, args...)::Nothing end # Updates Accelerator's cache Problem's Q

  Implementation for BasicPicardSolver:
  * atol(s::BasicPicardSolver) = s.atol
  * rtol(s::BasicPicardSolver) = s.rtol
  * maxiters(s::BasicPicardSolver) = s.maxiters
  * function residual!(s::BasicPicardSolver, p::BasicFixedPointProblem, args...)
        p.f!(s.fQ, p.Q, args...)
        s.R .= s.fQ .- p.Q
        return norm(s.R, weighted_norm)
    end
  * initialize!(s::BasicPicardSolver, p::Problem, args...) = residual!(s, p, args...)
  * function doiteration!(s::BasicPicardSolver, p::BasicFixedPointProblem, args...)
        p.Q .= s.fQ
        return residual!(s, p, args...)
    end
=#