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

Perhaps "Basic" and "Modified" should be replaced with "ZerothOrder" and "FirstOrder"?
Perhaps "Matrix" and "Vector" should be replaced with "Linear" and "Constant"?
Perhaps the threshold should be stored in the Problem, since the latter is already mutable?

Problem
- StandardProblem(Q, f!, frhs)                     # Find Q such that f(Q) = frhs, where f!(fQ, Q, args...) sets fQ = f(Q)
- BasicFixedPointProblem(Q, f!)                    # Find Q such that Q = f(Q), where f!(fQ, Q, args...) sets fQ = f(Q)
- ModifiedFixedPointProblem(Q, fMatrix!, fVector!) # Find Q such that fMatrix(Q) * Q = fVector(Q), where fMatrix!(fMatrixQ, Q, args...) sets fMatrixQ = fMatrix(Q) and fVector!(fVectorQ, Q, args...) sets ...
                                                   # BasicFixedPointProblem is equivalent to ModifiedFixedPointProblem with fMatrix(Q) = I and fVector(Q) = f(Q)
                                                   # For Richard's Equation, the typical ModifiedFixedPointProblem involves a tri-diagonal fMatrixQ.

https://diffeq.sciml.ai/dev/features/performance_overloads/#performance_overloads

dqdt = ODEFunction(f, jvp=f2)
SplitODEFunction(f1,f2) # dqdt = f1(q,t) + f2(q,t)

dqdt = f(q,t)

find q such that:
q - γ f(q,t) = qp

1) if f(q,t) = Lq (linear, homogeneous)
    - (I - γ L) q = qp
2) if f(q,t) = L(t) * q (linear, inhomogeneous)
    - (I - γ L(t)) q = qp
3) f(q,t) = L(q,t) q + g(q,t)
    - (I - γ L(q,t)) q = qp + γ g(q,t)
      - Modified Picard => (I - γ L(q_n,t)) q_{n+1} = qp + γ g(q_n,t)
    - if g(q,t) == 0, and L(q,t) == L(t) then equivalent to (1/2)
4) f(q,t) has Jacobian (or some approximation) J_f(q,t)
    - solve h(q,t) = q - γ f(q,t) - qp = 0 via Newton's method
      => J_h(q,t) = I - γ J_f(q,t)
    - gives iterations of the form
        q_{n+1} = q_n - J_h(q_n,t) \ h(q_n,t)   
        =>   J_h * (q_{n+1} - q_n) = -h(q_n,t)
        =>   [I - γ J_f(q_n,t)] * (q_{n+1} - q_n) = - [(q_n - γ f(q_n,t)) - qp]


1) Backward Euler ODE timestepper dqdt(q,t) = f(q,t)
  for each step
    solve!(q_next, BackwardEulerProblem(f,dt*alpha,q), solver)

2) IMEX scheme: dqdt(q,t) = f1(q,t) + f2(q,t)
    https://clima.github.io/TimeMachine.jl/dev/background/ark/
  for each step
    for each stage
      x = solve(BackwardEulerProblem(f1, dt*alpha[i,i], y), solver) 

      if f1(q,t) = Lq (linear, homogeneous)
        can either
          - use a direct solver: 
            form L (banded), and compute lu(L) at t = 0, reuse factorization
          - use an iterative solver:
            specify and reuse the preconditioner
        solve something that looks like  W = I - dt*alpha[i,i]*L => W x = y
        # at ODE init, or if `dt` or `alpha[i,i]` changes
         W = BackwardEulerOperator(ODEFunction(f1, jac=L)), dt*alpha[i,i])
         Wfact = factorize(W)
        # at each stage
         solve!(x, ImplicitProblem(Wfact, y), solver) 

        

        
        
      
ImplicitProblem(h, Qp): h(Q,t) = Qp
    if h = BackwardEulerOperator(f,γ)  => h(Q,t) = Q - γ f(Q,t)
    const BackwardEulerProblem = ImplicitProblem{BackwardEulerOperator}
    

    
# ImplicitTendencyAlgorithm solves either
  (Q^{n+1} - Q^n)/(α * Δt) = ∂Q∂t(Q^{n+1}, t^{n+1}, args...) or
  (Q^{n+1} - Q^n)/(α * Δt) = ∂Q∂tMatrix(Q^{n+1}, t^{n+1}, args...) * Q^{n+1} + ∂Q∂tVector(Q^{n+1}, t^{n+1}, args...)
# ImplicitTendencyAlgorithm chooses Problem type based on IterativeAlgorithm type and whether it was provided with ∂Q∂t or both ∂Q∂tMatrix and ∂Q∂tVector
  * StandardProblem:           Q^{n+1} - α * Δt * ∂Q∂t(Q^{n+1}, t^{n+1}, args...) = Q^n,
                               so f(Q) = Q - α * Δt * ∂Q∂t(Q, t^{n+1}, args...) and frhs = Q^n
  * BasicFixedPointProblem:    Q^{n+1} = α * Δt * ∂Q∂t(Q^{n+1}, t^{n+1}, args...) + Q^n,
                               so f(Q) = α * Δt * ∂Q∂t(Q, t^{n+1}, args...) + Q^n
  * ModifiedFixedPointProblem: (I - α * Δt * ∂Q∂tMatrix(Q^{n+1}, t^{n+1}, args...)) * Q^{n+1} = α * Δt * ∂Q∂tVector(Q^{n+1}, t^{n+1}, args...) + Q^n,
                               so fMatrix(Q) = I - α * Δt * ∂Q∂tMatrix(Q, t^{n+1}, args...) and fVector(Q) = α * Δt * ∂Q∂tVector(Q, t^{n+1}, args...) + Q^n
# Since α changes on each iteration of the ImplicitTendencyAlgorithm, the Problem must be mutable (otherwise, it would have to be re-constructed on each iteration).

Legend:
  - Denotes an abstract type
  > Denotes a concrete type
  The fields written after abstract types actually belong to their concrete subtypes

- Algorithm
  - SystemAlgorithm
    - DirectAlgorithm
      - DirectLinearAlgorithm
        - ColumnLUAlgorithm
          > SingleColumnLUAlgorithm{...}(...)
          > ManyColumnLUAlgorithm{...}(...)
    - IterativeAlgorithm
      - IterativeLinearAlgorithm{PT, FT}(preconditioner::PT, atol::FT, rtol::FT, maxiters::Int)              # Only solves StandardProblem
        > ConjugateGradientAlgorithm{...}(...)
        > GeneralizedConjugateResidualAlgorithm{...}(...)
        > GeneralizedMinimalResidualAlgorithm{...}(...)
        > BatchedGeneralizedMinimalResidualAlgorithm{...}(...)
      - IterativeNonlinearAlgorithm{FT}(atol::FT, rtol::FT, maxiters::Int)
        > BasicPicardAlgorithm                                                                               # Only solves BasicFixedPointProblem:    Q^{k+1} = f(Q^k)
        > ModifiedPicardAlgorithm{ILAT}(iterativelinearalgorithm::ILAT)                                      # Only solves ModifiedFixedPointProblem: Q^{k+1} = fMatrix(Q^k)⁻¹ * fVector(Q^k)
        > JacobianFreeNewtonKrylovAlgorithm{ILAT}(iterativelinearalgorithm::ILAT, finite_or_auto_diff::Bool) # Only solves StandardProblem:           Q^{k+1} = Q^k - df/dQ(Q^k)⁻¹ * (f(Q^k) - frhs)
      - AcceleratedAlgorithm{INAT}(iterativenonlinearalgorithm::INAT, depth::Int)                            # Only solves Problem that can be solved by iterativenonlinearalgorithm
        > AndersonAcceleratedAlgorithm
        > NonlinearGeneralizedMinimalResidualAlgorithm
  > ImplicitTendencyAlgorithm{SAT}(systemalgorithm::SAT, preconditioner_update_freq::Int)                    # The preconditioner_update_freq does not belong here, so where should it go???

- Cache
  - SystemCache
    - DirectCache
      - DirectLinearCache
        - ColumnLUCache
          > SingleColumnLUCache{...}(...)
          > ManyColumnLUCache{...}(...)
    - IterativeCache
      - IterativeLinearCache
        > ConjugateGradientCache{...}(...)
        > GeneralizedConjugateResidualCache{...}(...)
        > GeneralizedMinimalResidualCache{...}(...)
        > BatchedGeneralizedMinimalResidualCache{...}(...)
      - IterativeNonlinearCache
        > BasicPicardCache{AT}(fQ::AT, residual::AT)
        > ModifiedPicardCache{ILCT, MAT, VAT}(iterativelinearcache::ILCT, fMatrixQ::MAT, fVectorQ::VAT, residual::VAT)
        > JacobianFreeNewtonKrylovCache{ILCT, JVPT, AT}(iterativelinearcache::ILCT, jvp!::JVPT, ΔQ::AT, residual::AT)
      - AcceleratedCache{INCT}(iterativenonlinearcache::INCT)
        > AndersonAcceleratedCache{...}(...)
        > NonlinearGeneralizedMinimalResidualCache{...}(...)
  > ImplicitTendencyCache{SCT}(systemcache::SCT)

  Methods used for every IterativeAlgorithm:
  * function solve!(iterativealgorithm::IterativeAlgorithm, iterativecache::IterativeCache, problem::Problem, args...)
        iters = 0
        initial_residual_norm = initialize!(iterativealgorithm, iterativecache, problem, args...)
        initial_residual_norm < atol(solver) && return iters
        threshold = max(atol(solver), rtol(solver) * initial_residual_norm)
        m = maxiters(solver)
        has_converged = false
        while !has_converged && iters < m
            residual_norm = doiteration!(iterativealgorithm, iterativecache, problem, args...)
            converged = check_convergence(residual_norm, threshold, iters)
            iters += 1
        end
        has_converged || @warn "$(typeof(iterativealgorithm)) did not converge after $iters iterations"
        return iters
    end
  * function check_convergence(residual_norm, threshold, iters)
        isfinite(residual_norm) || error("Norm of residual is not finite after $iters iterations")
        return residual_norm < threshold
    end

  Interface that must be provided by every IterativeAlgorithm:
  * function allocatecache(::Algorithm, ::AbstractArray)::Cache end                                                                  # Generates the Cache that corresponds to the given Algorithm
  * function atol(::IterativeAlgorithm)::AbstractFloat end
  * function rtol(::IterativeAlgorithm)::AbstractFloat end
  * function maxiters(::IterativeAlgorithm)::Integer end
  * function residual!(::IterativeAlgorithm, ::IterativeCache, ::Problem, args...)::AbstractFloat end                                # Computes residual and returns its norm
  * function initialize!(::IterativeAlgorithm, ::IterativeCache, ::Problem, args...)::AbstractFloat end                              # Initializes IterativeCache and returns residual!
  * function doiteration!(::IterativeAlgorithm, ::IterativeCache, ::Problem, ::AbstractFloat, ::Integer, args...)::AbstractFloat end # Updates IterativeCache and Problem's Q, and returns residual!

  Implementation for AcceleratedAlgorithm:
  * allocatecache(a::AndersonAcceleratedAlgorithm, Q::AbstractArray) =
        AndersonAcceleratedCache(allocatecache(a.iterativenonlinearalgorithm, Q), similar(Q), ...)
  * allocatecache(a::NonlinearGeneralizedMinimalResidualAlgorithm, Q::AbstractArray) =
        NonlinearGeneralizedMinimalResidualCache(allocatecache(a.iterativenonlinearalgorithm, Q), similar(Q), ...)
  * atol(a::AcceleratedAlgorithm) = atol(s.iterativenonlinearalgorithm)
  * rtol(a::AcceleratedAlgorithm) = rtol(s.iterativenonlinearalgorithm)
  * maxiters(a::AcceleratedAlgorithm) = maxiters(s.iterativenonlinearalgorithm)
  * residual!(a::AcceleratedAlgorithm, c::AcceleratedCache, p::Problem, args...) =
        residual!(s.iterativenonlinearalgorithm, c.iterativenonlinearcache, p, args...)
  * initialize!(a::AcceleratedAlgorithm, c::AcceleratedCache, p::Problem, args...) =
        initialize!(s.iterativenonlinearalgorithm, c.iterativenonlinearcache, p, args...)
  * function doiteration!(a::AcceleratedAlgorithm, c::AcceleratedCache, p::Problem, threshold::AbstractFloat, iters::Integer, args...)
        a2 = a.iterativenonlinearalgorithm
        c2 = c.iterativenonlinearcache
        residual_norm = doiteration!(a2, c2, p, threshold, iters, args...)
        if !check_convergence(residual_norm, threshold, iters)
            accelerate!(a, c, p, threshold, iters, args...)
            residual_norm = residual!(a2, c2, p, args...)
        end
        return residual_norm
    end
  * function accelerate!(::AndersonAcceleratedAlgorithm, ::AndersonAcceleratedCache, ::Problem, ::AbstractFloat, ::Integer, args...)
        ... # Update AndersonAcceleratedCache and Problem's Q
        return Nothing
    end
  * function accelerate!(::NonlinearGeneralizedMinimalResidualAlgorithm, ::NonlinearGeneralizedMinimalResidualCache, ::Problem, ::AbstractFloat, ::Integer, args...)
        ... # Update NonlinearGeneralizedMinimalResidualCache and Problem's Q
        return Nothing
    end

  Implementation for BasicPicardAlgorithm:
  * allocatecache(a::BasicPicardAlgorithm, Q::AbstractArray) = BasicPicardCache(similar(Q), similar(Q))
  * atol(a::BasicPicardAlgorithm) = a.atol
  * rtol(a::BasicPicardAlgorithm) = a.rtol
  * maxiters(a::BasicPicardAlgorithm) = a.maxiters
  * function residual!(a::BasicPicardAlgorithm, c::BasicPicardCache, p::BasicFixedPointProblem, args...)
        p.f!(c.fQ, p.Q, args...)
        c.residual .= c.fQ .- p.Q
        return norm(c.residual, weighted_norm)
    end
  * initialize!(a::BasicPicardAlgorithm, c::BasicPicardCache, p::BasicFixedPointProblem, args...) = residual!(a, c, p, args...)
  * function doiteration!(a::BasicPicardAlgorithm, c::BasicPicardCache, p::BasicFixedPointProblem, threshold::AbstractFloat, iters::Integer, args...)
        p.Q .= c.fQ
        return residual!(a, c, p, args...)
    end
=#