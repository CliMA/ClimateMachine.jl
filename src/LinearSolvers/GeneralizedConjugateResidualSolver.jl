module GeneralizedConjugateResidualSolver

export GeneralizedConjugateResidual

using ..LinearSolvers
using ..MPIStateArrays

using LinearAlgebra
using LazyArrays
using StaticArrays

const LS = LinearSolvers

struct GeneralizedConjugateResidual{K, T, AT} <: LS.AbstractIterativeLinearSolver
  residual::AT
  L_residual::AT
  p::NTuple{K, AT}
  L_p::NTuple{K, AT}
  alpha::MArray{Tuple{K}, T, 1, K}
  normsq::MArray{Tuple{K}, T, 1, K}
  tolerance::T

  function GeneralizedConjugateResidual(K, Q::AT, tolerance) where AT
    T = eltype(Q)

    residual = similar(Q)
    L_residual = similar(Q)
    p = ntuple(i -> similar(Q), K)
    L_p = ntuple(i -> similar(Q), K)
    alpha = @MArray zeros(K)
    normsq = @MArray zeros(K)

    new{K, T, AT}(residual, L_residual, p, L_p, alpha, normsq, tolerance)
  end
end

function LS.initialize!(linearoperator!, Q, Qrhs, solver::GeneralizedConjugateResidual)
    residual = solver.residual
    p = solver.p
    L_p = solver.L_p

    @assert size(Q) == size(residual)

    linearoperator!(residual, Q)
    residual .-= Qrhs
    
    p[1] .= residual
    linearoperator!(L_p[1], p[1])
end

function LS.doiteration!(linearoperator!, Q, solver::GeneralizedConjugateResidual{K}) where K
 
  residual = solver.residual
  p = solver.p
  L_residual = solver.L_residual
  L_p = solver.L_p
  normsq = solver.normsq
  alpha = solver.alpha
  tolerance = solver.tolerance

  weighted = true
  
  residual_norm = nothing
  for k = 1:K
    normsq[k] = norm(L_p[k], weighted) ^ 2
    beta = -dot(residual, L_p[k], weighted) / normsq[k]

    Q .+= beta * p[k]
    residual .+= beta * L_p[k]

    residual_norm = norm(residual, Inf, weighted)

    if residual_norm <= tolerance
      return (true, residual_norm)
    end

    linearoperator!(L_residual, residual)
  
    for l = 1:k
      alpha[l] = -dot(L_residual, L_p[l], weighted) / normsq[l]
    end

    # first build `Broadcasted` expressions for p_{k+1} and L_{k+1} to do only
    # one kernel call for each and simplify restart
    expr_p = residual
    expr_L_p = L_residual
    # FIXME: iterate over 1:k once broadcasting handles the final assignment correctly !
    for l = 1:k-1
      expr_p = @~ @. expr_p + alpha[l] * p[l]
      expr_L_p = @~ @. expr_L_p + alpha[l] * L_p[l]
    end

    if k < K
      p[k + 1] .= @. expr_p + alpha[k] * p[k]
      L_p[k + 1] .= @. expr_L_p + alpha[k] * L_p[k]
    else # restart
      p[1] .= @. expr_p + alpha[k] * p[k] 
      L_p[1] .= @. expr_L_p + alpha[k] * L_p[k]  
    end
  end
  
  (false, residual_norm)
end

end
