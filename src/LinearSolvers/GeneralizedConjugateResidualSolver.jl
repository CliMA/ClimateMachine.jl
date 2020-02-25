module GeneralizedConjugateResidualSolver

export GeneralizedConjugateResidual

using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays: device, realview

using LinearAlgebra
using LazyArrays
using StaticArrays
using GPUifyLoops

"""
    GeneralizedConjugateResidual(K, Q; rtol, atol)

This is an object for solving linear systems using an iterative Krylov method.
The constructor parameter `K` is the number of steps after which the algorithm
is restarted (if it has not converged), `Q` is a reference state used only
to allocate the solver internal state, and `tolerance` specifies the convergence
criterion based on the relative residual norm. The amount of memory
required by the solver state is roughly `(2K + 2) * size(Q)`.
This object is intended to be passed to the [`linearsolve!`](@ref) command.

This uses the restarted Generalized Conjugate Residual method of Eisenstat (1983).

### References

    @article{eisenstat1983variational,
      title={Variational iterative methods for nonsymmetric systems of linear equations},
      author={Eisenstat, Stanley C and Elman, Howard C and Schultz, Martin H},
      journal={SIAM Journal on Numerical Analysis},
      volume={20},
      number={2},
      pages={345--357},
      year={1983},
      publisher={SIAM}
    }
"""
mutable struct GeneralizedConjugateResidual{K, T, AT} <: LS.AbstractIterativeLinearSolver
  residual::AT
  L_residual::AT
  p::NTuple{K, AT}
  L_p::NTuple{K, AT}
  alpha::MArray{Tuple{K}, T, 1, K}
  normsq::MArray{Tuple{K}, T, 1, K}
  rtol::T
  atol::T

  function GeneralizedConjugateResidual(K, Q::AT; rtol=√eps(eltype(AT)), atol=eps(eltype(AT))) where AT
    T = eltype(Q)

    residual = similar(Q)
    L_residual = similar(Q)
    p = ntuple(i -> similar(Q), K)
    L_p = ntuple(i -> similar(Q), K)
    alpha = @MArray zeros(K)
    normsq = @MArray zeros(K)

    new{K, T, AT}(residual, L_residual, p, L_p, alpha, normsq, rtol, atol)
  end
end

const weighted = false

function LS.initialize!(linearoperator!, Q, Qrhs,
                        solver::GeneralizedConjugateResidual, args...)
    residual = solver.residual
    p = solver.p
    L_p = solver.L_p

    @assert size(Q) == size(residual)
    rtol, atol = solver.rtol, solver.atol

    threshold = rtol * norm(Qrhs, weighted)
    linearoperator!(residual, Q, args...)
    residual .-= Qrhs

    converged = false
    residual_norm = norm(residual, weighted)
    if residual_norm < threshold
      converged = true
      return converged, threshold
    end

    p[1] .= residual
    linearoperator!(L_p[1], p[1], args...)

    threshold = max(atol, threshold)

    converged, threshold
end

function LS.doiteration!(linearoperator!, Q, Qrhs,
                         solver::GeneralizedConjugateResidual{K}, threshold,
                         args...) where K

  residual = solver.residual
  p = solver.p
  L_residual = solver.L_residual
  L_p = solver.L_p
  normsq = solver.normsq
  alpha = solver.alpha

  residual_norm = typemax(eltype(Q))
  for k = 1:K
    normsq[k] = norm(L_p[k], weighted) ^ 2
    beta = -dot(residual, L_p[k], weighted) / normsq[k]

    Q .+= beta * p[k]
    residual .+= beta * L_p[k]

    residual_norm = norm(residual, weighted)

    if residual_norm <= threshold
      return (true, k, residual_norm)
    end

    linearoperator!(L_residual, residual, args...)

    for l = 1:k
      alpha[l] = -dot(L_residual, L_p[l], weighted) / normsq[l]
    end

    if k < K
      rv_nextp = realview(p[k + 1])
      rv_L_nextp = realview(L_p[k + 1])
    else # restart
      rv_nextp = realview(p[1])
      rv_L_nextp = realview(L_p[1])
    end

    rv_residual = realview(residual)
    rv_p = realview.(p)
    rv_L_p = realview.(L_p)
    rv_L_residual = realview(L_residual)

    threads = 256
    blocks = div(length(rv_nextp) + threads - 1, threads)

    T = eltype(alpha)
    @launch(device(Q), threads = threads, blocks = blocks,
            LS.linearcombination!(rv_nextp, (one(T), alpha[1:k]...),
                                  (rv_residual, rv_p[1:k]...), false))

    @launch(device(Q), threads = threads, blocks = blocks,
            LS.linearcombination!(rv_L_nextp, (one(T), alpha[1:k]...),
                                  (rv_L_residual, rv_L_p[1:k]...), false))
  end

  (false, K, residual_norm)
end

end
