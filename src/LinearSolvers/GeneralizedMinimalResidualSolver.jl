module GeneralizedMinimalResidualSolver

export GeneralizedMinimalResidual

using ..LinearSolvers
using ..MPIStateArrays

using LinearAlgebra
using LazyArrays
using StaticArrays

const LS = LinearSolvers

"""
    GeneralizedMinimalResidual(K, Q, tolerance)

This is an object for solving linear systems using an iterative Krylov method.
The constructor parameter `K` is the number of steps after which the algorithm
is restarted, `Q` is a reference state used only to allocate the solver internal
state, and `tolerance` specifies the convergence threshold based on the residual
norm. Since the amount of additional memory required by the solver is 
`(K + 1) * size(Q)` in practical applications `K` should be kept small.
This object is intended to be passed to the [`linearsolve!`](@ref)
command.

This uses the restarted Generalized Minimal Residual method of Saad and Schultz (1986).

### References
    @article{saad1986gmres,
      title={GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems},
      author={Saad, Youcef and Schultz, Martin H},
      journal={SIAM Journal on scientific and statistical computing},
      volume={7},
      number={3},
      pages={856--869},
      year={1986},
      publisher={SIAM}
    }
"""
struct GeneralizedMinimalResidual{K, Ksq, T, AT} <: LS.AbstractIterativeLinearSolver
  residual::AT
  krylov_basis::NTuple{K, AT}
  H::MArray{Tuple{K, K}, T, 2, Ksq}
  givens_sin::MArray{Tuple{K}, T, 1, K}
  givens_cos::MArray{Tuple{K}, T, 1, K}
  beta::MArray{Tuple{K}, T, 1, K}
  tolerance::MArray{Tuple{1}, T, 1, 1}

  function GeneralizedMinimalResidual(K, Q::AT, tolerance) where AT
    T = eltype(Q)

    residual = similar(Q)
    krylov_basis = ntuple(i -> similar(Q), K)
    H = @MArray zeros(K, K)
    givens_sin = @MArray zeros(K)
    givens_cos = @MArray zeros(K)
    beta = @MArray zeros(K)

    new{K, K ^ 2, T, AT}(residual, krylov_basis, H, givens_sin, givens_cos, beta, (tolerance,))
  end
end

const weighted = true

function LS.initialize!(linearoperator!, Q, Qrhs, solver::GeneralizedMinimalResidual)
    residual = solver.residual
    beta = solver.beta
    krylov_basis = solver.krylov_basis

    @assert size(Q) == size(residual)

    linearoperator!(residual, Q)
    residual .*= -1
    residual .+= Qrhs

    residual_norm = norm(residual, weighted)
    beta[1] = residual_norm
    @. krylov_basis[1] = residual / residual_norm
end

function LS.doiteration!(linearoperator!, Q, Qrhs, solver::GeneralizedMinimalResidual{K}) where K
 
  residual = solver.residual
  krylov_basis = solver.krylov_basis
  H = solver.H
  givens_sin = solver.givens_sin
  givens_cos = solver.givens_cos
  beta = solver.beta
  tolerance = solver.tolerance[1]

  converged = false
  residual_norm = typemax(eltype(Q))
  
  k = 1
  for outer k = 1:K-1

    # Arnoldi
    linearoperator!(krylov_basis[k + 1], krylov_basis[k])
    for l = 1:k
      H[l, k] = dot(krylov_basis[k + 1], krylov_basis[l], weighted)
      @. krylov_basis[k + 1] -= H[l, k] * krylov_basis[l]
    end
    H[k + 1, k] = norm(krylov_basis[k + 1], weighted)
    krylov_basis[k + 1] ./= H[k + 1, k]

    # Givens rotation stuff
    for l = 1:k-1
      tmp = givens_cos[l] * H[l, k] + givens_sin[l] * H[l + 1, k]
      H[l + 1, k] = -givens_sin[l] * H[l, k] + givens_cos[l] * H[l + 1, k]
      H[l, k] = tmp
    end

    givens_sin[k], givens_cos[k] = normalize(SVector(H[k + 1, k], H[k, k]))

    H[k, k] = givens_cos[k] * H[k, k] + givens_sin[k] * H[k + 1, k]

    beta[k + 1] = -givens_sin[k] * beta[k]
    beta[k] *= givens_cos[k]

    residual_norm = abs(beta[k + 1])

    if residual_norm < tolerance
      converged = true
      break
    end
  end
  # @show k

  # reusing storage
  exp_coeffs = givens_cos

  # calculate the weights
  for i = reverse(1:k)
    t = beta[i]
    for j = reverse(i+1:k)
      t -= H[i,j] * exp_coeffs[j]
    end
    exp_coeffs[i] = t / H[i,i]
  end

  # compose the solution
  expr_Q = Q
  for l = 1:k
    expr_Q = @~ @. expr_Q + exp_coeffs[l] * krylov_basis[l]
  end
  Q .= expr_Q

  # if not converged restart
  converged || LS.initialize!(linearoperator!, Q, Qrhs, solver)
  
  (converged, residual_norm)
end

end
