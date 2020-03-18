module GeneralizedMinimalResidualSolver

export GeneralizedMinimalResidual

using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays: device, realview

using LinearAlgebra
using LazyArrays
using StaticArrays
using GPUifyLoops

"""
    GeneralizedMinimalResidual(Q; M, rtol, atol)

This is an object for solving linear systems using an iterative Krylov method.
The constructor parameter `M` is the number of steps after which the algorithm
is restarted (if it has not converged), `Q` is a reference state used only
to allocate the solver internal state, and `rtol` specifies the convergence
criterion based on the relative residual norm. The amount of memory
required for the solver state is roughly `(M + 1) * size(Q)`.
This object is intended to be passed to the [`linearsolve!`](@ref) command.

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
mutable struct GeneralizedMinimalResidual{M, MP1, MMP1, T, AT} <: LS.AbstractIterativeLinearSolver
  krylov_basis::NTuple{MP1, AT}
  "Hessenberg matrix"
  H::MArray{Tuple{MP1, M}, T, 2, MMP1}
  "rhs of the least squares problem"
  g0::MArray{Tuple{MP1, 1}, T, 2, MP1}
  rtol::T
  atol::T

  function GeneralizedMinimalResidual(Q::AT; M=min(20, eltype(Q)), rtol=√eps(eltype(AT)),
                                      atol=eps(eltype(AT))) where AT
    krylov_basis = ntuple(i -> similar(Q), M + 1)
    H = @MArray zeros(M + 1, M)
    g0 = @MArray zeros(M + 1)

    new{M, M + 1, M * (M + 1), eltype(Q), AT}(krylov_basis, H, g0, rtol, atol)
  end
end

const weighted = false

function LS.initialize!(linearoperator!, Q, Qrhs,
                        solver::GeneralizedMinimalResidual, args...)
    g0 = solver.g0
    krylov_basis = solver.krylov_basis
    rtol, atol = solver.rtol, solver.atol

    @assert size(Q) == size(krylov_basis[1])

    # store the initial residual in krylov_basis[1]
    linearoperator!(krylov_basis[1], Q, args...)
    @. krylov_basis[1] = Qrhs - krylov_basis[1]

    threshold = rtol * norm(krylov_basis[1], weighted)
    residual_norm = norm(krylov_basis[1], weighted)

    converged = false
    # FIXME: Should only be true for threshold zero
    if threshold < atol
      converged = true
      return converged, threshold
    end

    fill!(g0, 0)
    g0[1] = residual_norm
    krylov_basis[1] ./= residual_norm

    converged, max(threshold, atol)
end

function LS.doiteration!(linearoperator!, Q, Qrhs,
                         solver::GeneralizedMinimalResidual{M}, threshold,
                         args...) where M

  krylov_basis = solver.krylov_basis
  H = solver.H
  g0 = solver.g0

  converged = false
  residual_norm = typemax(eltype(Q))

  Ω = LinearAlgebra.Rotation{eltype(Q)}([])
  j = 1
  for outer j = 1:M

    # Arnoldi using the Modified Gram Schmidt orthonormalization
    linearoperator!(krylov_basis[j + 1], krylov_basis[j], args...)
    for i = 1:j
      H[i, j] = dot(krylov_basis[j + 1], krylov_basis[i], weighted)
      @. krylov_basis[j + 1] -= H[i, j] * krylov_basis[i]
    end
    H[j + 1, j] = norm(krylov_basis[j + 1], weighted)
    krylov_basis[j + 1] ./= H[j + 1, j]

    # apply the previous Givens rotations to the new column of H
    @views H[1:j, j:j] .= Ω * H[1:j, j:j]

    # compute a new Givens rotation to zero out H[j + 1, j]
    G, _ = givens(H, j, j + 1, j)

    # apply the new rotation to H and the rhs
    H .= G * H
    g0 .= G * g0

    # compose the new rotation with the others
    Ω = lmul!(G, Ω)

    residual_norm = abs(g0[j + 1])

    if residual_norm < threshold
      converged = true
      break
    end
  end

  # solve the triangular system
  y = SVector{j}(@views UpperTriangular(H[1:j, 1:j]) \ g0[1:j])

  ## compose the solution
  rv_Q = realview(Q)
  rv_krylov_basis = realview.(krylov_basis)
  threads = 256
  blocks = div(length(rv_Q) + threads - 1, threads)
  @launch(device(Q), threads = threads, blocks = blocks,
          LS.linearcombination!(rv_Q, y, rv_krylov_basis, true))

  # if not converged restart
  converged || LS.initialize!(linearoperator!, Q, Qrhs, solver, args...)

  (converged, j, residual_norm)
end

end
