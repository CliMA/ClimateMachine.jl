module ColumnwiseGMRESSolver

export StackGMRES, SingleColumnGMRES

using ..DGmethods
using ..DGmethods: BalanceLaw, DGModel
using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays: device, realview
using LinearAlgebra
using StaticArrays
using GPUifyLoops

abstract type AbstractColumnGMRESSolver{M} <: LS.AbstractIterativeLinearSolver end

"""

    StackGMRES(M, nhorzelem, Q::AT, tolerance) where AT

This solver performs the matrix-free generalised minimal residual method for
nodal columns residing in the same stack of elements. The stack-global
convergence criterion requires convergence in all nodal columns.
"""
struct StackGMRES{M, N, MP1, MMP1, T, I, AT} <: AbstractColumnGMRESSolver{M}
  krylov_basis::NTuple{MP1, AT}
  "Hessenberg matrix"
  H::NTuple{N, MArray{Tuple{MP1, M}, T, 2, MMP1}}
  "rhs of the least squares problem"
  g0::NTuple{N, MArray{Tuple{MP1, 1}, T, 2, MP1}}
  "Number of iterations until stop condition reached"
  stop_iter::MArray{Tuple{N}, I, 1, N}
  tolerances::MArray{Tuple{2}, T, 1, 2}

  function StackGMRES(Q::AT; rtol=√eps(eltype(AT)), atol=√eps(eltype(AT)),
                      M=min(20, length(Q)), nhorzelem::Int64=1
                     ) where {AT<:AbstractArray}
    krylov_basis = ntuple(i -> similar(Q), M + 1)
    H  = ntuple(x -> (@MArray zeros(M + 1, M)), nhorzelem)
    g0 = ntuple(x -> (@MArray zeros(M + 1)), nhorzelem)
    stop_iter = @MArray fill(-1, nhorzelem)

    new{M, nhorzelem, M + 1, M * (M + 1), eltype(Q), Int64, AT}(
        krylov_basis, H, g0, stop_iter, (rtol, atol))
  end
end

const weighted = false

function LS.initialize!(linearoperator!, Q, Qrhs,
                        solver::StackGMRES{M, nhorzelem}, args...) where {M, nhorzelem}
    ss = length(Q) ÷ nhorzelem
    krylov_basis = solver.krylov_basis
    FT = eltype(Q)

    @assert size(Q) == size(krylov_basis[1])

    # store the initial residual in krylov_basis[1]
    linearoperator!(krylov_basis[1], Q, args...)
    krylov_basis[1] .*= -1
    krylov_basis[1] .+= Qrhs

    rtol, atol = solver.tolerances
    converged = true

    threshold = rtol * norm(krylov_basis[1], weighted)

    for es in 1:nhorzelem
      stack = (es-1) * ss + 1 : es * ss

      g0 = solver.g0[es]
      @views residual_norm = norm(krylov_basis[1][stack], weighted)

      converged &= residual_norm < threshold
      if threshold < atol
        solver.stop_iter[es] = 0
        continue
      end

      fill!(g0, FT(0))
      g0[1] = residual_norm
      krylov_basis[1][stack] ./= residual_norm
    end

    converged, max(threshold, atol)
end

function LS.doiteration!(linearoperator!, Q, Qrhs, solver::StackGMRES{M, nhorzelem},
                         threshold, args...) where {M, nhorzelem}
  ss = length(Q) ÷ nhorzelem
  krylov_basis = solver.krylov_basis
  stop_iter = solver.stop_iter

  local max_residual_norm
  converged = false
  Gs = ntuple(x->LinearAlgebra.Rotation{eltype(Q)}([]), nhorzelem)

  for j = 1:M
    max_residual_norm = 0.0
    # Global evaluation step
    linearoperator!(krylov_basis[j + 1], krylov_basis[j], args...)

    for es = 1:nhorzelem
      stop_iter[es] == -1 || continue

      stack = (es-1) * ss + 1 : es * ss
      H = solver.H[es]
      g0 = solver.g0[es]
      Ω = Gs[es]

      for i = 1:j
        @views H[i, j] = dot(krylov_basis[j+1][stack],
                      krylov_basis[i][stack], weighted)
        @. @views krylov_basis[j + 1][stack] -= H[i, j] * krylov_basis[i][stack]
      end
      H[j + 1, j] = norm(krylov_basis[j + 1][stack], weighted)
      krylov_basis[j + 1][stack] ./= H[j + 1, j]

      # apply the previous Given rotations to the new column of H
      @views H[1:j, j:j] .= Ω * H[1:j, j:j]

      # compute a new Givens rotation to zero out H[j + 1, j]
      G, _ = givens(H, j, j + 1, j)

      # apply the new rotation to H and the rhs
      lmul!(G, H)
      lmul!(G, g0)

      # compose the new rotation with the others
      lmul!(G, Ω)

      res_local = abs(g0[j + 1])

      # Mask iteration if converged
      res_local < threshold && (stop_iter[es] = j)
      max_residual_norm = max(max_residual_norm, res_local)
    end

    if max_residual_norm < threshold
      converged = true
      break
    end
  end

  for es = 1:nhorzelem
    stack = (es-1) * ss + 1 : es * ss
    H = solver.H[es]
    g0 = solver.g0[es]
    j = stop_iter[es] == -1 ? M : stop_iter[es]

    # solve the triangular system
    y = SVector{j}(@views UpperTriangular(H[1:j, 1:j]) \ g0[1:j])

    ## compose the solution
    rv_Q = view(realview(Q), stack)
    rv_krylov_basis = [view(realview(krylov_basis[i]), stack) for i in 1:j]
    threads = 256
    blocks = div(length(rv_Q) + threads - 1, threads)
    @launch(device(Q), threads = threads, blocks = blocks,
            LS.linearcombination!(rv_Q, y, rv_krylov_basis, true))
  end

  inner = maximum(stop_iter)

  # if not converged restart
  converged || LS.initialize!(linearoperator!, Q, Qrhs, solver, args...)
  fill!(stop_iter, -1)

  (converged, inner, max_residual_norm)
end

end # module
