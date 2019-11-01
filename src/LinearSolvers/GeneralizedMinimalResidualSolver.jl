module GeneralizedMinimalResidualSolver

export GeneralizedMinimalResidual, VerticalGeneralizedMinimalResidual

using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays: device, realview
using ..Mesh.Grids

using LinearAlgebra
using LazyArrays
using StaticArrays
using GPUifyLoops

"""
    GeneralizedMinimalResidual(M, Q, tolerance)

This is an object for solving linear systems using an iterative Krylov method.
The constructor parameter `M` is the number of steps after which the algorithm
is restarted (if it has not converged), `Q` is a reference state used only
to allocate the solver internal state, and `tolerance` specifies the convergence
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
struct GeneralizedMinimalResidual{M, MP1, MMP1, T, AT} <: LS.AbstractIterativeLinearSolver
  krylov_basis::NTuple{MP1, AT}
  "Hessenberg matrix"
  H::MArray{Tuple{MP1, M}, T, 2, MMP1}
  "rhs of the least squares problem"
  g0::MArray{Tuple{MP1, 1}, T, 2, MP1}
  tolerance::MArray{Tuple{1}, T, 1, 1}

  function GeneralizedMinimalResidual(M, Q::AT, tolerance) where AT
    krylov_basis = ntuple(i -> similar(Q), M + 1)
    H = @MArray zeros(M + 1, M)
    g0 = @MArray zeros(M + 1)

    new{M, M + 1, M * (M + 1), eltype(Q), AT}(krylov_basis, H, g0, (tolerance,))
  end
end

const weighted = false

function LS.initialize!(linearoperator!, Q, Qrhs, solver::GeneralizedMinimalResidual)
    g0 = solver.g0
    krylov_basis = solver.krylov_basis

    @assert size(Q) == size(krylov_basis[1])

    # store the initial residual in krylov_basis[1]
    linearoperator!(krylov_basis[1], Q)
    krylov_basis[1] .*= -1
    krylov_basis[1] .+= Qrhs

    threshold = solver.tolerance[1] * norm(Qrhs, weighted)
    residual_norm = norm(krylov_basis[1], weighted)

    converged = false
    if residual_norm < threshold
      converged = true
      return converged, threshold
    end

    fill!(g0, 0)
    g0[1] = residual_norm
    krylov_basis[1] ./= residual_norm

    converged, threshold
end

function LS.doiteration!(linearoperator!, Q, Qrhs,
                         solver::GeneralizedMinimalResidual{M}, threshold) where M
 
  krylov_basis = solver.krylov_basis
  H = solver.H
  g0 = solver.g0

  converged = false
  residual_norm = typemax(eltype(Q))
  
  Ω = LinearAlgebra.Rotation{eltype(Q)}([])
  j = 1
  for outer j = 1:M

    # Arnoldi using the Modified Gram Schmidt orthonormalization
    linearoperator!(krylov_basis[j + 1], krylov_basis[j])
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
  converged || LS.initialize!(linearoperator!, Q, Qrhs, solver)
  
  (converged, j, residual_norm)
end






struct VerticalGeneralizedMinimalResidual{M, MP1, MMP1, T, AT} <: LS.AbstractIterativeLinearSolver
  nhorzelem::Int
  nvertelem::Int
  nhorznode::Int
  nvertnode::Int

  krylov_basis::NTuple{MP1, AT}
  "Hessenberg matrix"
  Hs::Matrix{MArray{Tuple{MP1, M}, T, 2, MMP1}}
  "rhs of the least squares problem"
  g0s::Matrix{MArray{Tuple{MP1, 1}, T, 2, MP1}}
  tolerance::MArray{Tuple{1}, T, 1, 1}

  function VerticalGeneralizedMinimalResidual(grid, M, Q::AT, tolerance) where AT
    topology = grid.topology
    nelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)

    N = polynomialorder(grid)
    Nq = N + 1
    nhorznode = Nq*Nq
    nvertnode = Nq

    krylov_basis = ntuple(i -> similar(Q), M + 1)
    Hs = [@MArray zeros(M + 1, M) for hn = 1:nhorznode, he = 1:nhorzelem]
    g0s = [@MArray zeros(M + 1) for hn = 1:nhorznode, he = 1:nhorzelem]

    new{M, M + 1, M * (M + 1), eltype(Q), AT}(nhorzelem, nvertelem, nhorznode, nvertnode, krylov_basis, Hs, g0s, (tolerance,))
  end
end

function LS.initialize!(linearoperator!, Q, Qrhs, solver::VerticalGeneralizedMinimalResidual)
  FT = eltype(Q)
  g0s = solver.g0s
  krylov_basis = solver.krylov_basis
  nhorzelem = solver.nhorzelem
  nvertelem = solver.nvertelem
  nhorznode = solver.nhorznode
  nvertnode = solver.nvertnode

  @assert size(Q) == size(krylov_basis[1])

  # store the initial residual in krylov_basis[1]
  linearoperator!(krylov_basis[1], Q)
  krylov_basis[1] .*= -1
  krylov_basis[1] .+= Qrhs

  threshold = zero(FT)
  residual_norm = zero(FT)
  for he = 1:nhorzelem
    re = (he-1)*nhorzelem+1 : he*nhorzelem
    for hn = 1:nhorznode
      rn = hn : nhorznode : nhorznode*nvertnode

      k = view(krylov_basis[1], rn, : , re)
      residual_norm = max(residual_norm, norm(k, weighted))
      threshold = max(threshold, solver.tolerance[1] * norm(view(Qrhs, rn, : , re), weighted))
    end
  end

  converged = false
  if residual_norm < threshold
    converged = true
    return converged, threshold
  end

  for he = 1:nhorzelem
    for hn = 1:nhorznode
      g0 = g0s[hn, he]
      fill!(g0, 0)
      g0[1] = residual_norm
    end
  end
  krylov_basis[1] ./= residual_norm

  converged, threshold
end

function LS.doiteration!(linearoperator!, Q, Qrhs,
                         solver::VerticalGeneralizedMinimalResidual{M}, threshold) where M
  FT = eltype(Q)
  krylov_basis = solver.krylov_basis
  Hs = solver.Hs
  g0s = solver.g0s

  converged = false
    
  Ωs = [LinearAlgebra.Rotation{eltype(Q)}([]) for hn in 1:nhorznode, he in 1:nhorzelem]
  j = 1
  for outer j = 1:M
    residual_norm = zero(FT)

    # Arnoldi using the Modified Gram Schmidt orthonormalization
    linearoperator!(krylov_basis[j + 1], krylov_basis[j])
    for he = 1:nhorzelem
      re = (he-1)*nhorzelem+1 : he*nhorzelem
      for hn = 1:nhorznode
        rn = hn : nhorznode : nhorznode*nvertnode

        H = Hs[hn, he]
        Ω = Ωs[hn, he]
        g0 = g0s[hn, he]
        K = map(k -> view(realview(k),rn,:,re), krylov_basis)
        for i = 1:j
          H[i, j] = dot(K[j + 1], K[i], weighted)
          @. K[j + 1] -= H[i, j] * K[i]
        end
        H[j + 1, j] = norm(K[j + 1], weighted)
        K[j + 1] ./= H[j + 1, j]
      
        # apply the previous Givens rotations to the new column of H
        @views H[1:j, j:j] .= Ω * H[1:j, j:j]

        # compute a new Givens rotation to zero out H[j + 1, j]
        G, _ = givens(H, j, j + 1, j)

        # apply the new rotation to H and the rhs
        H .= G * H
        g0 .= G * g0

        # compose the new rotation with the others
        Ω = lmul!(G, Ω)

        residual_norm = max(residual_norm, abs(g0[j + 1]))
      end
    end
    if residual_norm < threshold
      converged = true
      break
    end
  end
  for he = 1:nhorzelem
    re = (he-1)*nhorzelem+1 : he*nhorzelem
    for hn = 1:nhorznode
      rn = hn : nhorznode : nhorznode*nvertnode

      H = Hs[hn, he]
      Ω = Ωs[hn, he]
      K = map(k -> view(realview(k),rn,:,re), krylov_basis)

      # solve the triangular system
      y = SVector{j}(@views UpperTriangular(H[1:j, 1:j]) \ g0[1:j])

      ## compose the solution
      rv_Q = view(realview(Q),rn,:,re)
      threads = 256
      blocks = div(length(rv_Q) + threads - 1, threads)
      @launch(device(Q), threads = threads, blocks = blocks,
              LS.linearcombination!(rv_Q, y, K, true))
    end
  end

  # if not converged restart
  converged || LS.initialize!(linearoperator!, Q, Qrhs, solver)
  
  (converged, j, residual_norm)
end





end
