module ColumnwiseLUSolver

export ManyColumnLU, SingleColumnLU

using ..Mesh.Grids
using ..Mesh.Topologies
using ..DGmethods
using ..DGmethods: BalanceLaw, DGModel, num_state, num_diffusive
using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays
using LinearAlgebra
using GPUifyLoops

include("ColumnwiseLUSolver_kernels.jl")

abstract type AbstractColumnLUSolver <: AbstractLinearSolver  end

"""
    ManyColumnLU()

This solver is used for systems that are block diagonal where each block is
associated with a column of the mesh.  The systems are solved using a
non-pivoted LU factorization.
"""
struct ManyColumnLU <: AbstractColumnLUSolver  end

"""
    SingleColumnLU()

This solver is used for systems that are block diagonal where each block is
associated with a column of the mesh.  Moreover, each block is assumed to be
the same.  The systems are solved using a non-pivoted LU factorization.
"""
struct SingleColumnLU <: AbstractColumnLUSolver  end

struct ColumnwiseLU{F, AT}
  f::F
  A::AT
end

function LS.prefactorize(op, solver::AbstractColumnLUSolver, Q, args...)
  dg = op.f!

  # TODO: can we get away with just passing the grid?
  A = banded_matrix(op, dg, similar(Q), similar(Q), args...;
                    single_column = typeof(solver) <: SingleColumnLU)

  band_lu!(A, dg)

  ColumnwiseLU(dg, A)
end

function LS.linearsolve!(clu::ColumnwiseLU{F}, ::AbstractColumnLUSolver,
                         Q, Qrhs, args...) where {F <: DGModel}
  dg = clu.f
  A = clu.A
  Q .= Qrhs

  band_forward!(Q, A, dg)
  band_back!(Q, A, dg)
end

"""
    band_lu!(A, dg::DGModel)

"""
function band_lu!(A, dg::DGModel)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(A)
  device = typeof(A) <: Array ? CPU() : CUDA()

  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqj = dimensionality(grid) == 2 ? 1 : Nq

  eband = num_diffusive(bl, FT) == 0 ? 1 : 2

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  threads = (Nq, Nqj)
  blocks = nhorzelem

  if ndims(A) == 2
    # single column case
    #
    # TODO Would it be faster to copy the matrix to the host and factorize it
    # there?
    threads = (1, 1)
    blocks = 1
    A = reshape(A, 1, 1, size(A)..., 1)
  end

  @launch(device, threads=threads, blocks=blocks,
          band_lu_knl!(A, Val(Nq), Val(threads[1]), Val(threads[2]),
                       Val(nstate), Val(nvertelem), Val(blocks),
                       Val(eband)))
end

function band_forward!(Q, A, dg::DGModel)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(A)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqj = dimensionality(grid) == 2 ? 1 : Nq

  eband = num_diffusive(bl, FT) == 0 ? 1 : 2

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  @launch(device, threads=(Nq, Nqj), blocks=nhorzelem,
          band_forward_knl!(Q.data, A, Val(Nq), Val(Nqj), Val(nstate),
                            Val(nvertelem), Val(nhorzelem), Val(eband)))
end

function band_back!(Q, A, dg::DGModel)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(A)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqj = dimensionality(grid) == 2 ? 1 : Nq

  eband = num_diffusive(bl, FT) == 0 ? 1 : 2

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  @launch(device, threads=(Nq, Nqj), blocks=nhorzelem,
          band_back_knl!(Q.data, A, Val(Nq), Val(Nqj), Val(nstate),
                         Val(nvertelem), Val(nhorzelem), Val(eband)))
end


"""
    banded_matrix(dg::DGModel, [Q::MPIStateArray, dQ::MPIStateArray,
                  single_column=false])

Forms the banded matrices for each the column operator defined by the `DGModel`
dg.  If `single_column=false` then a banded matrix is stored for each column and
if `single_column=true` only the banded matrix associated with the first column
of the first element is stored. The bandwidth of the DG column banded matrix is
`p = q = (polynomialorder + 1) * nstate * nvertelem - 1` with `p` and `q` being
the upper and lower bandwidths.

The banded matrices are stored in the LAPACK band storage format
<https://www.netlib.org/lapack/lug/node124.html>.

The banded matrices are returned as an arrays where the array type matches that
of `Q`. If `single_column=false` then the returned array has 5 dimensions, which
are:
- first horizontal column index
- second horizontal column index
- band index (-q:p)
- vertical DOF index with state `s`, vertical DOF index `k`, and vertical
  element `ev` mapping to `s + nstate * (k - 1) + nstate * nvertelem * (ev - 1)`
- horizontal element index

If the `single_column=true` then the returned array has 2 dimensions which are
the band index and the vertical DOF index.
"""
function banded_matrix(dg::DGModel, Q::MPIStateArray = MPIStateArray(dg),
                       dQ::MPIStateArray = MPIStateArray(dg);
                       single_column = false)
  banded_matrix((dQ, Q) -> dg(dQ, Q, nothing, 0; increment=false),
                dg, Q, dQ; single_column = single_column)
end

"""
    banded_matrix(f!::Function, dg::DGModel,
                  Q::MPIStateArray = MPIStateArray(dg),
                  dQ::MPIStateArray = MPIStateArray(dg), args...;
                  single_column = false, args...)

Forms the banded matrices for each the column operator defined by the linear
operator `f!` which is assumed to have the same banded structure as the
`DGModel` dg.  If `single_column=false` then a banded matrix is stored for each
column and if `single_column=true` only the banded matrix associated with the
first column of the first element is stored. The bandwidth of the DG column
banded matrix is `p = q = (polynomialorder + 1) * nstate * nvertelem - 1` with
`p` and `q` being the upper and lower bandwidths.

The banded matrices are stored in the LAPACK band storage format
<https://www.netlib.org/lapack/lug/node124.html>.

The banded matrices are returned as an arrays where the array type matches that
of `Q`. If `single_column=false` then the returned array has 5 dimensions, which
are:
- first horizontal column index
- second horizontal column index
- band index (-q:p)
- vertical DOF index with state `s`, vertical DOF index `k`, and vertical
  element `ev` mapping to `s + nstate * (k - 1) + nstate * nvertelem * (ev - 1)`
- horizontal element index

If the `single_column=true` then the returned array has 2 dimensions which are
the band index and the vertical DOF index.

Here `args` are passed to `f!`.
"""
function banded_matrix(f!, dg::DGModel,
                       Q::MPIStateArray = MPIStateArray(dg),
                       dQ::MPIStateArray = MPIStateArray(dg),
                       args...; single_column = false)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(Q.data)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1

  # p is lower bandwidth
  # q is upper bandwidth
  eband = num_diffusive(bl, FT) == 0 ? 1 : 2
  p = q = nstate * Nq * eband - 1

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  dim = dimensionality(grid)

  Nqj = dim == 2 ? 1 : Nq

  # first horizontal DOF index
  # second horizontal DOF index
  # band index -q:p
  # vertical DOF index
  # horizontal element index
  A = if single_column
    similar(Q.data, p + q + 1, Nq * nstate * nvertelem)
  else
    similar(Q.data, Nq, Nqj, p + q + 1, Nq * nstate * nvertelem,
            nhorzelem)
  end
  fill!(A, zero(FT))

  # loop through all DOFs in a column and compute the matrix column
  for ev = 1:nvertelem
    for s = 1:nstate
      for k = 1:Nq
        # Set a single 1 per column and rest 0
        @launch(device, threads=(Nq, Nqj, Nq), blocks=(nvertelem, nhorzelem),
                knl_set_banded_data!(bl, Val(dim), Val(N), Val(nvertelem),
                                     Q.data, k, s, ev, 1:nhorzelem,
                                     1:nvertelem))

        # Get the matrix column
        f!(dQ, Q, args...)

        # Store the banded matrix
        @launch(device, threads=(Nq, Nqj, Nq),
                blocks=(2 * eband + 1, nhorzelem),
                knl_set_banded_matrix!(bl, Val(dim), Val(N), Val(nvertelem),
                                       Val(p), Val(q), Val(eband+1),
                                       A, dQ.data, k, s, ev, 1:nhorzelem,
                                       -eband:eband))
      end
    end
  end
  A
end


"""
    banded_matrix_vector_product!(dg::DGModel, A, dQ::MPIStateArray,
                                  Q::MPIStateArray)

Compute a matrix vector product `dQ = A * Q` where `A` is assumed to be a matrix
created using the `banded_matrix` function.

This function is primarily for testing purposes.
"""
function banded_matrix_vector_product!(dg::DGModel, A, dQ::MPIStateArray,
                                       Q::MPIStateArray)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

  FT = eltype(Q.data)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  eband = num_diffusive(bl, FT) == 0 ? 1 : 2
  nstate = num_state(bl, FT)
  N = polynomialorder(grid)
  Nq = N + 1
  p = q = nstate * Nq * eband - 1

  nrealelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  dim = dimensionality(grid)

  Nqj = dim == 2 ? 1 : Nq

  @launch(device, threads=(Nq, Nqj, Nq),
          blocks=(nvertelem, nhorzelem),
          knl_banded_matrix_vector_product!(bl, Val(dim), Val(N),
                                            Val(nvertelem), Val(p), Val(q),
                                            dQ.data, A, Q.data, 1:nhorzelem,
                                            1:nvertelem))
end

end
