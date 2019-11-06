module ColumnwiseLUSolver

export ManyColumnLU, SingleColumnLU

using ..DGmethods
using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays: device, realview
using LinearAlgebra
using GPUifyLoops

include("ColumnwiseLUSolver_kernels.jl")

abstract type AbstractColumnLUSolver <: AbstractLinearSolver  end

struct ManyColumnLU <: AbstractColumnLUSolver  end
struct SingleColumnLU <: AbstractColumnLUSolver  end

struct ColumnwiseLU{F, AT}
  f::F
  A::AT
end

function LS.prefactorize(op, solver::AbstractColumnLUSolver, Q, args...)
  dg = op.f!

  A = banded_matrix(op, dg, similar(dg), similar(dg), args...;
                    single_column=typeof(solver) <: SingleColumnLU)

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

end
