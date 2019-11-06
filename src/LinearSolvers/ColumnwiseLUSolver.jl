module ColumnwiseLUSolver

export ColumnwiseLU

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

struct ColumnwiseLU{f, AT}
  f::F
  A::AT
end

function prefactorize(op::EulerOperator{F}, solver::AbstractColumnLUSolver, Q,
                      args...) where {F <: DGModel}
  dg = op.f!

  single_column = typeof(solver) <: SingleColumnLU

  A = banded_matrix(op, dg, MPIStateArray(dg), MPIStateArray(dg),
                    single_column, args...)

  band_lu!(A, dg)

  ColumnwiseLU(dg, A)
end

function linearsolve!(clu::ColumnwiseLU{F}, ::AbstractColumnLUSolver,
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

  @launch(device, threads=(Nq, Nqj), blocks=nhorzelem,
          band_lu_knl!(A, Val(Nq), Val(Nqj), Val(nstate), Val(nvertelem),
                       Val(nhorzelem), Val(eband)))
end

function band_forward!(Q, A, dg::DGModel)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

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
          band_forward!(Q.data, A, Val(Nq), Val(Nqj), Val(nstate),
                        Val(nvertelem), Val(nhorzelem), Val(eband)))
end

function band_back!(Q, A, dg::DGModel)
  bl = dg.balancelaw
  grid = dg.grid
  topology = grid.topology
  @assert isstacked(topology)
  @assert typeof(dg.direction) <: VerticalDirection

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
          band_forward!(Q.data, A, Val(Nq), Val(Nqj), Val(nstate),
                        Val(nvertelem), Val(nhorzelem), Val(eband)))
end

end
