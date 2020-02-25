using MPI
using Test
using LinearAlgebra
using Random
using GPUifyLoops, StaticArrays

using CLIMA
using CLIMA.LinearSolvers
using CLIMA.ColumnwiseLUSolver: band_lu_knl!, band_forward_knl!,
                                band_back_knl!


CLIMA.init()
const ArrayType = CLIMA.array_type()
const device = ArrayType == Array ? CPU() : CUDA()

function band_to_full(B, p, q)
  _, n = size(B)

  A = similar(B, n, n) # assume square
  fill!(A, 0)
  for j = 1:n, i = max(1, j - q):min(j + p, n)
    A[i, j] = B[q + i - j + 1, j]
  end
  A
end

let
  Nq = 2
  nstate = 3
  nvertelem = 5
  nhorzelem = 4
  eband = 2

  FT = Float64
  m = n = Nq * nstate * nvertelem
  p = q = Nq * nstate * eband - 1

  Random.seed!(1234)
  AB = rand(FT, Nq, Nq, p + q + 1, n, nhorzelem)
  AB[:, :, q + 1, :, :] .+= 10 # Make A's diagonally dominate

  Random.seed!(5678)
  b = rand(FT, Nq, Nq, Nq, nstate, nvertelem, nhorzelem)
  x = similar(b)

  perm = (4, 3, 5, 1, 2, 6)
  bp = reshape(PermutedDimsArray(b, perm), n, Nq, Nq, nhorzelem)
  xp = reshape(PermutedDimsArray(x, perm), n, Nq, Nq, nhorzelem)

  threads = (Nq, Nq)
  blocks = nhorzelem
  d_F = ArrayType(AB)
  
  @launch(device, threads=threads, blocks=blocks,
          band_lu_knl!(d_F, Val(Nq), Val(Nq), Val(Nq), Val(nstate), Val(nvertelem),
                       Val(nhorzelem), Val(eband)))

  F = Array(d_F)

  for h = 1:nhorzelem, j = 1:Nq, i = 1:Nq
    B = AB[i, j, :, :, h]
    G = band_to_full(B, p, q)
    GLU = lu!(G, Val(false))

    H = band_to_full(F[i, j, :, :, h], p, q)

    @test H ≈ G

    xp[:, i, j, h] .= GLU \ bp[:, i, j, h]
  end

  b = reshape(b, Nq * Nq * Nq, nstate, nvertelem * nhorzelem)
  x = reshape(x, Nq * Nq * Nq, nstate, nvertelem * nhorzelem)

  d_x = ArrayType(b)

  @launch(device, threads=threads, blocks=blocks,
          band_forward_knl!(d_x, d_F, Val(Nq), Val(Nq), Val(nstate),
                            Val(nvertelem), Val(nhorzelem), Val(eband)))
  @launch(device, threads=threads, blocks=blocks,
          band_back_knl!(d_x, d_F, Val(Nq), Val(Nq), Val(nstate),
                         Val(nvertelem), Val(nhorzelem), Val(eband)))

  @test x ≈ Array(d_x)
end
