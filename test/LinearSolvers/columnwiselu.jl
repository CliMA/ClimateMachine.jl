using MPI
using Test
using LinearAlgebra
using Random
Random.seed!(1234)
using GPUifyLoops, StaticArrays

using CLIMA
using CLIMA.LinearSolvers
using CLIMA.ColumnwiseLUSolver: band_lu_knl!, band_forward_knl!,
                                band_back_knl!

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
  const device = CUDA()
else
  const ArrayType = Array
  const device = CPU()
end

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
  Nfields = 3
  Ne_vert = 5
  Ne_horz = 4
  EB = 2

  FT = Float64
  m = n = Nq * Nfields * Ne_vert
  p = q = Nq * Nfields * EB - 1

  AB = rand(FT, Nq, Nq, p + q + 1, n, Ne_horz)
  AB[:, :, q + 1, :, :] .+= 10 # Make A's diagonally dominate

  b = rand(FT, Nq, Nq, Nq, Nfields, Ne_vert, Ne_horz)
  x = similar(b)

  perm = (4, 3, 5, 1, 2, 6)
  bp = reshape(PermutedDimsArray(b, perm), n, Nq, Nq, Ne_horz)
  xp = reshape(PermutedDimsArray(x, perm), n, Nq, Nq, Ne_horz)

  threads = (Nq, Nq)
  blocks = Ne_horz
  d_F = ArrayType(AB)
  @launch(device, threads=threads, blocks=blocks,
          band_lu_knl!(d_F, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz),
                       Val(EB)))

  F = Array(d_F)

  for h = 1:Ne_horz, j = 1:Nq, i = 1:Nq
    B = AB[i, j, :, :, h]
    G = band_to_full(B, p, q)
    GLU = lu!(G, Val(false))

    H = band_to_full(F[i, j, :, :, h], p, q)

    @test H ≈ G

    xp[:, i, j, h] .= GLU \ bp[:, i, j, h]
  end

  d_x = ArrayType(b)

  @launch(device, threads=threads, blocks=blocks,
          band_forward_knl!(d_x, d_F, Val(Nq), Val(Nfields), Val(Ne_vert),
                            Val(Ne_horz), Val(EB)))
  @launch(device, threads=threads, blocks=blocks,
          band_back_knl!(d_x, d_F, Val(Nq), Val(Nfields), Val(Ne_vert),
                         Val(Ne_horz), Val(EB)))

  @test x ≈ Array(d_x)
end
