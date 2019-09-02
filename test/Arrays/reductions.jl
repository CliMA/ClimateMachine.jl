using Test, MPI
using LinearAlgebra

using CLIMA
using CLIMA.MPIStateArrays

MPI.Initialized() || MPI.Init()

const mpicomm = MPI.COMM_WORLD

@static if haspkg("CuArrays")
  using CuArrays
  # make sure that broadcasting is not being done by scalar indexing into CuArrays
  CuArrays.allowscalar(false)
  ArrayType = CuArray
else
  ArrayType = Array
end

mpisize = MPI.Comm_size(mpicomm)

@testset "MPIStateArray reductions" begin

  localsize = (4, 6, 8)

  A = Array{Float32}(reshape(1:prod(localsize), localsize))
  globalA = vcat([A for _ in 1:mpisize]...)

  QA = MPIStateArray{Tuple{localsize[1:2]...}, Float32, ArrayType}(mpicomm, localsize[3])
  QA .= A


  @test norm(QA, 1)   ≈ norm(globalA, 1)
  @test norm(QA)      ≈ norm(globalA)
  @test norm(QA, Inf) ≈ norm(globalA, Inf)

  @test norm(QA; dims=(1,3))      ≈ mapslices(norm, globalA; dims=(1,3))
  @test norm(QA, 1; dims=(1,3))   ≈ mapslices(S -> norm(S, 1), globalA, dims=(1,3))
  @test norm(QA, Inf; dims=(1,3)) ≈ mapslices(S -> norm(S, Inf), globalA, dims=(1,3))

  B = Array{Float32}(reshape(reverse(1:prod(localsize)), localsize))
  globalB = vcat([B for _ in 1:mpisize]...)

  QB = similar(QA)
  QB .= B

  @test isapprox(euclidean_distance(QA, QB), norm(globalA .- globalB))
  @test isapprox(dot(QA, QB), dot(globalA, globalB))
end
