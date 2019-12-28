using Test, MPI
using LinearAlgebra

using CLIMA
using CLIMA.MPIStateArrays

const ArrayType = CLIMA.array_type()
CLIMA.init()
const mpicomm = MPI.COMM_WORLD

mpisize = MPI.Comm_size(mpicomm)
mpirank = MPI.Comm_rank(mpicomm)

@testset "MPIStateArray reductions" begin

  localsize = (4, 6, 8)

  A = Array{Float32}(reshape(1:prod(localsize), localsize))
  globalA = vcat([A for _ in 1:mpisize]...)

  QA = MPIStateArray{Float32}(mpicomm, ArrayType, localsize...)
  QA .= A


  @test norm(QA, 1)   ≈ norm(globalA, 1)
  @test norm(QA)      ≈ norm(globalA)
  @test norm(QA, Inf) ≈ norm(globalA, Inf)

  @test norm(QA; dims=(1,3))      ≈ mapslices(norm, globalA; dims=(1,3))
  @test norm(QA, 1; dims=(1,3))   ≈ mapslices(S -> norm(S, 1), globalA, dims=(1,3))
  @test norm(QA, Inf; dims=(1,3)) ≈ mapslices(S -> norm(S, Inf), globalA, dims=(1,3))

  QAW = MPIStateArray{Float32}(mpicomm, ArrayType, localsize..., weights=ones(4,1,8))
  QAW .= A

  @test norm(QAW, 1)   ≈ norm(globalA, 1)
  @test norm(QAW)      ≈ norm(globalA)
  @test norm(QAW, Inf) ≈ norm(globalA, Inf)

  if ArrayType == Array
    # TODO: make this work with CuArrays
    @test norm(QAW; dims=(1,3))      ≈ mapslices(norm, globalA; dims=(1,3))
    @test norm(QAW, 1; dims=(1,3))   ≈ mapslices(S -> norm(S, 1), globalA, dims=(1,3))
    @test norm(QAW, Inf; dims=(1,3)) ≈ mapslices(S -> norm(S, Inf), globalA, dims=(1,3))
  end

  B = Array{Float32}(reshape(reverse(1:prod(localsize)), localsize))
  globalB = vcat([B for _ in 1:mpisize]...)

  QB = similar(QA)
  QB .= B

  @test isapprox(euclidean_distance(QA, QB), norm(globalA .- globalB))
  @test isapprox(dot(QA, QB), dot(globalA, globalB))

  C = fill(Float32(mpirank+1), localsize)
  globalC = vcat([fill(i, localsize) for i in 1:mpisize]...)
  QC = similar(QA)
  QC .= C

  @test sum(QC) == sum(globalC)
  @test Array(sum(QC;dims=(1,3))) == sum(globalC;dims=(1,3))
  @test maximum(QC) == maximum(globalC)
  @test Array(maximum(QC;dims=(1,3))) == maximum(globalC;dims=(1,3))
  @test minimum(QC) == minimum(globalC)
  @test Array(minimum(QC;dims=(1,3))) == minimum(globalC;dims=(1,3))
  end
