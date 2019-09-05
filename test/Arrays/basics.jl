using Test, MPI

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

@testset "MPIStateArray basics" begin
  Q = MPIStateArray{Tuple{4, 6}, Float32, ArrayType}(mpicomm, 8)

  @test eltype(Q) == Float32
  @test size(Q) == (4, 6, 8)
 
  fillval = 0.5f0
  fill!(Q, fillval)

  haspkg("CuArrays") && CuArrays.allowscalar(true)
  @test Q[1] ==  fillval
  @test Q[2, 3, 4] == fillval
  @test Q[end] == fillval
  
  @test Array(Q) == fill(fillval, 4, 6, 8)
  
  Q[2, 3, 4] = 2fillval
  @test Q[2, 3, 4] != fillval
  haspkg("CuArrays") && CuArrays.allowscalar(false)

  Qp = copy(Q)
  
  @test typeof(Qp) == typeof(Q)
  @test eltype(Qp) == eltype(Q)
  @test size(Qp) == size(Q)
  @test Array(Qp) == Array(Q)
  
  Qp = similar(Q)
  
  @test typeof(Qp) == typeof(Q)
  @test eltype(Qp) == eltype(Q)
  @test size(Qp) == size(Q)

  copyto!(Qp, Q)
  @test Array(Qp) == Array(Q)
end
