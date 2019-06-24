using Test, MPI

using CLIMA
using CLIMA.MPIStateArrays

const mpicomm = MPI.COMM_WORLD

@testset "MPIStateArray basics" begin
  Q = MPIStateArray{Tuple{4, 6}, Float32, Array}(mpicomm, 8)

  @test eltype(Q) == Float32
  @test size(Q) == (4, 6, 8)
 
  fillval = 0.5f0
  fill!(Q, fillval)

  @test Q[1] ==  fillval
  @test Q[2, 3, 4] == fillval
  @test Q[end] == fillval
  
  @test Array(Q) == fill(fillval, 4, 6, 8)
  
  Q[2, 3, 4] = 2fillval
  @test Q[2, 3, 4] != fillval
  
  Qp = similar(Q)
  
  @test typeof(Qp) == typeof(Q)
  @test eltype(Qp) == eltype(Q)
  @test size(Qp) == size(Q)
  
  A = rand(Float32, size(Q)...)
  Q .= A
  @test Array(Q) == A
  
  Qp .= Q
  @test Array(Qp) == Array(Q)
end

@static if haspkg("CuArrays")
  using CuArrays
  @testset "CUDA MPIStateArray basics" begin
    Q = MPIStateArray{Tuple{4, 6}, Float32, CuArray}(mpicomm, 8)

    @test eltype(Q) == Float32
    @test size(Q) == (4, 6, 8)
   
    fillval = 0.5f0
    fill!(Q, fillval)

    @test Q[1] ==  fillval
    @test Q[2, 3, 4] == fillval
    @test Q[end] == fillval
    
    @test Array(Q) == fill(fillval, 4, 6, 8)
    
    Q[2, 3, 4] = 2fillval
    @test Q[2, 3, 4] != fillval
    
    Qp = similar(Q)
    
    @test typeof(Qp) == typeof(Q)
    @test eltype(Qp) == eltype(Q)
    @test size(Qp) == size(Q)
    
    A = rand(Float32, size(Q)...)
    Q .= A
    @test Array(Q) == A
    
    Qp .= Q
    @test Array(Qp) == Array(Q)
  end
end
