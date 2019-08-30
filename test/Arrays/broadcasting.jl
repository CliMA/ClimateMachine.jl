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

@testset "MPIStateArray broadcasting" begin
  let
    localsize = (4, 6, 8)
    A = rand(Float32, localsize)
    B = rand(Float32, localsize)

    QA = MPIStateArray{Tuple{localsize[1:2]...}, Float32, ArrayType}(mpicomm, localsize[3])
    QB = similar(QA)

    QA .= A
    QB .= B

    @test Array(QA) == A
    @test Array(QB) == B

    QC = QA .+ QB
    @test typeof(QC) == typeof(QA)
    C = Array(QC)
    @test C == A .+ B

    QC = QA .+ sqrt.(QB)
    C = Array(QC)
    @test C ≈ A .+ sqrt.(B)

    QC = QA .+ sqrt.(QB) .* exp.(QA .- QB .^ 2)
    C = Array(QC)
    @test C ≈ A .+ sqrt.(B) .* exp.(A .- B .^ 2)

    # writing to an existing array instead of creating a new one
    fill!(QC, 0)
    QC .= QA .+ sqrt.(QB) .* exp.(QA .- QB .^ 2)
    C = Array(QC)
    @test C ≈ A .+ sqrt.(B) .* exp.(A .- B .^ 2)
  end

  let
    numelems = 12
    realelems = 1:7
    ghostelems = 8:12
    
    QA = MPIStateArray{Tuple{1, 1, 1}, Int, ArrayType}(mpicomm, numelems,
                                                   realelems = realelems,
                                                   ghostelems = ghostelems)
    QB  = similar(QA)

    fill!(QA, 1)
    fill!(QB, 3)

    QB .= QA .+ QB

    @test all(Array(QB)[realelems] .== 4)
    @test all(Array(QB)[ghostelems] .== 3)
  end
end
