using Test, MPI

using CLIMA
using CLIMA.MPIStateArrays

CLIMA.init()
const ArrayType = CLIMA.array_type()
const mpicomm = MPI.COMM_WORLD

@testset "MPIStateArray broadcasting" begin
  let
    localsize = (4, 6, 8)
    A = rand(Float32, localsize)
    B = rand(Float32, localsize)

    QA = MPIStateArray{Float32}(mpicomm, ArrayType, localsize...)
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
    
    QA = MPIStateArray{Int}(mpicomm, ArrayType, 1, 1, numelems,
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
