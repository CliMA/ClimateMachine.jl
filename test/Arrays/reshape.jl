using MPI
using Test
using ClimateMachine
using ClimateMachine.MPIStateArrays

ClimateMachine.init()
ArrayType = ClimateMachine.array_type()
mpicomm = MPI.COMM_WORLD
FT = Float32
Q = MPIStateArray{FT}(mpicomm, ArrayType, 4, 4, 4)
Qb = reshape(Q, (16, 4, 1));

Q .= 1
Qb .= 1

@testset "MPIStateArray Reshape basics" begin
    ClimateMachine.gpu_allowscalar(true)
    @test minimum(Q[:] .== 1)
    @test minimum(Qb[:] .== 1)

    @test eltype(Qb) == Float32
    @test size(Qb) == (16, 4, 1)

    fillval = 0.5f0
    fill!(Qb, fillval)

    @test Qb[1] == fillval
    @test Qb[8, 1, 1] == fillval
    @test Qb[end] == fillval

    @test Array(Qb) == fill(fillval, 16, 4, 1)

    Qb[8, 1, 1] = 2fillval
    @test Qb[8, 1, 1] != fillval
    ClimateMachine.gpu_allowscalar(false)
end
