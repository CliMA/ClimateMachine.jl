using Test, MPI

using CLIMA
using CLIMA.MPIStateArrays
using CLIMA.VariableTemplates: @vars, varsindex
using StaticArrays

CLIMA.init()
const ArrayType = CLIMA.array_type()
const mpicomm = MPI.COMM_WORLD

const V = @vars begin
    a::Float32
    b::SVector{3, Float32}
    c::SMatrix{3, 8, Float32}
    d::Float32
end

@testset "MPIStateArray varsindex" begin
    Q = MPIStateArray{Float32, V}(mpicomm, ArrayType, 4, 29, 8; commtag = 888)
    @test Q.a === view(MPIStateArrays.realview(Q), :, 1:1, :)
    @test Q.b === view(MPIStateArrays.realview(Q), :, 2:4, :)
    @test Q.c === view(MPIStateArrays.realview(Q), :, 5:28, :)
    @test Q.d === view(MPIStateArrays.realview(Q), :, 29:29, :)
    @test Q.commtag === 888

    P = similar(Q)
    @test P.a === view(MPIStateArrays.realview(P), :, 1:1, :)
    @test P.b === view(MPIStateArrays.realview(P), :, 2:4, :)
    @test P.c === view(MPIStateArrays.realview(P), :, 5:28, :)
    @test P.d === view(MPIStateArrays.realview(P), :, 29:29, :)
    @test P.commtag === 888

    A = MPIStateArray{Float32}(mpicomm, ArrayType, 4, 29, 8; commtag = 888)
    @test A.commtag === 888
    @test_throws ErrorException A.a
end
