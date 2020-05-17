using Test
include(joinpath("..", "testhelpers.jl"))

@testset "MPIStateArrays reductions" begin
    runmpi(joinpath(@__DIR__, "basics.jl"))
    runmpi(joinpath(@__DIR__, "broadcasting.jl"))
    runmpi(joinpath(@__DIR__, "reductions.jl"))
    runmpi(joinpath(@__DIR__, "reductions.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "varsindex.jl"))
end
