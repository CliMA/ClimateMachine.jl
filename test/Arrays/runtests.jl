using Test
include(joinpath("..", "testhelpers.jl"))

@testset "MPIStateArrays reductions" begin
    tests = [
        (1, "basics.jl")
        (1, "broadcasting.jl")
        (1, "reductions.jl")
        (3, "reductions.jl")
        (1, "varsindex.jl")
    ]

    runmpi(tests, @__FILE__)
end
