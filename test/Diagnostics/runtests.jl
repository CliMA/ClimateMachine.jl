using MPI, Test
include("../testhelpers.jl")

@testset "Diagnostics" begin
    runmpi(joinpath(@__DIR__, "sin_test.jl"), ntasks = 2)
end
