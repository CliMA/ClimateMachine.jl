using MPI, Test

include(joinpath("..", "testhelpers.jl"))

@testset "Diagnostics" begin
    runmpi(joinpath(@__DIR__, "sin_test.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "dm_tests.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "Debug/test_statecheck.jl"), ntasks = 2)
end
