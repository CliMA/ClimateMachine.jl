using MPI, Test
include("../testhelpers.jl")

@testset "Driver" begin
    tests = [(1, "driver_test.jl"), (1, "gcm_driver_test.jl")]

    runmpi(tests, @__FILE__)
end
