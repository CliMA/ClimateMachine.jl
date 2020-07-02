using MPI, Test

include("../testhelpers.jl")

@testset "Driver" begin
    runmpi(joinpath(@__DIR__, "cr_unit_tests.jl"), ntasks = 1)
    runmpi(joinpath(@__DIR__, "driver_test.jl"), ntasks = 1)
    runmpi(joinpath(@__DIR__, "mms3.jl"), ntasks = 2)
end
