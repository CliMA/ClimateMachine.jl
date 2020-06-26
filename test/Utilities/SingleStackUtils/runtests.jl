using MPI, Test
include(joinpath("..", "..", "testhelpers.jl"))

@testset "SingleStackUtils" begin
    runmpi(joinpath(@__DIR__, "ssu_tests.jl"))
end
