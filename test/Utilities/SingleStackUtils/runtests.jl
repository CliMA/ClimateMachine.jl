using MPI, Test
include(joinpath("..", "..", "testhelpers.jl"))

@testset "SingleStackUtils" begin
    runmpi(joinpath(@__DIR__, "horizontal_stats_test.jl"))
end
