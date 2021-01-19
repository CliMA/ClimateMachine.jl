using MPI, Test
include(joinpath("..", "..", "testhelpers.jl"))

@testset "FVMethods" begin
    include("test_WENO_reconstruction.jl")
end
