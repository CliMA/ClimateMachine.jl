using MPI, Test
include(joinpath("..", "..", "testhelpers.jl"))

@testset "FVMethods" begin
    include("WENO_reconstruction.jl")
end
