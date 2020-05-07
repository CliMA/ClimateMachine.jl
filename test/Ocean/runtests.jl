using MPI, Test
include("../testhelpers.jl")

@testset "Ocean" begin
    runmpi(joinpath(@__DIR__, "HydrostaticBoussinesq/test_divergence_free.jl"))
    runmpi(joinpath(@__DIR__, "HydrostaticBoussinesq/test_ocean_gyre.jl"))
    # runmpi(joinpath(@__DIR__,"shallow_water/GyreDriver.jl"))
end
