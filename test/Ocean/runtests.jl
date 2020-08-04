using MPI, Test
include("../testhelpers.jl")

@testset "Ocean" begin
    runmpi(joinpath(@__DIR__, "HydrostaticBoussinesq/test_ocean_gyre_short.jl"))
    runmpi(joinpath(@__DIR__, "SplitExplicit/test_spindown_short.jl"))
end
