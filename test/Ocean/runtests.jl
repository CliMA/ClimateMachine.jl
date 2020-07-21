using MPI, Test
include("../testhelpers.jl")

@testset "Ocean" begin
    runmpi(joinpath(@__DIR__, "SplitExplicit/test_vertical_integral_model.jl"))
    runmpi(joinpath(@__DIR__, "ShallowWater/2D_hydrostatic_spindown.jl"))
    #! format: off
    runmpi(joinpath(@__DIR__,"HydrostaticBoussinesq/3D_hydrostatic_spindown.jl"))
    #! format: on
    runmpi(joinpath(@__DIR__, "HydrostaticBoussinesq/test_divergence_free.jl"))
    runmpi(joinpath(@__DIR__, "HydrostaticBoussinesq/test_ocean_gyre.jl"))
    runmpi(joinpath(@__DIR__, "SplitExplicit/hydrostatic_spindown.jl"))
end
