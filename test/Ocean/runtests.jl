using MPI, Test

include("../testhelpers.jl")

@testset "Ocean" begin
    runmpi(joinpath(@__DIR__, "HydrostaticBoussinesq/test_ocean_gyre_short.jl"))
    runmpi(joinpath(@__DIR__, "SplitExplicit/test_spindown_short.jl"))
    include(joinpath("OceanProblems", "test_initial_value_problem.jl"))
    include(joinpath("Domains", "test_rectangular_domain.jl"))
    include(joinpath("Fields", "test_rectangular_element.jl"))
    include(joinpath("HydrostaticBoussinesqModel", "test_hydrostatic_boussinesq_model.jl"))
end
