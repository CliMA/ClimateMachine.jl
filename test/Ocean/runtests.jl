using MPI, Test
include("../testhelpers.jl")

@testset "Ocean" begin
  tests = [
    (1,"HydrostaticBoussinesq/test_divergence_free.jl"),
    (1,"HydrostaticBoussinesq/test_ocean_gyre.jl")
    # (1,"shallow_water/GyreDriver.jl"),
   ]

  runmpi(tests, @__FILE__)
end
