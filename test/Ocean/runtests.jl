using MPI, Test
include("../testhelpers.jl")

@testset "Ocean" begin
  tests = [
    # (1,"Hydrostatic_Boussinesq/test_divergence_free.jl"),
    # (1,"shallow_water/GyreDriver.jl"),
   ]

  runmpi(tests, @__FILE__)
end
