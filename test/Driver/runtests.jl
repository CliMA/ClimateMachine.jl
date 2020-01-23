using MPI, Test
include("../testhelpers.jl")

@testset "Driver" begin
  tests = [
    (2,"driver_test.jl")
   ]

  runmpi(tests, @__FILE__)
end
