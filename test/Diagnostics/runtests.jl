using MPI, Test
include("../testhelpers.jl")

@testset "Diagnostics" begin
  tests = [
    (2,"dycoms_sin_test.jl")
   ]

  runmpi(tests, @__FILE__)
end
