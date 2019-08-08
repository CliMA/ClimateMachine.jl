using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [
           (3, "Euler/isentropic_vortex.jl"),
          ]

  runmpi(tests, @__FILE__)
end
