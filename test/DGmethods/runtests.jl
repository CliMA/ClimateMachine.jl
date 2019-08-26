using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [
          ]

  runmpi(tests, @__FILE__)
end
