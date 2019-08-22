using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [(1, "BalanceLawUtilities/runtests.jl")
          ]

  runmpi(tests, @__FILE__)
end
