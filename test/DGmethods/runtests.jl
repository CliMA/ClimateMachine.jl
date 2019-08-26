using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [(1, "compressible_Navier_Stokes/ref_state.jl")
          ]

  runmpi(tests, @__FILE__)
end
