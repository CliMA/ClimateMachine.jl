using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [(1, "compressible_Navier_Stokes/hydrostatic_state.jl")
          ]

  runmpi(tests, @__FILE__)
end
