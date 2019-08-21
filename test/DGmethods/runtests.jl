using MPI, Test
include("../testhelpers.jl")

@testset "DGMethods (AtmosModel)" begin
  tests = [
           (1, "compressible_Navier_Stokes/rayleigh-benard_model.jl")
          ]

  runmpi(tests, @__FILE__)
end
