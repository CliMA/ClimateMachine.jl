using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [
    (1,"integral_test.jl")
    (1,"integral_test_sphere.jl")
          ]

  runmpi(tests, @__FILE__)
end
