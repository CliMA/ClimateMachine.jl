using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [
    (1,"integral_test.jl")
    (1,"integral_test_sphere.jl")
    (1, "compressible_Navier_Stokes/ref_state.jl")           
    (1, "compressible_Navier_Stokes/rising_bubble-model.jl")
    (1, "compressible_Navier_Stokes/density_current-model.jl")
          ]

  runmpi(tests, @__FILE__)
  if "linux" != lowercase(get(ENV,"TRAVIS_OS_NAME",""))
    moretests = [(1, "../../examples/DGmethods/ex_001_dycoms.jl")
                ]
    runmpi(moretests, @__FILE__)
  end
end
