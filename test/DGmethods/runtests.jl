using MPI, Test
include("../testhelpers.jl")

@testset "DGmethods" begin
  tests = [
    (1,"integral_test.jl")
    (1,"integral_test_sphere.jl")
    (1, "Euler/isentropicvortex.jl")
    (1, "Euler/isentropicvortex-imex.jl")
    (1, "Euler/isentropicvortex-multirate.jl")
    (1, "advection_diffusion/pseudo1D_advection_diffusion.jl")
    (1, "compressible_Navier_Stokes/ref_state.jl")
    (1, "horizontal_integral_test.jl")
    (2, "courant.jl")
   ]

  runmpi(tests, @__FILE__)
end
