using MPI, Test
include(joinpath("..", "..", "testhelpers.jl"))

@testset "DGmethods" begin
    runmpi(joinpath(@__DIR__, "vars_test.jl"))
    runmpi(joinpath(@__DIR__, "integral_test.jl"))
    runmpi(joinpath(@__DIR__, "integral_test_sphere.jl"))
    runmpi(joinpath(@__DIR__, "Euler/isentropicvortex.jl"))
    runmpi(joinpath(@__DIR__, "Euler/isentropicvortex_imex.jl"))
    runmpi(joinpath(@__DIR__, "Euler/isentropicvortex_multirate.jl"))
    runmpi(joinpath(
        @__DIR__,
        "advection_diffusion/pseudo1D_advection_diffusion.jl",
    ))
    runmpi(joinpath(
        @__DIR__,
        "advection_diffusion/advection_diffusion_model_1dimex_bgmres.jl",
    ))
    runmpi(joinpath(@__DIR__, "compressible_Navier_Stokes/ref_state.jl"))
    runmpi(joinpath(@__DIR__, "horizontal_integral_test.jl"))
    runmpi(joinpath(@__DIR__, "courant.jl"), ntasks = 2)
end
