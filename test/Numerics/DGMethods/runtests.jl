using MPI, Test
include(joinpath("..", "..", "testhelpers.jl"))

@testset "DGMethods" begin
    runmpi(joinpath(@__DIR__, "courant.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "grad_test.jl"))
    runmpi(joinpath(@__DIR__, "grad_test_sphere.jl"))
    runmpi(joinpath(@__DIR__, "horizontal_integral_test.jl"))
    runmpi(joinpath(@__DIR__, "integral_test.jl"))
    runmpi(joinpath(@__DIR__, "integral_test_sphere.jl"))
    runmpi(joinpath(@__DIR__, "vars_test.jl"))
end
