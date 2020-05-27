using Test, MPI
include(joinpath("..", "..", "testhelpers.jl"))

@testset "ODE Solvers" begin
    runmpi(joinpath(@__DIR__, "ode_tests_basic.jl"))
    runmpi(joinpath(@__DIR__, "genericcb_tests.jl"))
end
