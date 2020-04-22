using Test, MPI
include(joinpath("..","..","testhelpers.jl"))

@testset "ODE Solvers" begin
    tests = [(1, "ode_tests_basic.jl"), (1, "genericcb_tests.jl")]
    runmpi(tests, @__FILE__)
end
