using Test, MPI
include(joinpath("..","..","testhelpers.jl"))

include("iterativesolvers.jl")

@testset "Linear Solvers Poisson" begin
    tests = [
        (1, "columnwiselu.jl"),
        (1, "poisson.jl"),
        (1, "bandedsystem.jl"),
        (1, "cg.jl"),
    ]
    runmpi(tests, @__FILE__)
end
