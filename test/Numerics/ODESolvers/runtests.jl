using Test, MPI
include(joinpath("..", "..", "testhelpers.jl"))

@testset "ODE Solvers" begin
    runmpi(joinpath(@__DIR__, "callbacks.jl"))
end
