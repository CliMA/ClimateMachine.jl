using Test, MPI
include(joinpath("..", "..", "testhelpers.jl"))

include("iterativesolvers.jl")

@testset "Batched Krylov solvers" begin
    runmpi(joinpath(@__DIR__, "bgmres.jl"); ntask = 2)
end

@testset "Linear Solvers Poisson" begin
    runmpi(joinpath(@__DIR__, "columnwiselu.jl"))
    runmpi(joinpath(@__DIR__, "poisson.jl"))
    runmpi(joinpath(@__DIR__, "bandedsystem.jl"))
    runmpi(joinpath(@__DIR__, "cg.jl"))
end
