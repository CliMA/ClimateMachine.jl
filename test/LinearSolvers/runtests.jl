using Test, MPI
include("../testhelpers.jl")

include("smallsystem.jl")
include("columnwiselu.jl")

@testset "Linear Solvers Poisson" begin
  tests = [(1, "poisson.jl")]
  runmpi(tests, @__FILE__)
end
