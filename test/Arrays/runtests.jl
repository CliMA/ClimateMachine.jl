using Test, MPI
include("../testhelpers.jl")


MPI.Initialized() || MPI.Init()

include("basics.jl")
include("broadcasting.jl")

MPI.Finalize()

@testset "MPIStateArrays reductions" begin

  tests = [(1, "reductions.jl"),
           (3, "reductions.jl")
          ]

  runmpi(tests, @__FILE__)
end
