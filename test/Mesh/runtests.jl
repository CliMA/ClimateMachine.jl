using MPI, Test

include("../testhelpers.jl")

MPI.Init()

include("topology.jl")
include("grid_integral.jl")

MPI.Finalize()

@testset "MPI Jobs" begin

  tests = [(5, "mpi_connect_1d.jl")
           (2, "mpi_connect_ell.jl")
           (3, "mpi_connect.jl")
           (3, "mpi_connect_stacked.jl")
           (2, "mpi_connect_stacked_3d.jl")
           (5, "mpi_connect_sphere.jl")]

  runmpi(tests, @__FILE__)
end
