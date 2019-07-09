using MPI, Test

include("../testhelpers.jl")

MPI.Init()

include("BrickMesh.jl")
include("Elements.jl")
include("Metrics.jl")

include("topology.jl")
include("grid_integral.jl")
include("filter.jl")

MPI.Finalize()

@testset "MPI Jobs" begin

  tests = [(3, "mpi_centroid.jl")
           (5, "mpi_connect_1d.jl")
           (2, "mpi_connect_ell.jl")
           (3, "mpi_connect.jl")
           (3, "mpi_connect_stacked.jl")
           (2, "mpi_connect_stacked_3d.jl")
           (5, "mpi_connect_sphere.jl")
           (3, "mpi_getpartition.jl")
           (5, "mpi_getpartition.jl")
           (3, "mpi_partition.jl")
           (1, "mpi_sortcolumns.jl")
           (4, "mpi_sortcolumns.jl")]

  runmpi(tests, @__FILE__)
end
