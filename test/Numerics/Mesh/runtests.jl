using MPI, Test

include(joinpath("..", "..", "testhelpers.jl"))

include("BrickMesh.jl")
include("Elements.jl")
include("Metrics.jl")

include("topology.jl")
include("grid_integral.jl")
include("filter.jl")
include("Geometry.jl")


# runmpi won't work if we do not finalize
# This is not so nice since other tests that are run direction and call MPI.Init
# will fail if we do finalize here (since runmpi won't work in an initialized
# state)
MPI.Initialized() && MPI.Finalize()

# The MPI library doesn't actually call the C library's `MPI_Finalize()` until
# all of the MPI object have been finalized. So we run the garbage collector to
# make sure MPI is actually finalized.
Base.GC.gc()

@testset "MPI Jobs" begin

    tests = [
        (3, "mpi_centroid.jl")
        (2, "mpi_connect_ell.jl")
        (3, "interpolation.jl")
        (3, "mpi_connect.jl")
        (3, "mpi_connect_stacked.jl")
        (2, "mpi_connect_stacked_3d.jl")
        (3, "mpi_getpartition.jl")
        (3, "mpi_partition.jl")
        (1, "mpi_sortcolumns.jl")
    ]

    runmpi(tests, @__FILE__)
end
