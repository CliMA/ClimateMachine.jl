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
    runmpi(joinpath(@__DIR__, "mpi_centroid.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_connect_ell.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "interpolation.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_connect.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_connect_stacked.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_connect_stacked_3d.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "mpi_getpartition.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_partition.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_sortcolumns.jl"), ntasks = 1)
end
