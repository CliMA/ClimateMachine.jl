using MPI, Test

include(joinpath("..", "..", "testhelpers.jl"))

@testset "Mesh BrickMesh" begin
    runmpi(joinpath(@__DIR__, "BrickMesh.jl"))
end

@testset "Mesh Elements" begin
    runmpi(joinpath(@__DIR__, "Elements.jl"))
end

@testset "Mesh Metrics" begin
    runmpi(joinpath(@__DIR__, "Metrics.jl"))
end

@testset "Mesh Topology" begin
    runmpi(joinpath(@__DIR__, "topology.jl"))
end

@testset "Mesh GridIntegral" begin
    runmpi(joinpath(@__DIR__, "grid_integral.jl"))
end

@testset "Mesh Filter" begin
    runmpi(joinpath(@__DIR__, "filter.jl"))
end

@testset "Mesh Filter TMAR" begin
    runmpi(joinpath(@__DIR__, "filter_TMAR.jl"))
end

@testset "Mesh Geometry" begin
    runmpi(joinpath(@__DIR__, ("Geometry.jl")))
end

@testset "Mesh MPI Jobs" begin
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
