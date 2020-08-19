using Test
using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.Abstractions
import ClimateMachine.Abstractions: DiscontinuousSpectralElementGrid
using Impero, Printf, MPI, LinearAlgebra
#import ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
Ω = Circle(0,2π) × Circle(0,2π) # Ω = S¹(0,2π) × Interval(-1,1) × Interval(-2,2), Ω = Earth()

# error messages
DiscontinuousSpectralElementGrid(Ω)
DiscontinuousSpectralElementGrid(Ω, elements = (10,10,10))
DiscontinuousSpectralElementGrid(Ω, elements = (10,10,10), polynomialorder = (3,3,3))
DiscontinuousSpectralElementGrid(Ω, elements = (10,10), polynomialorder = (3,4))
DiscontinuousSpectralElementGrid(Ω×Ω, elements = (10,10,10,10), polynomialorder = (3,3,3,3))
# functional 2D
grid = DiscontinuousSpectralElementGrid(Ω, elements = (10,10), polynomialorder = (4,4), array = ArrayType)
# functional 3D
Ω = Circle(0,2π) × Circle(0,2π) × Interval(-1,1)
grid = DiscontinuousSpectralElementGrid(Ω, elements = (2,2,2), polynomialorder = (4,4,4), array = ArrayType)
##
@testset "domain test" begin
    # Previous
    dim = 3
    FT = Float64
    Nh = 3
    Nv = 3
    # Defining grid structure
    periodicity = ntuple(j -> false, dim)
    brickrange = (
        ntuple(
            j -> range(FT(-1); length = Nh + 1,
                stop = 1),
            dim - 1,
        )...,
        range(FT(-5); length = Nv + 1, stop = 5),
    )
    topl = StackedBrickTopology(
                                mpicomm,
                                brickrange;
                                periodicity = periodicity,
                                boundary = (
                                    ntuple(j -> (1, 2), dim - 1)...,
                                    (3, 4),
                                )
    )
    N = 1
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    # Impero
    Ω = Interval(-1,1) × Interval(-1,1) × Interval(-5,5)
    igrid = DiscontinuousSpectralElementGrid(
        Ω, 
        elements = (Nh,Nh,Nv), 
        polynomialorder = (N,N,N), 
        array = ArrayType,
        )
    @test norm(grid.D .- igrid.D) ≈ 0.0
    @test norm(grid.vgeo - igrid.vgeo) ≈ 0.0
    @test norm(grid.sgeo - igrid.sgeo) ≈ 0.0

    # Multiplying by 1.0 because norm(CuArray{Int}) gives error
    # TODO: update packages to fix this error
    @test norm((grid.elemtobndy - igrid.elemtobndy)*1.0) ≈ 0.0
    # Previous
    dim = 3
    FT = Float64
    Nh = 3
    Nv = 3
    # Defining grid structure
    periodicity = (false, true, true)
    brickrange = (
        ntuple(
            j -> range(FT(-1); length = Nh + 1,
                stop = 1),
            dim - 1,
        )...,
        range(FT(-5); length = Nv + 1, stop = 5),
    )
    topl = StackedBrickTopology(
                                mpicomm,
                                brickrange;
                                periodicity = periodicity,
                                boundary = (
                                    ntuple(j -> (1, 2), dim - 1)...,
                                    (3, 4),
                                )
    )
    N = 1
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )
    # Impero
    Ω = Interval(-1,1) × Circle(-1,1) × Circle(-5,5)
    igrid = DiscontinuousSpectralElementGrid(
        Ω, 
        elements = (Nh,Nh,Nv), 
        polynomialorder = (N,N,N), 
        array = ArrayType,
        )
    @test norm(grid.D .- igrid.D) ≈ 0.0
    @test norm(grid.vgeo - igrid.vgeo) ≈ 0.0
    @test norm(grid.sgeo - igrid.sgeo) ≈ 0.0

    # Multiplying by 1.0 because norm(CuArray{Int}) gives error
    # TODO: update packages to fix this error
    @test norm((grid.elemtobndy - igrid.elemtobndy)*1.0) ≈ 0.0
end
