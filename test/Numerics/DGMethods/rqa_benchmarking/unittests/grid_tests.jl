using ClimateMachine
using MPI
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids

using Test

MPI.Init()

include("../interface/grid/domains.jl")
include("../interface/grid/grids.jl")

@testset "grid tests" begin
    """
        DiscretizedDomain(::ProductDomain, _...)
    """
    interval  = IntervalDomain(min = -1, max = 2)
    interval2 = IntervalDomain(min = 2, max = 3, periodic=true)
    product   = ProductDomain(domains = (interval, interval2))
    grid      = DiscretizedDomain(
        product,
        elements = 5,
        polynomial_order = (3,2),
        overintegration_order = (1,2),
        FT = Float64,
        mpicomm = MPI.COMM_WORLD,
        array = ClimateMachine.array_type(),
        topology = StackedBrickTopology,
        brick_builder = uniform_brick_builder,
    )
    @test grid.domain == product
    @test grid.resolution == (elements = 5, polynomial_order = (3,2), overintegration_order = (1,2))

    """
        DiscretizedDomain(::SphericalShell, _...)
    """
    sphere = SphericalShell(radius = 1, height = 2)
    grid  = DiscretizedDomain(
        sphere;
        elements = (vertical = 5, horizontal = 4),
        polynomial_order = (4, 3),
        overintegration_order = (2, 2),
        FT = Float64,
        mpicomm = MPI.COMM_WORLD,
        array = ClimateMachine.array_type()
    )
    @test grid.domain == sphere
    @test grid.resolution == (elements = (vertical = 5, horizontal = 4), polynomial_order = (4, 3), overintegration_order = (2, 2))
    @test polynomialorders(grid.numerical) == (6, 6, 5)

    """
    DiscontinuousSpectralElementGrid(domain::ProductDomain; _...)
    """
    grid = DiscontinuousSpectralElementGrid(
        product, 
        elements = (5,5),
        polynomialorder = (3,2),
    )
    @test polynomialorders(grid) == (3,2)

    """
    DiscontinuousSpectralElementGrid(domain::SphericalShell; _...)
    """
    grid = DiscontinuousSpectralElementGrid(
        sphere, 
        elements = (horizontal = 5, vertical = 3),
        polynomialorder = (horizontal = 4, vertical = 2),
    )
    @test polynomialorders(grid) == (4, 4, 2)
end