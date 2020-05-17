using Test, MPI
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Geometry
using StaticArrays

MPI.Initialized() || MPI.Init()

@testset "LocalGeometry" begin
    FT = Float64
    ArrayType = Array

    xmin = 0
    ymin = 0
    zmin = 0
    xmax = 2000
    ymax = 400
    zmax = 2000

    Ne = (20, 2, 20)

    polynomialorder = 4

    brickrange = (
        range(FT(xmin); length = Ne[1] + 1, stop = xmax),
        range(FT(ymin); length = Ne[2] + 1, stop = ymax),
        range(FT(zmin); length = Ne[3] + 1, stop = zmax),
    )
    topl = StackedBrickTopology(
        MPI.COMM_SELF,
        brickrange,
        periodicity = (true, true, false),
    )

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    S = (
        (xmax - xmin) / (polynomialorder * Ne[1]),
        (ymax - ymin) / (polynomialorder * Ne[2]),
        (zmax - zmin) / (polynomialorder * Ne[3]),
    )
    Savg = cbrt(prod(S))
    M = SDiagonal(S .^ -2)

    for e in 1:size(grid.vgeo, 3)
        for n in 1:size(grid.vgeo, 1)
            g = LocalGeometry(Val(polynomialorder), grid.vgeo, n, e)
            @test lengthscale(g) ≈ Savg
            @test Geometry.resolutionmetric(g) ≈ M
        end
    end
end
