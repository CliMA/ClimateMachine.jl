using Test, MPI
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Geometry
using StaticArrays
using LinearAlgebra

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
    Δx_r = xmax / Ne[1] / polynomialorder
    Δy_r = ymax / Ne[2] / polynomialorder
    Δz_r = zmax / Ne[3] / polynomialorder

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
    
    # Define arbitrary unit vectors
    x̂ = [1; 0; 0];
    ŷ = [0; 1; 0];
    ẑ = [0; 0; 1];
    t̂ = cross(x̂,ẑ)

    N = polynomialorder
    Np = (N + 1)^3
    for e in 1:size(grid.vgeo, 3)
        for n in 1:size(grid.vgeo, 1)
            g = LocalGeometry{Np, N}(grid.vgeo, n, e)
            @test lengthscale(g) ≈ Savg
            @test Geometry.resolutionmetric(g) ≈ M
        end
    end
    
    Δx = 1/sqrt(x̂' * M * x̂)
    Δy = 1/sqrt(ŷ' * M * ŷ)
    Δz = 1/sqrt(ẑ' * M * ẑ)
    Δt = 1/sqrt(t̂' * M * t̂)
 
    @test Δx ≈ Δx_r
    @test Δy ≈ Δy_r
    @test Δz ≈ Δz_r
    @test Δt ≈ Δy_r
    
end
