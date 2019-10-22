@testset "Filter tests" begin
  using Test
  using CLIMA
  import GaussQuadrature
  using MPI
  using LinearAlgebra

  let
    # Values computed with:
    #   https://github.com/tcew/nodal-dg/blob/master/Codes1.1/Codes1D/Filter1D.m
    W = [0x3fe98f3cd0d725e8   0x3fddfd863c6c9a44   0xbfe111110d0fd334   0x3fddbe357bce0b5c   0xbfc970267f929618
         0x3fb608a150f6f927   0x3fe99528b1a1cd8d   0x3fcd41d41f8bae45   0xbfc987d5fabab8d5   0x3fb5da1cd858af87
         0xbfb333332eb1cd92   0x3fc666666826f178   0x3fe999999798faaa   0x3fc666666826f176   0xbfb333332eb1cd94
         0x3fb5da1cd858af84   0xbfc987d5fabab8d4   0x3fcd41d41f8bae46   0x3fe99528b1a1cd8e   0x3fb608a150f6f924
         0xbfc970267f929618   0x3fddbe357bce0b5c   0xbfe111110d0fd333   0x3fddfd863c6c9a44   0x3fe98f3cd0d725e8]
    W = reinterpret.(Float64, W)

    N = size(W, 1) - 1

    topology = CLIMA.Mesh.Topologies.BrickTopology(MPI.COMM_SELF, -1.0:2.0:1.0);

    grid = CLIMA.Mesh.Grids.DiscontinuousSpectralElementGrid(topology;
                                                        polynomialorder = N,
                                                        FloatType = Float64,
                                                        DeviceArray = Array)

    filter = CLIMA.Mesh.Filters.ExponentialFilter(grid, 0, 32)
    @test filter.filter ≈ W
  end

  let
    # Values computed with:
    #   https://github.com/tcew/nodal-dg/blob/master/Codes1.1/Codes1D/Filter1D.m
    W = [0x3fd822e5f54ecb62   0x3fedd204a0f08ef8   0xbfc7d3aa58fd6968   0xbfbf74682ac4d276
         0x3fc7db36e726d8c1   0x3fe59d16feee478b   0x3fc6745bfbb91e20   0xbfa30fbb7a645448
         0xbfa30fbb7a645455   0x3fc6745bfbb91e26   0x3fe59d16feee478a   0x3fc7db36e726d8c4
         0xbfbf74682ac4d280   0xbfc7d3aa58fd6962   0x3fedd204a0f08ef7   0x3fd822e5f54ecb62]
    W = reinterpret.(Float64, W)

    N = size(W, 1) - 1

    topology = CLIMA.Mesh.Topologies.BrickTopology(MPI.COMM_SELF, -1.0:2.0:1.0);
    grid = CLIMA.Mesh.Grids.DiscontinuousSpectralElementGrid(topology;
                                                        polynomialorder = N,
                                                        FloatType = Float64,
                                                        DeviceArray = Array)

    filter = CLIMA.Mesh.Filters.ExponentialFilter(grid, 1, 4)
    @test filter.filter ≈ W
  end

  let
    T = Float64
    N = 5
    Nc = 4

    topology = CLIMA.Mesh.Topologies.BrickTopology(MPI.COMM_SELF, -1.0:2.0:1.0);
    grid = CLIMA.Mesh.Grids.DiscontinuousSpectralElementGrid(topology;
                                                        polynomialorder = N,
                                                        FloatType = T,
                                                        DeviceArray = Array)

    ξ = CLIMA.Mesh.Grids.referencepoints(grid)
    a, b = GaussQuadrature.legendre_coefs(T, N)
    V = GaussQuadrature.orthonormal_poly(ξ, a, b)

    Σ = ones(T, N+1)
    Σ[(Nc:N).+1] .= 0

    W = V*Diagonal(Σ)/V

    filter = CLIMA.Mesh.Filters.CutoffFilter(grid, Nc)
    @test filter.filter ≈ W
  end
end
