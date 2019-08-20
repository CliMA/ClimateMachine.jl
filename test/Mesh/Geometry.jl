using Test, MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using StaticArrays

MPI.Initialized() || MPI.Init()

@testset "LocalGeometry" begin
  DF = Float64
  ArrayType = Array

  xmin      = 0
  ymin      = 0
  zmin      = 0
  xmax      = 2000
  ymax      = 400
  zmax      = 2000

  Ne        = (20,2,20)

  polynomialorder = 4

  brickrange = (range(DF(xmin); length=Ne[1]+1, stop=xmax),
                range(DF(ymin); length=Ne[2]+1, stop=ymax),
                range(DF(zmin); length=Ne[3]+1, stop=zmax))
  topl = StackedBrickTopology(MPI.COMM_SELF, brickrange, periodicity = (true, true, false))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                          )

  S = ((xmax-xmin)/(polynomialorder*Ne[1]), (ymax-ymin)/(polynomialorder*Ne[2]), (zmax-zmin)/(polynomialorder*Ne[3]))
  Savg = cbrt(prod(S))
  M = SDiagonal(S.^-2)

  for e in 1:size(grid.vgeo,3)
    for n in 1:size(grid.vgeo,1)
      g = LocalGeometry(Val(polynomialorder), grid.vgeo, n, e)
      @test lengthscale(g) ≈ Savg
      @test Geometry.resolutionmetric(g) ≈ M    
    end
  end
end



