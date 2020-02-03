using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Filters
using CLIMA.MPIStateArrays
using CLIMA.DGBalanceLawDiscretizations
using Printf
using LinearAlgebra
using Logging

function initialcondition!(Q, x, y, z, _)
  @inbounds Q[1] = abs(x) - 0.1
end

using Test
function run(mpicomm, dim, ArrayType, Ne, N, FT)
  brickrange = ntuple(j->range(FT(-1); length=Ne[j]+1, stop=1), dim)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 1,
                           flux! = (x...) -> (),
                           numerical_flux! = (x...) -> ())

  Q = MPIStateArray(spacedisc, initialcondition!)

  initialsumQ = weightedsum(Q)
  @test minimum(Q.realdata) < 0

  Filters.apply!(Q, 1, spacedisc.grid, TMARFilter())

  sumQ = weightedsum(Q)

  @test minimum(Q.realdata) >= 0
  @test isapprox(initialsumQ, sumQ; rtol = 10eps(FT))
end

let
  CLIMA.init()
  ArrayTypes = (CLIMA.array_type(),)

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
             ll == "WARN"  ? Logging.Warn  :
             ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  numelem = (2, 2, 2)
  polynomialorder = 4

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64, Float32)
      for dim = 2:3
        @info (ArrayType, FT, dim)
        run(mpicomm, dim, ArrayType, numelem, polynomialorder, FT)
      end
    end
  end
end

nothing
