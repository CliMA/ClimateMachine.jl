using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.MPIStateArrays
using Printf
using LinearAlgebra
using Logging

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

using Test
function run(mpicomm, dim, ArrayType, Ne, DFloat)
  N = 3

  brickrange = ntuple(j->range(DFloat(-1); length=Ne[j]+1, stop=1), dim)
  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 4,
                           flux! = (x...) -> (),
                           numerical_flux! = (x...) -> ())

  filter = CutoffFilter(grid, 2)

  # Legendre Polynomials
  l0(r) = 1
  l1(r) = r
  l2(r) = (3*r^2-1)/2
  l3(r) = (5*r^3-3r)/2

  low(x, y, z)  = l0(x) * l0(y) + 4 * l1(x) * l1(y) + 5 * l1(z) +
                  6 * l1(z) * l1(x)

  high(x, y, z) = l2(x) * l3(y) + l3(x) + l2(y) + l3(z) * l1(y)

  filtered_both(x, y, z) = high(x, y, z)

  filtered_vertical(x, y, z) = (dim == 2) ? l2(x) * l3(y) + l2(y) :
                                            l3(z) * l1(y)
  filtered_horizontal(x, y, z) = (dim == 2) ? l2(x) * l3(y) + l3(x) :
                                              l2(x) * l3(y) + l3(x) + l2(y)

  for horizontal = false:true
    for vertical = false:true

      if horizontal && vertical
        filtered = filtered_both
      elseif horizontal
        filtered = filtered_horizontal
      elseif vertical
        filtered = filtered_vertical
      else
        filtered = (x...) -> zero(x[1])
      end

      Q = MPIStateArray(spacedisc) do Q, x, y, z, _...
        @inbounds begin
          Q[1] = low(x, y, z) + high(x, y, z)
          Q[2] = low(x, y, z) + high(x, y, z)
          Q[3] = low(x, y, z) + high(x, y, z)
          Q[4] = low(x, y, z) + high(x, y, z)
        end
      end
      P = MPIStateArray(spacedisc) do P, x, y, z, _...
        @inbounds begin
          P[1] = low(x, y, z) + high(x, y, z) - filtered(x, y, z)
          P[2] = low(x, y, z) + high(x, y, z)
          P[3] = low(x, y, z) + high(x, y, z) - filtered(x, y, z)
          P[4] = low(x, y, z) + high(x, y, z)
        end
      end

      DGBalanceLawDiscretizations.apply!(Q, (1,3), spacedisc, filter;
                                         horizontal=horizontal,
                                         vertical=vertical)

      @test Array(Q.Q) ≈ Array(P.Q)
    end
  end
end

let
  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
             ll == "WARN"  ? Logging.Warn  :
             ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  numelem = (1, 1, 1)
  lvls = 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64, Float32)
      for dim = 2:3
        err = zeros(DFloat, lvls)
        for l = 1:lvls
          @info (ArrayType, DFloat, dim)
          run(mpicomm, dim, ArrayType, ntuple(j->2^(l-1) * numelem[j], dim),
              DFloat)
        end
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
