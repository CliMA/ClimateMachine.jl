using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.MPIStateArrays
using CLIMA.DGBalanceLawDiscretizations
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

const _nauxstate = 7
@inline function auxiliary_state_initialization!(aux, x, y, z, dim)
  @inbounds begin
    aux[1] = x
    aux[2] = y
    aux[3] = z
    if dim == 2
      aux[4] = x*y + z*y
      aux[5] = 2*x*y + sin(x)*y^2/2 - (z-1)^2*y^3/3
    else
      aux[4] = x*z + z^2/2
      aux[5] = 2*x*z + sin(x)*y*z - (1+(z-1)^3)*y^2/3
    end
  end
end
@inline function integral_knl(val, Q, aux)
  @inbounds begin
    x, y, z = aux[1], aux[2], aux[3]
    val[1] = x + z
    val[2] = 2*x + sin(x)*y - (z-1)^2*y^2
  end
end

using Test
function run(mpicomm, dim, ArrayType, Ne, N, DFloat)

  brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=3), dim)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, dim))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 0,
                           flux! = (x...) -> (),
                           numerical_flux! = (x...) -> (),
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! = (x...) ->
                           auxiliary_state_initialization!(x..., dim))

  Q = MPIStateArray(spacedisc)

  DGBalanceLawDiscretizations.indefinite_stack_integral!(spacedisc,
                                                         integral_knl, Q,
                                                         (6, 7))

  # Wrapping in Array ensure both GPU and CPU code use same approx
  @test Array(spacedisc.auxstate.data[:, 4, :]) ≈ Array(spacedisc.auxstate.data[:, 6, :])
  @test Array(spacedisc.auxstate.data[:, 5, :]) ≈ Array(spacedisc.auxstate.data[:, 7, :])
end

let
  MPI.Initialized() || MPI.Init()

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

  numelem = (5, 5, 5)
  lvls = 1

  polynomialorder = 4

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      for dim = 2:3
        err = zeros(DFloat, lvls)
        for l = 1:lvls
          @info (ArrayType, DFloat, dim)
          run(mpicomm, dim, ArrayType, ntuple(j->2^(l-1) * numelem[j], dim),
              polynomialorder, DFloat)
        end
      end
    end
  end
end

nothing
