using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using Printf
using LinearAlgebra
using Logging
using CLIMA.MPIStateArrays

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  const ArrayTypes = VERSION >= v"1.2-pre.25" ? (Array, CuArray) : (Array,)
else
  const ArrayTypes = (Array, )
end

@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    r = hypot(x, y, z)
    aux[1] = r
    aux[2] = x / r
    aux[3] = y / r
    aux[4] = z / r
  end
end

using Test
function run(mpicomm, topl, ArrayType, N, DFloat)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 0,
                           flux! = (x...) -> (),
                           numerical_flux! = (x...) -> (),
                           numerical_boundary_flux! = (x...) -> (),
                           auxiliary_state_length = 4,
                           auxiliary_state_initialization! = (x...) ->
                           auxiliary_state_initialization!(x...))

  exact_aux = copy(spacedisc.auxstate)

  DGBalanceLawDiscretizations.grad_auxiliary_state!(spacedisc, 1, (2,3,4))

  euclidean_distance(exact_aux, spacedisc.auxstate)
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
  @static if Base.find_package("CUDAnative") !== nothing
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  numelem = (5, 5, 1)

  polynomialorder = 4

  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1//2
  Router = 1

  expected_result = [2.6990557059226476e-04;
                     1.5939109758106752e-05;
                     1.0009810808262489e-06;
                     6.2636204797003758e-08]

  lvls = integration_testing ? length(expected_result) : 1

  for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      err = zeros(DFloat, lvls)
      for l = 1:lvls
        @info (ArrayType, DFloat, "sphere", l)
        Nhorz = 2^(l-1) * base_Nhorz
        Nvert = 2^(l-1) * base_Nvert
        Rrange = range(DFloat(Rinner); length=Nvert+1, stop=Router)
        topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
        err[l] = run(mpicomm, topl, ArrayType, polynomialorder, DFloat)
        @test expected_result[l] ≈ err[l]
      end
      if integration_testing
        @info begin
          msg = ""
          for l = 1:lvls-1
            rate = log2(err[l]) - log2(err[l+1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
