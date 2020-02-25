using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using Printf
using LinearAlgebra
using Logging
using CLIMA.MPIStateArrays

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
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
function run(mpicomm, topl, ArrayType, N, FT)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp,
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
  CLIMA.init()
  ArrayTypes = (CLIMA.array_type(),)

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  numelem = (5, 5, 1)

  polynomialorder = 4

  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1//2
  Router = 1

  expected_result = [2.0924087890777517e-04;
                     1.3897932154337201e-05;
                     8.8256018429045312e-07;
                     5.5381072850485303e-08];

  lvls = integration_testing ? length(expected_result) : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64,) #Float32)
      err = zeros(FT, lvls)
      for l = 1:lvls
        @info (ArrayType, FT, "sphere", l)
        Nhorz = 2^(l-1) * base_Nhorz
        Nvert = 2^(l-1) * base_Nvert
        Rrange = range(FT(Rinner); length=Nvert+1, stop=Router)
        topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
        err[l] = run(mpicomm, topl, ArrayType, polynomialorder, FT)

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

nothing
