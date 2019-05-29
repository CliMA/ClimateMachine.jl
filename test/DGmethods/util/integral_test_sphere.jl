using MPI
using CLIMA
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

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

const _nauxstate = 4
const _a_r, _a_θ, _a_ϕ, _a_f = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z, Rinner)
  @inbounds begin
    r = hypot(x, y, z)
    θ = atan(y , x)
    ϕ = asin(z / r)
    aux[_a_r] = r
    aux[_a_θ] = θ
    aux[_a_ϕ] = ϕ

    # Exact integral
    a = 1 + sin(θ)^2 + sin(ϕ)^2
    aux[_a_f] = exp(-a * r^2) - exp(-a * Rinner^2)
  end
end

@inline function integral_knl(val, Q, aux)
  @inbounds begin
    r, θ, ϕ = aux[_a_r], aux[_a_θ], aux[_a_ϕ]
    a = 1 + sin(θ)^2 + sin(ϕ)^2
    val[1] = -2r * a * exp(-a * r^2)
  end
end

using Test
function run(mpicomm, topl, ArrayType, N, DFloat, Rinner)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp,
                                         )

  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 0,
                           flux! = (x...) -> (),
                           numerical_flux! = (x...) -> (),
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! = (x...) ->
                           auxiliary_state_initialization!(x..., Rinner),
                           numerical_boundary_flux! = (x...) -> ())

  Q = MPIStateArray(spacedisc)
  exact_aux = copy(spacedisc.auxstate)

  DGBalanceLawDiscretizations.indefinite_stack_integral!(spacedisc,
                                                         integral_knl, Q,
                                                         _nauxstate)

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
  @static if haspkg("CUDAnative")
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  polynomialorder = 4

  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1//2
  Router = 1

  polynomialorder = 4

  expected_result = [4.4043147275367782e-07;
                     6.8386492128632267e-09;
                     1.0679315671964862e-10;
                     1.6687191893223590e-12]


  lvls = integration_testing ? length(expected_result) : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      err = zeros(DFloat, lvls)
      for l = 1:lvls
        @info (ArrayType, DFloat, "sphere", l)
        Nhorz = 2^(l-1) * base_Nhorz
        Nvert = 2^(l-1) * base_Nvert
        Rrange = range(DFloat(Rinner); length=Nvert+1, stop=Router)
        topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
        err[l] = run(mpicomm, topl, ArrayType, polynomialorder, DFloat,
                     DFloat(Rinner))
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

