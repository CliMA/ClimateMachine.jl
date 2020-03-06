using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

include("mms_solution_generated.jl")
include("mms_model.jl")


# initial condition

function run(mpicomm, ArrayType, dim, topl, warpfun, N, timeend, FT, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )
  dg = DGModel(MMSModel{dim}(),
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))


  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end

using Test
let
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  base_num_elem = 4

  expected_result = [1.6694721292986181e-01 5.4178750150416337e-03 2.3066867400713085e-04;
                     3.3672443923201158e-02 1.7603832251132654e-03 9.1108401774885506e-05]
  lvls = integration_testing ? size(expected_result, 2) : 1

  @testset "mms_bc_dgmodel" begin
    for FT in (Float64,) #Float32)
      result = zeros(FT, lvls)
      for dim = 2:3
        for l = 1:lvls
          if dim == 2
            Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
            brickrange = (range(FT(0); length=Ne[1]+1, stop=1),
                          range(FT(0); length=Ne[2]+1, stop=1))
            topl = BrickTopology(mpicomm, brickrange,
                                 periodicity = (false, false))
            dt = 1e-2 / Ne[1]
            warpfun = (x1, x2, _) -> begin
              (x1 + sin(x1*x2), x2 + sin(2*x1*x2), 0)
            end

          elseif dim == 3
            Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
            brickrange = (range(FT(0); length=Ne[1]+1, stop=1),
                          range(FT(0); length=Ne[2]+1, stop=1),
            range(FT(0); length=Ne[2]+1, stop=1))
            topl = BrickTopology(mpicomm, brickrange,
                                 periodicity = (false, false, false))
            dt = 5e-3 / Ne[1]
            warpfun = (x1, x2, x3) -> begin
              (x1 + (x1-1/2)*cos(2*π*x2*x3)/4,
               x2 + exp(sin(2π*(x1*x2+x3)))/20,
              x3 + x1/4 + x2^2/2 + sin(x1*x2*x3))
            end
          end
          timeend = 1
          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps

          @info (ArrayType, FT, dim)
          result[l] = run(mpicomm, ArrayType, dim, topl, warpfun,
                          polynomialorder, timeend, FT, dt)
          @test result[l] ≈ FT(expected_result[dim-1, l])
        end
        if integration_testing
          @info begin
            msg = ""
            for l = 1:lvls-1
              rate = log2(result[l]) - log2(result[l+1])
              msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
            end
            msg
          end
        end
      end
    end
  end
end

nothing

