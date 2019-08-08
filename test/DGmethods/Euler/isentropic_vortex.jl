using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
end

include("standalone_euler_model.jl")

const halfperiod = 5
struct IsentropicVortex <: EulerProblem end
function initial_condition!(m::EulerModel, ::IsentropicVortex, state::Vars,
                            aux::Vars, (x, y, z), t)
  DFloat = eltype(state.ρ)

  γ::Float64    = γ_exact
  uinf::Float64 = 2
  vinf::Float64 = 1
  Tinf::Float64 = 1
  λ::Float64    = 5

  xs = Float64(x) - uinf * Float64(t)
  ys = Float64(y) - vinf * Float64(t)

  # make the function periodic
  xtn = floor((xs + halfperiod) / (2halfperiod))
  ytn = floor((ys + halfperiod) / (2halfperiod))
  xp = xs - xtn * 2halfperiod
  yp = ys - ytn * 2halfperiod

  rsq = xp^2 + yp^2

  u = uinf - λ * (1//2) * exp(1 - rsq) * yp / π
  v = vinf + λ * (1//2) * exp(1 - rsq) * xp / π
  w = zero(Float64)

  γm1 = γ - 1
  ρ = (Tinf - (γm1 * λ^2 * exp(2 * (1 - rsq)) / (γ * 16 * π^2)))^(1 / γm1)
  p = ρ^γ
  ρe = p / γm1 + ρ * (u^2 + v^2 + w^2) / 2

  state.ρ = DFloat(ρ)
  state.ρu⃗ = SVector{3, DFloat}(ρ * u, ρ * v, ρ * w)
  state.ρe = DFloat(ρe)
end

function run(mpicomm, ArrayType, topl, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  dg = DGModel(EulerModel(IsentropicVortex()),
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))

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

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, ))

  engf = norm(Q)
  Qe = init_ode_state(dg, param, DFloat(timeend))

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

  polynomialorder = 4
  base_num_elem = 4
  timeend = 1
  numelem = (5, 5, 1)

  expected_error = Dict()

  expected_error[Float32, 2, 1] = 5.7115197181701660f-01
  expected_error[Float32, 2, 2] = 6.9431319832801819f-02
  # CPU and GPU get slightly different numbers for this test
  @static if haspkg("CuArrays")
    expected_error[Float32, 2, 3] = 3.7826781626790762e-03
  else
    expected_error[Float32, 2, 3] = 3.7752657663077116e-03
  end

  expected_error[Float32, 3, 1] = 1.8061398267745972f+00
  expected_error[Float32, 3, 2] = 2.1956385672092438f-01
  @static if haspkg("CuArrays")
    expected_error[Float32, 3, 3] = 1.1993319727480412f-02
  else
    expected_error[Float32, 3, 3] = 1.1954923160374165f-02
  end

  expected_error[Float64, 2, 1] = 5.7115689019456495e-01
  expected_error[Float64, 2, 2] = 6.9418982796523573e-02
  expected_error[Float64, 2, 3] = 3.2927550219067014e-03

  expected_error[Float64, 3, 1] = 1.8061566743070110e+00
  expected_error[Float64, 3, 2] = 2.1952209848920567e-01
  expected_error[Float64, 3, 3] = 1.0412605646145325e-02

  @static if haspkg("CuArrays")
    ArrayType = CuArray
  else
    ArrayType = Array
  end

  # On Azure only run first level
  lvls = parse(Bool, lowercase(get(ENV,"TF_BUILD","false"))) ? 1 : 3

  @testset "$(@__FILE__)" for DFloat in (Float32, Float64)
    result = zeros(DFloat, lvls)
    for dim = 2:3
      for l = 1:lvls
        Ne = ntuple(j->2^(l-1) * numelem[j], dim)
        dt = 1e-2 / Ne[1]
        brickrange = ntuple(j->range(DFloat(-halfperiod); length=Ne[j]+1,
                                     stop=halfperiod), dim)
        topl = BrickTopology(mpicomm, brickrange,
                             periodicity = ntuple(j->true, dim))
        dt = 1e-2 / Ne[1]

        nsteps = ceil(Int64, timeend / dt)
        dt = timeend / nsteps

        @info (ArrayType, DFloat, dim)
        result[l] = run(mpicomm, ArrayType, topl, polynomialorder,
                        timeend, DFloat, dt)
        @test result[l] ≈ DFloat(expected_error[DFloat, dim, l])
      end
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

nothing
