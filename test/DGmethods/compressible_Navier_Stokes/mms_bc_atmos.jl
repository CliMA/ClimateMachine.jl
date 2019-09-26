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
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

include("mms_solution_generated.jl")

using CLIMA.Atmos
import CLIMA.Atmos: MoistureModel, temperature, pressure, soundspeed, total_specific_enthalpy

"""
    MMSDryModel

Assumes the moisture components is in the dry limit.
"""
struct MMSDryModel <: MoistureModel
end

function total_specific_enthalpy(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars)
  zero(eltype(state))
end
function pressure(m::MMSDryModel, orientation::Orientation, state::Vars, aux::Vars)
  T = eltype(state)
  γ = T(7)/T(5)
  ρinv = 1 / state.ρ
  return (γ-1)*(state.ρe - ρinv/2 * sum(abs2, state.ρu))
end

function soundspeed(m::MMSDryModel, orientation::Orientation, state::Vars, aux::Vars)
  T = eltype(state)
  γ = T(7)/T(5)
  ρinv = 1 / state.ρ
  p = pressure(m, orientation, state, aux)
  sqrt(ρinv * γ * p)
end

function mms2_init_state!(state::Vars, aux::Vars, (x1,x2,x3), t)
  state.ρ = ρ_g(t, x1, x2, x3, Val(2))
  state.ρu = SVector(U_g(t, x1, x2, x3, Val(2)),
                     V_g(t, x1, x2, x3, Val(2)),
                     W_g(t, x1, x2, x3, Val(2)))
  state.ρe = E_g(t, x1, x2, x3, Val(2))
end

function mms2_source!(source::Vars, state::Vars, aux::Vars, t::Real)
  x1,x2,x3 = aux.coord
  source.ρ  = Sρ_g(t, x1, x2, x3, Val(2))
  source.ρu = SVector(SU_g(t, x1, x2, x3, Val(2)),
                      SV_g(t, x1, x2, x3, Val(2)),
                      SW_g(t, x1, x2, x3, Val(2)))
  source.ρe = SE_g(t, x1, x2, x3, Val(2))
end

function mms3_init_state!(state::Vars, aux::Vars, (x1,x2,x3), t)
  state.ρ = ρ_g(t, x1, x2, x3, Val(3))
  state.ρu = SVector(U_g(t, x1, x2, x3, Val(3)),
                     V_g(t, x1, x2, x3, Val(3)),
                     W_g(t, x1, x2, x3, Val(3)))
  state.ρe = E_g(t, x1, x2, x3, Val(3))
end

function mms3_source!(source::Vars, state::Vars, aux::Vars, t::Real)
  x1,x2,x3 = aux.coord
  source.ρ  = Sρ_g(t, x1, x2, x3, Val(3))
  source.ρu = SVector(SU_g(t, x1, x2, x3, Val(3)),
                      SV_g(t, x1, x2, x3, Val(3)),
                      SW_g(t, x1, x2, x3, Val(3)))
  source.ρe = SE_g(t, x1, x2, x3, Val(3))
end

# initial condition

function run(mpicomm, ArrayType, dim, topl, warpfun, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )

  if dim == 2
    model = AtmosModel(NoOrientation(),
                       NoReferenceState(),
                       ConstantViscosityWithDivergence(DFloat(μ_exact)),
                       MMSDryModel(),
                       NoRadiation(),
                       mms2_source!,
                       InitStateBC(),
                       mms2_init_state!)
  else
    model = AtmosModel(NoOrientation(),
                       NoReferenceState(),
                       ConstantViscosityWithDivergence(DFloat(μ_exact)),
                       MMSDryModel(),
                       NoRadiation(),
                       mms3_source!,
                       InitStateBC(),
                       mms3_init_state!)
  end

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  Q = init_ode_state(dg, DFloat(0))

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
  Qe = init_ode_state(dg, DFloat(timeend))

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

  expected_result = [1.5606226382564500e-01 5.3302790086802504e-03 2.2574728860707139e-04;
                     2.5803100360042141e-02 1.1794776908545315e-03 6.1785354745749247e-05]
lvls = integration_testing ? size(expected_result, 2) : 1

@testset "$(@__FILE__)" for ArrayType in ArrayTypes
for DFloat in (Float64,) #Float32)
  result = zeros(DFloat, lvls)
  for dim = 2:3
    for l = 1:lvls
      if dim == 2
        Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
        brickrange = (range(DFloat(0); length=Ne[1]+1, stop=1),
                      range(DFloat(0); length=Ne[2]+1, stop=1))
        topl = BrickTopology(mpicomm, brickrange,
                             periodicity = (false, false))
        dt = 1e-2 / Ne[1]
        warpfun = (x1, x2, _) -> begin
          (x1 + sin(x1*x2), x2 + sin(2*x1*x2), 0)
        end

      elseif dim == 3
        Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
        brickrange = (range(DFloat(0); length=Ne[1]+1, stop=1),
                      range(DFloat(0); length=Ne[2]+1, stop=1),
        range(DFloat(0); length=Ne[2]+1, stop=1))
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

      @info (ArrayType, DFloat, dim)
      result[l] = run(mpicomm, ArrayType, dim, topl, warpfun,
                      polynomialorder, timeend, DFloat, dt)
      @test result[l] ≈ DFloat(expected_result[dim-1, l])
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


#nothing
