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
using CLIMA.VariableTemplates: flattenednames
using CLIMA.Ocean3D
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.PlanetParameters: grav
import CLIMA.Ocean3D: ocean_init_aux!, ocean_init_state!, NoFlux_NoSlip,
                      FreeSlip, ocean_boundary_state!
import CLIMA.DGmethods: update_aux!, vars_state, vars_aux

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
end

@inline function ocean_boundary_state!(m::HydrostaticBoussinesqModel,
                                       bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, NoFlux_NoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, FreeSlip(), x...)
  end
end

struct SimpleBox{T} <: HydrostaticBoussinesqProblem
  Lx::T
  Ly::T
  H::T
  τ0_wind::T
end

# aux is Filled afer the state
function ocean_init_aux!(P::SimpleBox, aux, geom)
  DFloat = eltype(aux)
  β::DFloat = 1e-11
  f0::DFloat = 1e-4
  @inbounds y = geom.coord[2]
  @inbounds Ly = P.Ly
  τ0_wind = P.τ0_wind

  aux.f = f0 + y * β
  aux.SST_relax = 25 * (Ly - y) / Ly
  aux.τ_wind = -τ0_wind * cos(y * 2π / Ly)
end

function ocean_init_state!(p::SimpleBox, state, aux, coords, t)
  @inbounds z = coords[3]
  @inbounds H = p.H
  state.u = @SVector [0,0]
  state.η = 0
  state.θ = 9 + 8z / H
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

  @static if haspkg("CuArrays")
    ArrayType = CuArray
  else
    ArrayType = Array
  end

  DFloat = Float64

  N = 4
  Ne = (10, 10, 4)
  L = SVector{3, DFloat}(1e6, 1e6, 400)
  timeend = 100 * 86400
  H::DFloat = L[3]
  ch::DFloat = sqrt(grav * H)
  cv::DFloat = 0
  c = @SVector [ch, ch, cv]
  brickrange = (range(DFloat(0); length=Ne[1]+1, stop=L[1]),
                range(DFloat(0); length=Ne[2]+1, stop=L[2]),
                range(DFloat(-L[3]); length=Ne[3]+1, stop=0))
  topl = StackedBrickTopology(mpicomm, brickrange;
                              periodicity = (false, false, false),
                              boundary = ((1, 1), (1, 1), (2, 2)))
  @show dt = 240 # (L[1] / c) / Ne[1] / N^2
  timeend = 4 * 365 * 86400
  tout = 24 * 60 * 60

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  problem = SimpleBox(L..., DFloat(1e-1))
  αT::DFloat = 2e-4
  νh::DFloat = 1e4
  νz::DFloat = 1e-2
  λ_relax::DFloat = 1 // 86400
  model = HydrostaticBoussinesqModel(problem, c..., αT, λ_relax, νh, νz)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))
  update_aux!(dg, model, Q, param.aux, DFloat(0), param.blparam)

  step = [0]
  vtkpath = "vtk_hydrostatic_Boussinesq_simple_box_vert_filter"
  mkpath(vtkpath)
  function do_output(step)
    outprefix = @sprintf("%s/mpirank%04d_step%04d",vtkpath,
                         MPI.Comm_rank(mpicomm), step[1])
    @info "doing VTK output" outprefix
    statenames = flattenednames(vars_state(model, eltype(Q)))
    auxnames = flattenednames(vars_aux(model, eltype(Q)))
    writevtk(outprefix, Q, dg, statenames, param.aux, auxnames)
  end
  do_output(step)
  @show nout = ceil(Int64, tout / dt)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(nout)  do (init=false)
    do_output(step)
    step[1] += 1
    nothing
  end

  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(1, mpicomm) do (s=false)
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

  lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s""" eng0 ArrayType

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo,cbvtk))
  nothing

end
