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
using CLIMA.Ocean2D
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.PlanetParameters: grav
import CLIMA.Ocean2D: hb2d_init_aux!, hb2d_init_state!, hb2d_boundary_state!
import CLIMA.DGmethods: update_aux!, vars_state, vars_aux

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
end

struct SimpleBox2D{T} <: HB2DProblem
  Lˣ::T
  H::T
  θᴱ::T
end

# α is Filled afer the state
function hb2d_init_aux!(P::SimpleBox2D, α, geom)
  @inbounds begin
    x = geom.coord[1]
    y = geom.coord[2]

    Lˣ = P.Lˣ
    H  = P.H
    θᴱ = P.θᴱ

    # stream function
    # Ψ(x,y) = cos(π//Lˣ * (x - Lˣ//2)) * cos(π//H * (y + H/2))
    u = -π/Lˣ * cos.(π/Lˣ * (x .- Lˣ/2)) .* sin.(π/H * (y .+ H/2))
    v =  π/H  * sin.(π/Lˣ * (x .- Lˣ/2)) .* cos.(π/H * (y .+ H/2))
    w =  0.0

    α.u = @SVector [ u, v, w ]

    # α.θʳ =  θᴱ * (1 - x / Lˣ)
  end
end

function hb2d_init_state!(P::SimpleBox2D, Q, α, coords, t)
  @inbounds y = coords[2]
  @inbounds H = P.H

  Q.θ = 9 + 8y/H
end

###################
# PARAM SELECTION #
###################
DFloat = Float64
vtkpath = "vtk_box2D"

const timeend = 30 * 86400 # 4 * 365 * 86400
const tout    = 24 * 60 * 60

const N  = 4
const Ne = (10, 10)
const Lˣ = 1e6
const H  = 400

const cʰ = sqrt(grav * H)
const cᶻ = 0

const θᴱ = 25
const κʰ = 0 # 1e3
const κᶻ = 0 # 1e-3
const λʳ = 0 # 1 // 86400

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

  brickrange = (range(DFloat(0);  length=Ne[1]+1, stop=Lˣ),
                range(DFloat(-H); length=Ne[2]+1, stop=0))
  topl = StackedBrickTopology(mpicomm, brickrange;
                              periodicity = (true, true))

  dt = 1 # 240 # (L[1] / c) / Ne[1] / N^2
  @show nout = ceil(Int64, tout / dt)
  @show dt = tout / nout

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )


  problem = SimpleBox2D{DFloat}(Lˣ, H, θᴱ)

  model = HB2DModel{typeof(problem),DFloat}(problem, cʰ, cᶻ)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))

  if isdir(vtkpath)
    rm(vtkpath, recursive=true)
  end
  mkpath(vtkpath)

  step = [0]
  function do_output(step)
    outprefix = @sprintf("%s/mpirank%04d_step%04d",vtkpath,
                         MPI.Comm_rank(mpicomm), step[1])
    @info "doing VTK output" outprefix
    statenames = flattenednames(vars_state(model, eltype(Q)))
    auxnames = flattenednames(vars_aux(model, eltype(Q)))
    writevtk(outprefix, Q, dg, statenames, param.aux, auxnames)
  end
  do_output(step)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(nout)  do (init=false)
    do_output(step)
    step[1] += 1
    nothing
  end

  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
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
  # lsrk = LSRKEulerMethod(dg, Q; dt=dt, t0=0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s""" eng0 ArrayType

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo,cbvtk))
  nothing

end
