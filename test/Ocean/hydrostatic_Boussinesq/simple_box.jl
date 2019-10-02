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
import CLIMA.Ocean3D: ocean_init_aux!, ocean_init_state!, Coastline,
                      OceanFloor, OceanSurface, ocean_boundary_state!
import CLIMA.DGmethods: update_aux!, vars_state, vars_aux

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
end

HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

@inline function ocean_boundary_state!(m::HBModel, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, Coastline(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloor(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurface(), x...)
  end
end

struct SimpleBox{T} <: HBProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  fₒ::T
  β::T
  θᴱ::T
end

# α is Filled afer the state
function ocean_init_aux!(P::SimpleBox, α, geom)
  DFloat = eltype(α)
  @inbounds y = geom.coord[2]

  Lʸ = P.Lʸ
  τₒ = P.τₒ
  fₒ = P.fₒ
  β  = P.β
  θᴱ = P.θᴱ

  α.τ  = -τₒ * cos(y * 2π / Lʸ)
  α.f  =  fₒ + β * y
  α.θʳ =  θᴱ * (1 - y / Lʸ)

end

function ocean_init_state!(p::SimpleBox, Q, α, coords, t)
  @inbounds z = coords[3]
  @inbounds H = p.H
  Q.u = @SVector [0,0]
  Q.η = 0
  z > -200 ? Q.θ = 10 : Q.θ = 5
end

###################
# PARAM SELECTION #
###################
DFloat = Float64
vtkpath = "vtk_hydrostatic_Boussinesq_simple_box_κ"

const timeend = 0.25 * 86400 # 4 * 365 * 86400
const tout    = 120 # 60 * 60

const Lˣ = 1e6
const Lʸ = 1e6
const H  = 400
const cʰ = sqrt(grav * H)
const cᶻ = 0

const τₒ = 1e-1
const fₒ = 1e-4
const β  = 1e-11
const θᴱ = 25

const αᵀ = 0 # 2e-4
const νʰ = 1e4
const νᶻ = 1e-2
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

  DFloat = Float64

  N = 4
  Ne = (2, 2, 4)
  L = SVector{3, DFloat}(Lˣ, Lʸ, H)
  c = @SVector [cʰ, cʰ, cᶻ]
  brickrange = (range(DFloat(0); length=Ne[1]+1, stop=L[1]),
                range(DFloat(0); length=Ne[2]+1, stop=L[2]),
                range(DFloat(-L[3]); length=Ne[3]+1, stop=0))
  topl = StackedBrickTopology(mpicomm, brickrange;
                              periodicity = (false, false, false),
                              boundary = ((1, 1), (1, 1), (2, 3)))

  dt = 120 # 240 # (L[1] / c) / Ne[1] / N^2
  @show nout = ceil(Int64, tout / dt)
  @show dt = tout / nout

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )


  problem = SimpleBox{DFloat}(L..., τₒ, fₒ, β, θᴱ)

  model = HBModel{typeof(problem),DFloat}(problem, c...,αᵀ, λʳ, νʰ, νᶻ, κʰ, κᶻ)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))
  update_aux!(dg, model, Q, param.aux, DFloat(0), param.blparam)

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

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s""" eng0 ArrayType

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo,cbvtk))
  nothing

end
