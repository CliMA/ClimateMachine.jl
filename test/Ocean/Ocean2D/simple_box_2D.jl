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
  L::T
  H::T
  τ::T
  θᴱ::T
end

# α is Filled afer the state
function hb2d_init_aux!(P::SimpleBox2D, α, geom)
  @inbounds begin
    x = geom.coord[1]
    y = geom.coord[2]

    L  = P.L
    H  = P.H
    τ  = P.τ
    θᴱ = P.θᴱ

    # stream function
    # Ψ(x,y) =  (L*H/τ) * cos.(π * (x/L .- 1/2)) .* cos.(π * (y/H .+ 1/2))
    u = -π*L/τ * cos.(π * (x/L .- 1/2)) .* sin.(π * (y/H .+ 1/2))
    v =  π*H/τ * sin.(π * (x/L .- 1/2)) .* cos.(π * (y/H .+ 1/2))


    # stream function
    # Ψ(x,y) =  (L*H/τ) * cos.(π * (x/L .- 1/2)) .* cos.(π * (y/H .+ 1/5)).^10
    # u = -π*L/τ * cos.(π * (x/L .- 1/2)) .* sin.(π * (y/H .+ 1/5)) * 10 * cos.(π * (y/H .+ 1/5)).^9
    # v =  π*H/τ * sin.(π * (x/L .- 1/2)) .* cos.(π * (y/H .+ 1/5)).^10

    # stream function
    # Ψ(x,y) =  (L*H/τ) * sin.(π * (x/L)) .* sin.(π * (y/H).^2)
    u = -π*L/τ * sin.(π * (x/L)) .* cos.(π * (y/H).^2) * 2 * (y/H)
    v =  π*H/τ * cos.(π * (x/L)) .* sin.(π * (y/H).^2)
    
    w =  0.0
    α.u = @SVector [ u, v, w ]

    # α.θʳ =  θᴱ * (1 - x / L)
  end
end

function hb2d_init_state!(P::SimpleBox2D, Q, α, coords, t)
  @inbounds x = coords[1]
  @inbounds y = coords[2]
  @inbounds H = P.H
  @inbounds L = P.L

  Q.θ = 9 + 8y/H

  σ = 1.0
  x° = 3//4 * L
  y° = -H/2
  # Q.θ = 10 * exp(-σ * ((x - x°)^2 + (y - y°)^2))
end

###################
# PARAM SELECTION #
###################
DFloat = Float64
vtkpath = "vtk_box2D_boundary"

const timeend = 86400 # 4 * 365 * 86400
const tout    = 60 * 60

const N  = 12
const Ne = (10, 10)
const L  = 1e6
const H  = 400
const τ  = 86400

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

  brickrange = (range(DFloat(0);  length=Ne[1]+1, stop=L),
                range(DFloat(-H); length=Ne[2]+1, stop=0))
  topl = StackedBrickTopology(mpicomm, brickrange;
                              periodicity = (false, false))

  dt = 60 # 240 # (L[1] / c) / Ne[1] / N^2
  @show nout = ceil(Int64, tout / dt)
  @show dt = tout / nout

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )


  problem = SimpleBox2D{DFloat}(L, H, τ, θᴱ)

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

    return nothing
  end
  
  do_output(step)
  
  cbvtk = GenericCallbacks.EveryXSimulationSteps(nout)  do (init=false)
    do_output(step)
    step[1] += 1
    return nothing
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
