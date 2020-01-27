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
using CLIMA.HydrostaticBoussinesq
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.PlanetParameters: grav
import CLIMA.HydrostaticBoussinesq: ocean_init_aux!, ocean_init_state!,
                                    ocean_boundary_state!,
                                    CoastlineFreeSlip, CoastlineNoSlip,
                                    OceanFloorFreeSlip, OceanFloorNoSlip,
                                    OceanSurfaceNoStressNoForcing,
                                    OceanSurfaceStressNoForcing,
                                    OceanSurfaceNoStressForcing,
                                    OceanSurfaceStressForcing
import CLIMA.DGmethods: update_aux!, update_aux_diffusive!,
                        vars_state, vars_aux
using GPUifyLoops

const ArrayType = CLIMA.array_type()

HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

@inline function ocean_boundary_state!(m::HBModel, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloorNoSlip(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
  end
end

struct SimpleBox{T} <: HBProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  fₒ::T
  β::T
  λʳ::T
  θᴱ::T
end

# A is Filled afer the state
function ocean_init_aux!(m::HBModel, P::SimpleBox, A, geom)
  FT = eltype(A)
  @inbounds y = geom.coord[2]

  Lʸ = P.Lʸ
  τₒ = P.τₒ
  fₒ = P.fₒ
  β  = P.β
  θᴱ = P.θᴱ

  A.τ  = -τₒ * cos(y * π / Lʸ)
  A.f  =  fₒ + β * y
  A.θʳ =  θᴱ * (1 - y / Lʸ)

  κʰ = m.κʰ
  κᶻ = m.κᶻ

  # A.κ = @SMatrix [ κʰ -0 -0; -0 κʰ -0; -0 -0 κᶻ]
  A.κᶻ = κᶻ

end

function ocean_init_state!(P::SimpleBox, Q, A, coords, t)
  @inbounds z = coords[3]
  @inbounds H = P.H

  Q.u = @SVector [0,0]
  Q.η = 0
  Q.θ = 9 + 8z/H
end

###################
# PARAM SELECTION #
###################
FT = Float64
vtkpath = "vtk_testing_update_aux"

const timeend = 3 * 30 * 86400   # s
const tout    = 6 * 24 * 60 * 60 # s

const N  = 4
const Nˣ = 20
const Nʸ = 20
const Nᶻ = 20
const Lˣ = 4e6  # m
const Lʸ = 4e6  # m
const H  = 1000 # m

xrange = range(FT(0);  length=Nˣ+1, stop=Lˣ)
yrange = range(FT(0);  length=Nʸ+1, stop=Lʸ)
zrange = range(FT(-H); length=Nᶻ+1, stop=0)

# xrange = [Lˣ/2 * (1 - cos(x)) for x in range(FT(0); length=Nˣ+1, stop=π)]
# yrange = [Lʸ/2 * (1 - cos(y)) for y in range(FT(0); length=Nʸ+1, stop=π)]
# zrange = -H * [1, 0.95, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0]
# zrange = -H * [1, 0.9875, 0.975, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0375, 0.025, 0.0125, 0]

const cʰ = sqrt(grav * H)
const cᶻ = 0

const τₒ = 1e-1  # (m/s)^2
const fₒ = 1e-4  # Hz
const β  = 1e-11 # Hz / m
const θᴱ = 25    # K

const αᵀ = 2e-4  # (m/s)^2 / K
const νʰ = 5e3   # m^2 / s
const νᶻ = 5e-3  # m^2 / s
const κʰ = 1e3   # m^2 / s
const κᶻ = 1e-4  # m^2 / s
const λʳ = 4 // 86400 # m / s
let
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
             ll == "WARN"  ? Logging.Warn  :
             ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  brickrange = (xrange, yrange, zrange)
  topl = StackedBrickTopology(mpicomm, brickrange;
                              periodicity = (false, false, false),
                              boundary = ((1, 1), (1, 1), (2, 3)))

  minΔx = Lˣ / Nˣ / (N + 1)
  minΔz = (H  / Nᶻ / (N + 1))^2
  CFL_acoustic = minΔx / cʰ
  CFL_diffusive = minΔz / (1000 * κᶻ)
  CFL_viscous = minΔz / νᶻ
  dt = 1//2 * minimum([CFL_acoustic, CFL_diffusive, CFL_viscous])
  nout = ceil(Int64, tout / dt)
  dt = tout / nout

  @info @sprintf("""Update
                    Acoustic CFL  = %.1f
                    Diffusive CFL = %.1f
                    Viscous CFL   = %.1f
                    Timestep      = %.1f""",
                 CFL_acoustic, CFL_diffusive, CFL_viscous, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )


  prob = SimpleBox{FT}(Lˣ, Lʸ, H, τₒ, fₒ, β, λʳ, θᴱ)

  model = HBModel{typeof(prob),FT}(prob, cʰ, cʰ, cᶻ, αᵀ, νʰ, νᶻ, κʰ, κᶻ)

  dg = OceanDGModel(model,
                    grid,
                    Rusanov(),
                    CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0); forcecpu=true)
  update_aux!(dg, model, Q, FT(0))
  update_aux_diffusive!(dg, model, Q, FT(0))

  if isdir(vtkpath)
    rm(vtkpath, recursive=true)
  end
  mkpath(vtkpath)
  mkpath(vtkpath*"/weekly")
  mkpath(vtkpath*"/monthly")

  step = [0, 0]
  function do_output(span, step)
    outprefix = @sprintf("%s/%s/mpirank%04d_step%04d",vtkpath, span,
                         MPI.Comm_rank(mpicomm), step)
    @info "doing VTK output" outprefix
    statenames = flattenednames(vars_state(model, eltype(Q)))
    auxnames = flattenednames(vars_aux(model, eltype(Q)))
    writevtk(outprefix, Q, dg, statenames, dg.auxstate, auxnames)
  end

  do_output("weekly", step[1])
  cbvtkw = GenericCallbacks.EveryXSimulationSteps(nout)  do (init=false)
    do_output("weekly", step[1])
    step[1] += 1
    nothing
  end

  do_output("monthly", step[2])
  cbvtkm = GenericCallbacks.EveryXSimulationSteps(5*nout)  do (init=false)
    do_output("monthly", step[2])
    step[2] += 1
    nothing
  end

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

  lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s""" eng0 ArrayType

  solve!(Q, lsrk, nothing; timeend=timeend, callbacks=(cbinfo,cbvtkw,cbvtkm))

  return nothing
end
