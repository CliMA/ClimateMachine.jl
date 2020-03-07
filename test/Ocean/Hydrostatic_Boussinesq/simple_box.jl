using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.VariableTemplates: flattenednames
using CLIMA.HydrostaticBoussinesq
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.PlanetParameters: grav
using CLIMA.HydrostaticBoussinesq: AbstractHydrostaticBoussinesqProblem
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


HBModel   = HydrostaticBoussinesqModel

struct OceanGyre{T} <: AbstractHydrostaticBoussinesqProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  fₒ::T
  β::T
  λʳ::T
  θᴱ::T
end

@inline function ocean_boundary_state!(m::HBModel, p::OceanGyre, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloorNoSlip(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
  end
end

# A is Filled afer the state
function ocean_init_aux!(m::HBModel, P::OceanGyre, A, geom)
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

  A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
  A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

function ocean_init_state!(P::OceanGyre, Q, A, coords, t)
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
vtkpath = "vtk_ekman_spiral"

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
  ArrayType = CLIMA.array_type()
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
  minΔz = H  / Nᶻ / (N + 1)
  CFL_gravity = minΔx / cʰ
  CFL_diffusive = minΔz^2 / (1000 * κᶻ)
  CFL_viscous = minΔz^2 / νᶻ
  dt = 1//2 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])
  nout = ceil(Int64, tout / dt)
  dt = tout / nout

  @info @sprintf("""Update
                    Gravity CFL   = %.1f
                    Diffusive CFL = %.1f
                    Viscous CFL   = %.1f
                    Timestep      = %.1f""",
                 CFL_gravity, CFL_diffusive, CFL_viscous, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )


  prob = OceanGyre{FT}(Lˣ, Lʸ, H, τₒ, fₒ, β, λʳ, θᴱ)

  model = HBModel{FT}(prob, cʰ = cʰ)

  dg = OceanDGModel(model,
                    grid,
                    Rusanov(),
                    CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0); init_on_cpu=true)
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
