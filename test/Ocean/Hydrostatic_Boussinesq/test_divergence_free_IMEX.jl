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
using CLIMA.VariableTemplates: flattenednames, Vars
using CLIMA.HydrostaticBoussinesq
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.PlanetParameters: grav
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.ColumnwiseLUSolver: ManyColumnLU, SingleColumnLU
import CLIMA.HydrostaticBoussinesq: ocean_init_aux!, ocean_init_state!,
                                    ocean_boundary_state!,
                                    CoastlineFreeSlip, CoastlineNoSlip,
                                    OceanFloorFreeSlip, OceanFloorNoSlip,
                                    OceanSurfaceNoStressNoForcing,
                                    OceanSurfaceStressNoForcing,
                                    OceanSurfaceNoStressForcing,
                                    OceanSurfaceStressForcing
import CLIMA.DGmethods: update_aux!, vars_state, vars_aux, VerticalDirection
using Test
using GPUifyLoops

const ArrayType = CLIMA.array_type()

HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

@inline function ocean_boundary_state!(m::HBModel, bctype, x...)
  if bctype == 1
    ocean_boundary_state!(m, CoastlineNoSlip(), x...)
  elseif bctype == 2
    ocean_boundary_state!(m, OceanFloorFreeSlip(), x...)
  elseif bctype == 3
    ocean_boundary_state!(m, OceanSurfaceStressNoForcing(), x...)
  end
end

struct HomogeneousSimpleBox{T} <: HydrostaticBoussinesqProblem
  Lˣ::T
  Lʸ::T
  H::T
  τₒ::T
  fₒ::T
  β::T
end

HSBox = HomogeneousSimpleBox

# aux is Filled afer the state
function ocean_init_aux!(m::HBModel, P::HSBox, A, geom)
  FT = eltype(A)
  @inbounds y = geom.coord[2]

  Lʸ = P.Lʸ
  τₒ = P.τₒ
  fₒ = P.fₒ
  β  = P.β

  A.τ  = -τₒ * cos(y * π / Lʸ)
  A.f  =  fₒ + β * y

  A.ν = @SVector [m.νʰ, m.νʰ, m.νᶻ]
  A.κ = @SVector [m.κʰ, m.κʰ, m.κᶻ]
end

function ocean_init_state!(p::HSBox, state, aux, coords, t)
  @inbounds z = coords[3]
  @inbounds H = p.H

  state.u = @SVector [rand(),rand()]
  state.η = 0
  state.θ = 20
end

function main()
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
  dt = 60
  nout = ceil(Int64, tout / dt)
  dt = tout / nout

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )


  prob = HSBox{FT}(Lˣ, Lʸ, H, τₒ, fₒ, β)

  model = HBModel{typeof(prob),FT}(prob, cʰ, cʰ, cᶻ, αᵀ, νʰ, νᶻ, κʰ, κᶻ)

  linearmodel = LinearHBModel(model)

  dg = OceanDGModel(model,
                    grid,
                    Rusanov(),
                    CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient())

  lineardg = DGModel(linearmodel,
                     grid,
                     Rusanov(),
                     CentralNumericalFluxDiffusive(),
                     CentralNumericalFluxGradient();
                     direction=VerticalDirection(),
                     auxstate=dg.auxstate)

  Q = init_ode_state(dg, FT(0); forcecpu=true)
  update_aux!(dg, model, Q, FT(0))

  linearsolver = SingleColumnLU() # ManyColumnLU()

  odesolver = ARK2GiraldoKellyConstantinescu(dg, lineardg, linearsolver, Q;
                                             dt = dt, t0 = 0,
                                             split_nonlinear_linear=false)

  cbvector = make_callbacks(step, nout, mpicomm, odesolver, dg, model, Q)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s""" eng0 ArrayType

  solve!(Q, odesolver, nothing; timeend=timeend, callbacks=cbvector, adjustfinalstep=false)

  maxQ =  Vars{vars_state(model, FT)}(maximum(Q, dims=(1,3)))
  minQ =  Vars{vars_state(model, FT)}(minimum(Q, dims=(1,3)))

  @test maxQ.θ ≈ minQ.θ

  return nothing
end

function make_callbacks(step, nout, mpicomm, odesolver, dg, model, Q)
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(odesolver),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  return (cbinfo,)
end

#################
# RUN THE TESTS #
#################
FT = Float64

const timeend = 3600 # s
const tout    = 60 * 60 # s

const N  = 4
const Nˣ = 20
const Nʸ = 20
const Nᶻ = 50
const Lˣ = 4e6   # m
const Lʸ = 4e6   # m
const H  = 400   # m

xrange = range(FT(0);  length=Nˣ+1, stop=Lˣ)
yrange = range(FT(0);  length=Nʸ+1, stop=Lʸ)
zrange = range(FT(-H); length=Nᶻ+1, stop=0)

const cʰ = sqrt(grav * H)
const cᶻ = 0

const τₒ = 1e-1  # (m/s)^2
const fₒ = 1e-4  # Hz
const β  = 1e-11 # Hz / m

const αᵀ = 2e-4  # (m/s)^2 / K
const νʰ = 5e3   # m^2 / s
const νᶻ = 5e-3  # m^2 / s
const κʰ = 1e3   # m^2 / s
const κᶻ = 1e-10  # m^2 / s

main()
