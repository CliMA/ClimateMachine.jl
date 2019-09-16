
# Load Packages 
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
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
using CLIMA.SubgridScaleParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using Random

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array,)
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- # 
const dim               = 3
const (xmin, xmax)      = (0,12800)
const (ymin, ymax)      = (0,400)
const (zmin, zmax)      = (0,6400)
const Ne                = (50,2,25)
const polynomialorder   = 4
const dt                = 0.01
const timeend           = 10dt

# ------------- Initial condition function ----------- # 
function initialise_density_current!(state::Vars, aux::Vars, (x1,x2,x3), t)
  DF                = eltype(state)
  R_gas::DF         = R_d
  c_p::DF           = cp_d
  c_v::DF           = cv_d
  p0::DF            = MSLP
  # initialise with dry domain 
  q_tot::DF         = 0
  q_liq::DF         = 0
  q_ice::DF         = 0 
  # perturbation parameters for rising bubble
  rx                = 4000
  rz                = 2000
  xc                = 0
  zc                = 3000
  r                 = sqrt((x1 - xc)^2/rx^2 + (x3 - zc)^2/rz^2)
  θ_ref::DF         = 300
  θ_c::DF           = -15
  Δθ::DF            = 0
  if r <= 1
    Δθ = θ_c * (1 + cospi(r))/2
  end
  qvar              = PhasePartition(q_tot)
  θ                 = θ_ref + Δθ # potential temperature
  π_exner           = DF(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ                 = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  P                 = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T                 = P / (ρ * R_gas) # temperature
  U, V, W           = DF(0) , DF(0) , DF(0)  # momentum components
  # energy definitions
  e_kin             = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
  e_pot             = grav * x3
  e_int             = internal_energy(T, qvar)
  E                 = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
  
  state.ρ      = ρ
  state.ρu     = SVector(U, V, W)
  state.ρe     = E
  state.moisture.ρq_tot = DF(0)
end
# --------------- Driver definition ------------------ # 
function run(mpicomm, ArrayType, 
             topl, dim, Ne, polynomialorder, 
             timeend, DF, dt)
  # -------------- Define grid ----------------------------------- # 
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                           )
  # -------------- Define model ---------------------------------- # 
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{DF}(C_smag), 
                     EquilMoist(), 
                     NoRadiation(),
                     Gravity(), NoFluxBC(), initialise_density_current!)
  # -------------- Define dgbalancelaw --------------------------- # 
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DF(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s""" eng0 ArrayType

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
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

  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(3000)  do (init=false)
    mkpath("./vtk-dc/")
      outprefix = @sprintf("./vtk-dc/DC_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
      @debug "doing VTK output" outprefix
      writevtk(outprefix, Q, dg, flattenednames(vars_state(model, DF)))
      step[1] += 1
      nothing
  end


  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo,cbvtk))
  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, param, DF(timeend))
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
engf/eng0
end
# --------------- Test block / Loggers ------------------ # 
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
  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
  DF = Float64
  brickrange = (range(DF(xmin); length=Ne[1]+1, stop=xmax),
                range(DF(ymin); length=Ne[2]+1, stop=ymax),
                range(DF(zmin); length=Ne[3]+1, stop=zmax))
  topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
  engf_eng0 = run(mpicomm, ArrayType, 
                  topl, dim, Ne, polynomialorder, 
                  timeend, DF, dt)
  @test engf_eng0 ≈ DF(9.9999970927037096e-01)
  end
end

#nothing
