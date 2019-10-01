# Load Packages 
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.LinearSolvers
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.SubgridScaleParameters
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
using Random
using CLIMA.Atmos: vars_state, vars_aux

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
const (xmin,xmax)      = (0,1000)
const (ymin,ymax)      = (0,400)
const (zmin,zmax)      = (0,1000)
const Ne        = (10,2,10)
const polynomialorder = 4
const dim       = 3
const dt        = 0.1
const timeend   = 10dt
# ------------- Initial condition function ----------- # 
"""
@article{doi:10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2,
author = {Robert, A},
title = {Bubble Convection Experiments with a Semi-implicit Formulation of the Euler Equations},
journal = {Journal of the Atmospheric Sciences},
volume = {50},
number = {13},
pages = {1865-1873},
year = {1993},
doi = {10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
URL = {https://doi.org/10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
eprint = {https://doi.org/10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
}
"""
function Initialise_Rising_Bubble!(state::Vars, aux::Vars, (x1,x2,x3), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP
  
  xc::FT        = 500
  zc::FT        = 260
  r             = sqrt((x1 - xc)^2 + (x3 - zc)^2)
  rc::FT        = 250
  θ_ref::FT     = 303
  Δθ::FT        = 0
  
  if r <= rc 
    Δθ          = FT(1//2) 
  end
  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  state.moisture.ρq_tot = FT(0)
end
# --------------- Driver definition ------------------ # 
function run(mpicomm, ArrayType, LinearType,
             topl, dim, Ne, polynomialorder, 
             timeend, FT, dt)
  # -------------- Define grid ----------------------------------- # 
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                           )
  # -------------- Define model ---------------------------------- # 
  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(IsothermalProfile(FT(T_0)),FT(0)),
                     Vreman{FT}(C_smag),
                     EquilMoist(), 
                     NoRadiation(),
                     Gravity(),
                     NoFluxBC(),
                     Initialise_Rising_Bubble!)
  # -------------- Define dgbalancelaw --------------------------- # 
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  linmodel = LinearType(model)
  lindg = DGModel(linmodel,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty(); auxstate=dg.auxstate)

  Q = init_ode_state(dg, FT(0))

  linearsolver = GeneralizedMinimalResidual(10, Q, sqrt(eps(FT)))
  ark = ARK548L2SA2KennedyCarpenter(dg, lindg, linearsolver, Q; dt = dt, t0 = 0)


  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s
  FloatType = %s""" eng0 ArrayType FT

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
                     norm(Q) = %.16e""", ODESolvers.gettime(ark),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(3000)  do (init=false)
    mkpath("./vtk-rtb/")
      outprefix = @sprintf("./vtk-rtb/DC_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
      @debug "doing VTK output" outprefix
      writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)), param[1], flattenednames(vars_aux(model,FT)))
      step[1] += 1
      nothing
  end

  solve!(Q, ark; timeend=timeend, callbacks=(cbinfo,cbvtk))
  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))
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
    FloatType = (Float32, Float64)
    for FT in FloatType
      brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                    range(FT(ymin); length=Ne[2]+1, stop=ymax),
                    range(FT(zmin); length=Ne[3]+1, stop=zmax))
      topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
      for LinearType in (AtmosAcousticLinearModel, AtmosAcousticGravityLinearModel)
        engf_eng0 = run(mpicomm, ArrayType, LinearType,
                        topl, dim, Ne, polynomialorder, 
                        timeend, FT, dt)
        @test engf_eng0 ≈ FT(0.9999997771981113)
      end
    end
  end
end

#nothing
