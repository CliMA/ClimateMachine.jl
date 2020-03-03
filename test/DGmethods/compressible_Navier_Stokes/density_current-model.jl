
# Load Packages
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
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
using CLIMA.Atmos: vars_state, vars_aux

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- #
const dim               = 3
const (xmin, xmax)      = (0,12800)
const (ymin, ymax)      = (0,400)
const (zmin, zmax)      = (0,6400)
const Ne                = (100,2,50)
const polynomialorder   = 4
const dt                = 0.01
const timeend           = 10dt

# ------------- Initial condition function ----------- #
"""
@article{doi:10.1002/fld.1650170103,
author = {Straka, J. M. and Wilhelmson, Robert B. and Wicker, Louis J. and Anderson, John R. and Droegemeier, Kelvin K.},
title = {Numerical solutions of a non-linear density current: A benchmark solution and comparisons},
journal = {International Journal for Numerical Methods in Fluids},
volume = {17},
number = {1},
pages = {1-22},
doi = {10.1002/fld.1650170103},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650170103},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.1650170103},
year = {1993}
}
"""
function Initialise_Density_Current!(bl, state::Vars, aux::Vars, (x1,x2,x3), t)
  FT                = eltype(state)
  R_gas::FT         = R_d
  c_p::FT           = cp_d
  c_v::FT           = cv_d
  p0::FT            = MSLP
  # initialise with dry domain
  q_tot::FT         = 0
  q_liq::FT         = 0
  q_ice::FT         = 0
  # perturbation parameters for rising bubble
  rx                = 4000
  rz                = 2000
  xc                = 0
  zc                = 3000
  r                 = sqrt((x1 - xc)^2/rx^2 + (x3 - zc)^2/rz^2)
  θ_ref::FT         = 300
  θ_c::FT           = -15
  Δθ::FT            = 0
  if r <= 1
    Δθ = θ_c * (1 + cospi(r))/2
  end
  qvar              = PhasePartition(q_tot)
  θ                 = θ_ref + Δθ # potential temperature
  π_exner           = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ                 = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  ts                = LiquidIcePotTempSHumEquil(θ, ρ, q_tot)
  q_pt              = PhasePartition(ts)

  U, V, W           = FT(0) , FT(0) , FT(0)  # momentum components
  # energy definitions
  e_kin             = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
  e_pot             = gravitational_potential(bl.orientation, aux)
  e_int             = internal_energy(ts)
  E                 = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
  state.ρ      = ρ
  state.ρu     = SVector(U, V, W)
  state.ρe     = E
  state.moisture.ρq_tot = ρ*q_pt.tot
end
# --------------- Driver definition ------------------ #
function run(mpicomm, ArrayType,
             topl, dim, Ne, polynomialorder,
             timeend, FT, dt)
  # -------------- Define grid ----------------------------------- #
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                           )
  # -------------- Define model ---------------------------------- #
  source = Gravity()
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         ref_state=HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0)),
                        turbulence=AnisoMinDiss{FT}(1),
                            source=source,
                 boundarycondition=NoFluxBC(),
                        init_state=Initialise_Density_Current!)
  # -------------- Define dgbalancelaw --------------------------- #
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

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
      writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)), dg.auxstate, flattenednames(vars_aux(model,FT)))
      step[1] += 1
      nothing
  end


  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo,cbvtk))
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
  CLIMA.init()
  ArrayType = CLIMA.array_type()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  for FT in (Float32, Float64)
    brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                  range(FT(ymin); length=Ne[2]+1, stop=ymax),
                  range(FT(zmin); length=Ne[3]+1, stop=zmax))
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
    engf_eng0 = run(mpicomm, ArrayType,
                    topl, dim, Ne, polynomialorder,
                    timeend, FT, dt)
    @test engf_eng0 ≈ FT(9.9999970927037096e-01)
  end
end


#nothing
