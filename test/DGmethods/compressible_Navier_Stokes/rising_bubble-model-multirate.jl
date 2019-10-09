# Load Packages 
using MPI
using CLIMA
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.MPIStateArrays
using CLIMA.MultirateRungeKuttaMethod
using CLIMA.LinearSolvers
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.SubgridScaleParameters
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks: EveryXSimulationSteps, EveryXWallTimeSeconds
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using Random
using CLIMA.Atmos: vars_state, ReferenceState
import CLIMA.Atmos: atmos_init_aux!, vars_aux
using CLIMA.DGmethods: EveryDirection, HorizontalDirection, VerticalDirection

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
const dim = 3
const polynomialorder = 4
const domain_start = (0, 0, 0)
const domain_end = (1000, dim == 2 ? 1 : 1000, 1000)
const Ne = (10, dim == 2 ? 1 : 10, 70)
const Δxyz = @. (domain_end - domain_start) / Ne / polynomialorder
const dt = min(Δxyz...) / soundspeed_air(300.0) / polynomialorder
const timeend = 100
const output_vtk = true
const outputtime = 5
const smooth_bubble = true
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
  DF            = eltype(state)
  R_gas::DF     = R_d
  c_p::DF       = cp_d
  c_v::DF       = cv_d
  γ::DF         = c_p / c_v
  p0::DF        = MSLP
  
  xc::DF        = 500
  zc::DF        = 260
  r             = sqrt((x1 - xc)^2 + (x2 - xc)^2 + (x3 - zc)^2)
  rc::DF        = 250
  θ_ref::DF     = 303
  Δθ::DF        = 0
  θ_c::DF = 1 // 2
 
  if smooth_bubble
    a::DF   =  50
    s::DF   = 100
    if r <= a
      Δθ = θ_c
    elseif r > a
      Δθ = θ_c * exp(-(r - a)^2 / s^2)
    end
  else
    if r <= rc 
      Δθ          = θ_c
    end
  end
  
  if t < 0
    Δθ = 0
  end

  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = DF(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(DF(0),DF(0),DF(0))
  # energy definitions
  e_kin        = DF(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  #state.moisture.ρq_tot = DF(0)
end
struct RisingBubbleReferenceState <: ReferenceState end
vars_aux(::RisingBubbleReferenceState, DT) = @vars(ρ::DT, ρe::DT, p::DT, T::DT)
function atmos_init_aux!(m::RisingBubbleReferenceState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  x1, x2, x3 = geom.coord
  DF            = eltype(aux)
  R_gas::DF     = R_d
  c_p::DF       = cp_d
  c_v::DF       = cv_d
  γ::DF         = c_p / c_v
  p0::DF        = MSLP
  θ_ref::DF     = 303

  θ            = θ_ref
  π_exner      = DF(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  e_kin        = DF(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)

  aux.ref_state.ρ = ρ
  aux.ref_state.ρe = ρe_tot
  aux.ref_state.p = P
  aux.ref_state.T = T
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
                     RisingBubbleReferenceState(),
                     ConstantViscosityWithDivergence(DF(0)),
                     DryModel(), 
                     #EquilMoist(), 
                     NoRadiation(),
                     Gravity(),
                     NoFluxBC(),
                     Initialise_Rising_Bubble!)

  fast_model = AtmosAcousticLinearModel(model)
  slow_model = AtmosAcousticNonlinearModel(model)

  # -------------- Define dgbalancelaw --------------------------- # 
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  hor_fast_dg = DGModel(fast_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                    auxstate=dg.auxstate, direction=HorizontalDirection())

  ver_fast_dg = DGModel(fast_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                    auxstate=dg.auxstate, direction=VerticalDirection())

  slow_dg = DGModel(slow_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                    auxstate=dg.auxstate)

  Q = init_ode_state(dg, DF(0))
  Qinit = init_ode_state(dg, DF(-1))


  slow_dt = 13dt
  fast_dt = dt
  slow_ode_solver = LSRK54CarpenterKennedy(slow_dg, Q; dt = slow_dt)

  linearsolver = GeneralizedMinimalResidual(10, Q, sqrt(eps(DF)))
  fast_ode_solver = ARK548L2SA2KennedyCarpenter(hor_fast_dg, ver_fast_dg,
                                                linearsolver, Q; dt = fast_dt,
                                                split_nonlinear_linear=true)

  ode_solver = MultirateRungeKutta((slow_ode_solver, fast_ode_solver))

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s
  FloatType = %s""" eng0 ArrayType DF

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(ode_solver),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end
  callbacks = (cbinfo,)

  if output_vtk
    # create vtk dir
    vtkdir = "vtk_rtb"
    mkpath(vtkdir)
    
    vtkstep = 0
    # output initial step
    Qdiff = Q .- Qinit
    do_output(mpicomm, vtkdir, vtkstep, dg, Qdiff, model)

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime / slow_dt)) do
      vtkstep += 1
      Qdiff = Q .- Qinit
      do_output(mpicomm, vtkdir, vtkstep, dg, Qdiff, model)
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, ode_solver; timeend=DF(timeend), callbacks=callbacks)
  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, DF(timeend))
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

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "rtb")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  writevtk(filename, Q, dg, statenames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, statenames)

    @info "Done writing VTK: $pvtuprefix"
  end
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
    FloatType = (Float64,)
    for DF in FloatType
      brickrange = ntuple(d -> range(DF(domain_start[d]); length=Ne[d]+1, stop=domain_end[d]), 3)
      topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, false, false))
      engf_eng0 = run(mpicomm, ArrayType, 
                      topl, dim, Ne, polynomialorder, 
                      timeend, DF, dt)
      #@test engf_eng0 ≈ DF(9.9999993807738441e-01)
    end
  end
end

