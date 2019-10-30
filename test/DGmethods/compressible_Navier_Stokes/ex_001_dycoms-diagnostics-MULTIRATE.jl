using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.MultirateRungeKuttaMethod
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.StrongStabilityPreservingRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod
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

using CLIMA.Atmos: vars_state, vars_aux

using Random 
const seed = MersenneTwister(0)

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray 
else
  const ArrayType = Array
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

"""
  Initial Condition for DYCOMS_RF01 LES
@article{doi:10.1175/MWR2930.1,
author = {Stevens, Bjorn and Moeng, Chin-Hoh and Ackerman, 
          Andrew S. and Bretherton, Christopher S. and Chlond, 
          Andreas and de Roode, Stephan and Edwards, James and Golaz, 
          Jean-Christophe and Jiang, Hongli and Khairoutdinov, 
          Marat and Kirkpatrick, Michael P. and Lewellen, David C. and Lock, Adrian and 
          Maeller, Frank and Stevens, David E. and Whelan, Eoin and Zhu, Ping},
title = {Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus},
journal = {Monthly Weather Review},
volume = {133},
number = {6},
pages = {1443-1462},
year = {2005},
doi = {10.1175/MWR2930.1},
URL = {https://doi.org/10.1175/MWR2930.1},
eprint = {https://doi.org/10.1175/MWR2930.1}
}
"""
function Initialise_DYCOMS!(state::Vars, aux::Vars, (x,y,z), t)
  FT         = eltype(state)
  xvert::FT  = z

  epsdv::FT     = molmass_ratio
  q_tot_sfc::FT = 8.1e-3
  Rm_sfc::FT    = gas_constant_air(PhasePartition(q_tot_sfc))
  ρ_sfc::FT     = 1.22
  P_sfc::FT     = 1.0178e5
  T_BL::FT      = 285.0
  T_sfc::FT     = P_sfc/(ρ_sfc * Rm_sfc);
  
  q_liq::FT      = 0
  q_ice::FT      = 0
  zb::FT         = 600   
  zi::FT         = 840 
  dz_cloud       = zi - zb
  q_liq_peak::FT = 4.5e-4
  
  if xvert > zb && xvert <= zi        
    q_liq = (xvert - zb)*q_liq_peak/dz_cloud
  end
  if ( xvert <= zi)
    θ_liq  = FT(289)
    q_tot  = FT(8.1e-3)
  else
    θ_liq = FT(297.5) + (xvert - zi)^(FT(1/3))
    q_tot = FT(1.5e-3)
  end

  q_pt = PhasePartition(q_tot, q_liq, FT(0))
  Rm    = gas_constant_air(q_pt)
  cpm   = cp_m(q_pt)
  #Pressure
  H = Rm_sfc * T_BL / grav;
  P = P_sfc * exp(-xvert/H);
  #Exner
  exner_dry = exner(P, PhasePartition(FT(0)))
  #Temperature 
  T             = exner_dry*θ_liq + LH_v0*q_liq/(cpm*exner_dry);
  #Density
  ρ             = P/(Rm*T);
  #Potential Temperature
  θv     = virtual_pottemp(T, P, q_pt)
  # energy definitions
  u, v, w     = FT(7), FT(-5.5), FT(0)
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = FT(1//2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(U, V, W) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
end   

function run(mpicomm, ArrayType, dim, topology, 
             polynomialorder, timeend, FT, FastMethod, 
             C_smag, LHF, SHF, C_drag, zmax, zsponge, brickrange)
  
  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder)

  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(IsothermalProfile(FT(T_0+15)), FT(0)),
                     SmagorinskyLilly{FT}(C_smag),
                     EquilMoist(),
                     StevensRadiation{FT}(85,1,840,1.22,3.75e-6,70,22),
                     (Gravity(),
                      RayleighSponge{FT}(zmax, zsponge,1),
                      Subsidence(),
                      GeostrophicForcing{FT}(7.62e-5,7,-5.5)),
                     DYCOMS_BC{FT}(C_drag,LHF,SHF),
                     Initialise_DYCOMS!)

  # The linear model has the fast time scales
  fast_model = AtmosAcousticLinearModel(model)
  # The nonlinear model has the slow time scales
  slow_model = RemainderModel(model, (fast_model,))

  dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty())
  
  fast_dg = DGModel(fast_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                    auxstate=dg.auxstate)
  
  slow_dg = DGModel(slow_model,
                    grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty();
                    auxstate=dg.auxstate)

  # determine the slow time step
  elementsize = minimum(step.(brickrange))
  slow_dt = 0.2 #elementsize / soundspeed_air(FT(T_0)) / polynomialorder ^ 2
  nsteps = ceil(Int, timeend / slow_dt)
  slow_dt = timeend / nsteps

  # arbitrary and not needed for stabilty, just for testing
  fast_dt = slow_dt / 3

  Q = init_ode_state(dg, FT(0))

  slow_ode_solver = LSRK144NiegemannDiehlBusch(slow_dg, Q; dt = slow_dt)
  # check if FastMethod is ARK, is there a better way ?
  fast_ode_solver = FastMethod(fast_dg, Q; dt = fast_dt)

  ode_solver = MultirateRungeKutta((slow_ode_solver, fast_ode_solver))

  eng0 = norm(Q)
  @info @sprintf """Starting 
                    slow_dt   = %.16e
                    fast_dt   = %.16e
                    norm(Q₀)  = %.16e
                    """ slow_dt fast_dt eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      runtime = Dates.format(convert(DateTime, now() - starttime[]), dateformat"HH:MM:SS")
      @info @sprintf """Update
                        simtime = %.16e
                        runtime = %s
                        norm(Q) = %.16e
                        """ gettime(ode_solver) runtime energy
    end
    nothing
  end
  # create vtk dir
  vtkdir = "vtk_dycoms_multirate" * "_$(FastMethod)"
  mkpath(vtkdir)
  vtkstep = 0
  # output initial step
  # setup the output callback
  outputtime = timeend
  cbvtk = GenericCallbacks.EveryXSimulationSteps(floor(outputtime / slow_dt)) do
    vtkstep += 1
    Qe = init_ode_state(dg, gettime(ode_solver))
    nothing
  end

  solve!(Q, ode_solver; timeend=timeend, callbacks=(cbinfo,cbvtk))

  # final statistics
  Qe = init_ode_state(dg, timeend)
  engf = norm(Q)
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished 
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf/engfe
engf/eng0
end
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

  # Problem type
  FT = Float32
  # DG polynomial order 
  N = 4
  # SGS Filter constants
  C_smag = FT(0.15)
  LHF    = FT(-115)
  SHF    = FT(-15)
  C_drag = FT(0.0011)
  # User defined domain parameters
  brickrange = (grid1d(0, 2000, elemsize=FT(50)*N),
                grid1d(0, 2000, elemsize=FT(50)*N),
                grid1d(0, 1500, elemsize=FT(20)*N))
  zmax = brickrange[3][end]
  zsponge = FT(1200.0)
  
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity = (true, true, false),
                              boundary=((0,0),(0,0),(1,2)))
  dt = 0.5
  timeend = FT(5000dt)
  dim = 3
  FastMethod = SSPRK33ShuOsher
  result = run(mpicomm, ArrayType, dim, topl, 
               N, timeend, FT, FastMethod, 
               C_smag, LHF, SHF, C_drag, zmax, zsponge, brickrange)
end
#nothing
