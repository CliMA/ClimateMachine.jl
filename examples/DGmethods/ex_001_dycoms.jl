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
  const ArrayTypes = (CuArray,) 
else
  const ArrayTypes = (Array,)
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
  ziplus::FT     = 875
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
  u1, u2 = FT(6), FT(7)
  v1, v2 = FT(-4.25), FT(-5.5)
  w = FT(0)
  if (xvert <= zi)
      u, v = u1, v1
  elseif (xvert >= ziplus)
      u, v = u2, v2
  else
      m = (ziplus - zi)/(u2 - u1)
      u = (xvert - zi)/m + u1
      
      m = (ziplus - zi)/(v2 - v1)
      v = (xvert - zi)/m + v1
  end    
  e_kin       = FT(1//2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(ρ*u, ρ*v, ρ*w) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
end   


function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, zmax, zsponge)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          )
  u_ref =  7.0
  v_ref = -5.5
  w_ref = 0.0  
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{FT}(C_smag),
                     EquilMoist(),
                     StevensRadiation{FT}(85, 1, 840, 1.22, 3.75e-6, 70, 22),
                     (Gravity(), 
                      RayleighSponge{FT}(zmax, zsponge, 1, u_ref, v_ref, w_ref), 
                      Subsidence(), 
                      GeostrophicForcing{FT}(7.62e-5, 7, -5.5)), 
                     DYCOMS_BC{FT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
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

  step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(5000) do (init=false)
    mkpath("./vtk-dycoms/")
    outprefix = @sprintf("./vtk-dycoms/dycoms_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)), 
             dg.auxstate, flattenednames(vars_aux(model,FT)))
        
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
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
    # Problem type
    FT = Float64
    # DG polynomial order 
    N = 4
    # SGS Filter constants
    C_smag = FT(0.15)
    LHF    = FT(115)
    SHF    = FT(15)
    C_drag = FT(0.0011)
    # User defined domain parameters
    brickrange = (grid1d(0, 2000, elemsize=FT(50)*N),
                  grid1d(0, 2000, elemsize=FT(50)*N),
                  grid1d(0, 1500, elemsize=FT(20)*N))
    zmax = brickrange[3][end]
    zsponge = FT(1200)
    
    topl = StackedBrickTopology(mpicomm, brickrange,
                                periodicity = (true, true, false),
                                boundary=((0,0),(0,0),(1,2)))
    dt = 0.01
    timeend = 100dt
    dim = 3
    @info (ArrayType, FT, dim)
    result = run(mpicomm, ArrayType, dim, topl, 
                 N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, zmax, zsponge)
    @test result ≈ FT(1.000033004957446) #FT(0.9999737848359238)
  end
end

#nothing
