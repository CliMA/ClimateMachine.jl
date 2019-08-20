
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
using CLIMA.Vtk

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


function Initialise_DYCOMS!(state::Vars, aux::Vars, (x,y,z), t)
    
  DF         = eltype(state)
  xvert::DF  = z

  #These constants are those used by Stevens et al. (2005)
  R_d::DF     = 287.0
  cp_d::DF    = 1015.0
  cp_v::DF    = 1859.0
  cp_l::DF    = 4181.0
  Lv::DF      = 2.47e6
  epsdv::DF   = 1.61
  g::DF       = grav
  p0::DF      = 1.0178e5
  ρ0::DF      = 1.22
  r_tot_sfc::DF=8.1e-3
  Rm_sfc          = R_d * (1.0 + (epsdv - 1.0)*r_tot_sfc)
  ρ_sfc::DF   = 1.22
  P_sfc           = 1.0178e5
  T_0::DF     = 285.0
  T_sfc           = P_sfc/(ρ_sfc * Rm_sfc);
  
  # --------------------------------------------------
  # INITIALISE ARRAYS FOR INTERPOLATED VALUES
  # --------------------------------------------------
  randnum1   = 0#rand(1)[1] / 100
  randnum2   = 0#rand(1)[1] / 100
    
  q_liq      = 0.0
  q_ice      = 0.0
  zb         = 600.0    #initial cloud bottom
  zi         = 840.0    #initial cloud top
  dz_cloud   = zi - zb
  q_liq_peak = 0.00045 #cloud mixing ratio at z_i    
  if xvert > zb && xvert <= zi        
    q_liq = (xvert - zb)*q_liq_peak/dz_cloud
  end

  if ( xvert <= zi)
    θ_liq  = 289.0
    r_tot      = 8.1e-3                  #kg/kg  specific humidity --> approx. to mixing ratio is ok
    q_tot      = r_tot #/(1.0 - r_tot)     #total water mixing ratio
  else
    θ_liq = 297.5 + (xvert - zi)^(1/3)
    r_tot     = 1.5e-3                    #kg/kg  specific humidity --> approx. to mixing ratio is ok
    q_tot     = r_tot #/(1.0 - r_tot)      #total water mixing ratio
  end

  if xvert <= 200.0
      θ_liq += randnum1 * θ_liq 
      q_tot += randnum2 * q_tot
  end
  
  Rm       = R_d * (1 + (epsdv - 1)*q_tot - epsdv*q_liq);
  cpm     = cp_d + (cp_v - cp_d)*q_tot + (cp_l - cp_v)*q_liq;

  #Pressure
  H = Rm_sfc * T_0 / g;
  P = P_sfc * exp(-xvert/H);
  
  #Exner
  exner = (P/P_sfc)^(R_d/cp_d);
  
  #T, Tv 
  T     = exner*θ_liq + Lv*q_liq/(cpm*exner);
  Tv    = T*(1 + (epsdv - 1)*q_tot - epsdv*q_liq);
  
  #Density
  ρ  = P/(Rm*T);
  
  #θ, θv
  θ      = T/exner;
  θv     = θ*(1 + (epsdv - 1)*q_tot - epsdv*q_liq);
  PhPart = PhasePartition(q_tot, q_liq, q_ice)

  # energy definitions
  u, v, w     = 7, -5.5, 0.0 
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = 0.5 * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, PhPart)

  state.ρ     = ρ
  state.ρu    = SVector(U, V, W) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
    
end


function source!(source::Vars, state::Vars, aux::Vars, t::Real)
  nothing
end

function run(mpicomm, ArrayType, dim, topl, N, timeend, DF, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  model = AtmosModel(ConstantViscosityWithDivergence(DF(1.0)),
                     EquilMoist(),
                     NoRadiation(),
                     source!, DYCOMS_BC(), Initialise_DYCOMS!)

  dg = DGModel(model,
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DF(0))


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


  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, ))

  # Print some end of the simulation information
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
  errf
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

  polynomialorder = 4
  (xmin,xmax) = (0, 1000)
  (ymin,ymax) = (0, 1000)
  (zmin, zmax) =(0, 1500)
  Ne = (10,2,10)
  DF = Float64
  brickrange = (range(DF(xmin), length=Ne[1]+1, DF(xmax)),
                range(DF(ymin), length=Ne[2]+1, DF(ymax)),
                range(DF(zmin), length=Ne[3]+1, DF(zmax)))
  topl = BrickTopology(mpicomm, brickrange,periodicity = (true, true, false))
  dt = 0.001
  timeend = dt
  dim = 3
  @info (ArrayType, DF, dim)
  result = run(mpicomm, ArrayType, dim, topl, 
                  polynomialorder, timeend, DF, dt)
end


#nothing
