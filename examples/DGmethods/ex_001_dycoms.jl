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

"""
  Initial Condition for DYCOMS_RF01 LES
"""
function Initialise_DYCOMS!(state::Vars, aux::Vars, (x,y,z), t)
    
  DF         = eltype(state)
  xvert::DF  = z

  epsdv::DF     = molmass_ratio
  q_tot_sfc::DF = 8.1e-3
  Rm_sfc::DF    = gas_constant_air(PhasePartition(q_tot_sfc))
  ρ_sfc::DF     = 1.22
  P_sfc::DF     = 1.0178e5
  T_BL::DF      = 285.0
  T_sfc::DF     = P_sfc/(ρ_sfc * Rm_sfc);
  
  q_liq::DF      = 0
  q_ice::DF      = 0
  zb::DF         = 600   
  zi::DF         = 840 
  dz_cloud       = zi - zb
  q_liq_peak::DF = 4.5e-4
  if xvert > zb && xvert <= zi        
    q_liq = (xvert - zb)*q_liq_peak/dz_cloud
  end

  if ( xvert <= zi)
    θ_liq  = DF(289)
    q_tot  = q_tot_sfc
  else
    θ_liq = DF(297.5) + (xvert - zi)^(DF(1//3))
    q_tot = DF(1.5e-3)
  end
  #=
  if xvert <= 200.0
      θ_liq += θ_liq 
      q_tot += q_tot
  end
  =# 

  #Pressure
  H = Rm_sfc * T_BL / grav;
  P = P_sfc * exp(-xvert/H);
  
  # Thermodynamic state
  q_pt = PhasePartition(q_tot, q_liq, q_ice)
  ts = LiquidIcePotTempSHumEquil_no_ρ(θ_liq, q_pt, P)

  #Density
  ρ  = air_density(ts)
  T  = air_temperature(ts)
  
  # energy definitions
  u, v, w     = DF(7), DF(-5.5), DF(0) 
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = (u^2 + v^2 + w^2) / 2
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)

  state.ρ     = ρ
  state.ρu    = SVector(U, V, W) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
    
  q_test = q_pt.liq
  q_max = max(q_test,q_pt.liq)

end

function source!(source::Vars, state::Vars, aux::Vars, t::Real)
  DF = eltype(state)
  source.ρu = SVector(DF(0), DF(0), -state.ρ * grav)
end

function run(mpicomm, ArrayType, dim, topl, N, timeend, DF, dt, C_smag, Δ)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  model = AtmosModel(SmagorinskyLilly(DF(C_smag),DF(Δ)),
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

  step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1000) do (init=false)
    mkpath("./vtk-dycoms/")
    outprefix = @sprintf("./vtk-dycoms/dycoms_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, dg)
        
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, cbvtk))

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
  
  # Problem type
  DF = Float32
  # DG polynomial order 
  polynomialorder = 4
  # User specified grid spacing
  Δx    = DF(35)
  Δy    = DF(35)
  Δz    = DF(10)
  # SGS Filter constants
  C_smag = DF(0.15)
  Δ     = DF(cbrt(Δx * Δy * Δz))
  # Physical domain extents 
  (xmin, xmax) = (0, 2000)
  (ymin, ymax) = (0, 2000)
  (zmin, zmax) = (0, 1500)
  #Get Nex, Ney from resolution
  Lx = xmax - xmin
  Ly = ymax - ymin
  Lz = zmax - ymin
  # User defines the grid size:
  Nex = ceil(Int64, (Lx/Δx - 1)/polynomialorder)
  Ney = ceil(Int64, (Ly/Δy - 1)/polynomialorder)
  Nez = ceil(Int64, (Lz/Δz - 1)/polynomialorder)
  Ne = (Nex, Ney, Nez)
  # User defined domain parameters
  brickrange = (range(DF(xmin), length=Ne[1]+1, DF(xmax)),
                range(DF(ymin), length=Ne[2]+1, DF(ymax)),
                range(DF(zmin), length=Ne[3]+1, DF(zmax)))
  topl = BrickTopology(mpicomm, brickrange,periodicity = (true, true, false), boundary=[1 2 3; 4 5 6])
  dt = 0.001
  timeend = dt
  dim = 3
  @info (ArrayType, DF, dim)
  result = run(mpicomm, ArrayType, dim, topl, 
                  polynomialorder, timeend, DF, dt, C_smag, Δ)
end

#nothing
