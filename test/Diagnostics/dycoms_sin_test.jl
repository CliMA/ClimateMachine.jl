using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Diagnostics
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.Atmos: vars_state, vars_aux
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging
using Printf
using Dates
using FileIO
using Test

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

function Initialise_DYCOMS!(state::Vars, aux::Vars, (x,y,z), t)
  FT            = eltype(state)
  xvert::FT     = z
  Rd::FT        = R_d
  # These constants are those used by Stevens et al. (2005)
  qref::FT      = FT(9.0e-3)
  q_tot_sfc::FT = qref
  q_pt_sfc      = PhasePartition(q_tot_sfc)
  Rm_sfc::FT    = 461.5 #gas_constant_air(q_pt_sfc)
  T_sfc::FT     = 292.5
  P_sfc::FT     = 101780 #MSLP
  ρ_sfc::FT     = P_sfc / Rm_sfc / T_sfc
  # Specify moisture profiles
  q_liq::FT      = 0
  q_ice::FT      = 0
  zb::FT         = 600         # initial cloud bottom
  zi::FT         = 840         # initial cloud top
  ziplus::FT     = 875
  dz_cloud       = zi - zb
  q_liq_peak::FT = 0.00045     # cloud mixing ratio at z_i
  if xvert > zb && xvert <= zi
    q_liq = (xvert - zb)*q_liq_peak/dz_cloud
  end
  if xvert <= zi
    θ_liq = FT(289)
    q_tot = qref
  else
    θ_liq = FT(297.5) + (xvert - zi)^(FT(1/3))
    q_tot = FT(1.5e-3)
  end

  # Calculate PhasePartition object for vertical domain extent
  q_pt  = PhasePartition(q_tot, q_liq, q_ice)
  Rm    = gas_constant_air(q_pt)

  # Pressure
  H     = Rm_sfc * T_sfc / grav;
  p     = P_sfc * exp(-xvert/H);
  # Density, Temperature
  T = air_temperature_from_liquid_ice_pottemp_given_pressure(θ_liq, p, q_pt)
  ρ = air_density(T, p, q_pt)

  # Assign State Variables
  u1, u2 = FT(6), FT(7)
  v1, v2 = FT(-4.25), FT(-5.5)
  w = FT(10 + 0.5 * sin(2 * π * ((x/1500) + (y/1500))))
  u = (5 + 2 * sin(2 * π * ((x/1500) + (y/1500))))
  v = FT(5 + 2 * sin(2 * π * ((x/1500) + (y/1500))))
  e_kin       = FT(1/2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(ρ*u, ρ*v, ρ*w)
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
end

let
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  out_dir = get(ENV, "OUT_DIR", "output")
  mkpath(out_dir)

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
  Δx, Δy, Δz = 50, 50, 20
  xmin, xmax = 0, 1500
  ymin, ymax = 0, 1500
  zmin, zmax = 0, 1500

  grid_resolution = [Δx, Δy, Δz]
  domain_size     = [xmin, xmax, ymin, ymax, zmin, zmax]
  dim             = length(grid_resolution)

  brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                grid1d(ymin, ymax, elemsize=FT(grid_resolution[2])*N),
                grid1d(zmin, zmax, elemsize=FT(grid_resolution[end])*N))
  zmax = brickrange[dim][end]
  zsponge = FT(1200.0)

  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity = (true, true, false),
                              boundary=((0,0),(0,0),(1,2)))
  dt = 0.01
  timeend =  dt

  # Grid setup (topl contains brickrange information)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  # Problem constants
  # Radiation model
  κ             = FT(85)
  α_z           = FT(1)
  z_i           = FT(840)
  D_subsidence  = FT(3.75e-6)
  ρ_i           = FT(1.13)
  F_0           = FT(70)
  F_1           = FT(22)
  # Geostrophic forcing
  f_coriolis    = FT(7.62e-5)
  u_geostrophic = FT(7.0)
  v_geostrophic = FT(-5.5)
  w_ref         = FT(0)
  u_relaxation  = SVector(u_geostrophic, v_geostrophic, w_ref)

  # Model definition
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{FT}(C_smag),
                     EquilMoist(),
                     NoPrecipitation(),
                     DYCOMSRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1),
                     NoSubsidence{FT}(),
                     (Gravity(),
                      RayleighSponge{FT}(zmax, zsponge, 1, u_relaxation, 2),
                      GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)),
                     DYCOMS_BC{FT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)
  # Balancelaw description
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())
  Q = init_ode_state(dg, FT(0))
  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  # Get statistics during run
  diagnostics_time_str = replace(string(now()), ":" => ".")
  cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(50) do (init=false)
    sim_time_str = string(ODESolvers.gettime(lsrk))
    gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                       out_dir, ODESolvers.gettime(lsrk))
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbdiagnostics,))

  # Get statistics at the end of the run
  sim_time_str = string(ODESolvers.gettime(lsrk))
  gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                     out_dir, ODESolvers.gettime(lsrk))

  # Check results
  mpirank = MPI.Comm_rank(mpicomm)
  if mpirank == 0
    d = load(joinpath(out_dir, "diagnostics-$(diagnostics_time_str).jld2"))
    Nqk = size(d["0.0"], 1)
    Nev = size(d["0.0"], 2)
    Err = 0
    Err1 = 0
    S = zeros(Nqk * Nev)
    S1 = zeros(Nqk * Nev)
    Z = zeros(Nqk * Nev)
    for ev in 1:Nev
      for k in 1:Nqk
        dv = Diagnostics.diagnostic_vars(d["0.0"][k,ev])
        S[k+(ev-1)*Nqk] = dv.vert_eddy_u_flx
        S1[k+(ev-1)*Nqk] = dv.u
        Z[k+(ev-1)*Nqk] = dv.z
        Err += (S[k+(ev-1)*Nqk] - 0.5)^2
        Err1 += (S1[k+(ev-1)*Nqk] - 5)^2
      end
    end
    Err = sqrt(Err / (Nqk * Nev))
    @test Err <= 2e-15
    @test Err1 <= 1e-16
  end
end
