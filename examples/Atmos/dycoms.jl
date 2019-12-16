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
using CLIMA.TicToc
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK

using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                   CentralNumericalFluxDiffusive,
                                   CentralNumericalFluxNonDiffusive
import CLIMA.DGmethods.NumericalFluxes: update_penalty!, numerical_flux_diffusive!,
                                    NumericalFluxNonDiffusive
import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                    vars_diffusive, vars_integrals, flux_nondiffusive!,
                    flux_diffusive!, source!, wavespeed,
                    boundary_state!, update_aux!,
                    gradvariables!, init_aux!, init_state!,
                    LocalGeometry, indefinite_stack_integral!,
                    reverse_indefinite_stack_integral!, integrate_aux!, num_integrals,
                    DGModel, nodal_update_aux!, diffusive!,
                    copy_stack_field_down!, create_state

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# ------------------- Boundary Conditions -------------------------- #
function atmos_boundary_state!(::Rusanov, f::Function, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  f(stateP, auxP, nM, stateM, auxM, bctype, t)
end

function atmos_boundary_state!(::CentralNumericalFluxDiffusive, f::Function,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  f(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

# lookup boundary condition by face
function atmos_boundary_state!(nf::Rusanov, bctup::Tuple, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, stateP, auxP, nM, stateM, auxM,
                        bctype, t)
end

function atmos_boundary_state!(nf::CentralNumericalFluxDiffusive,
                               bctup::Tuple, m::AtmosModel, stateP::Vars,
                               diffP::Vars, auxP::Vars, nM, stateM::Vars,
                               diffM::Vars, auxM::Vars, bctype, t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, stateP, diffP, auxP, nM, stateM,
                        diffM, auxM, bctype, t)
end

"""
  DYCOMS_BC
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
"""
struct DYCOMS_BC{FT} 
  C_drag::FT
  LHF::FT
  SHF::FT
end
function atmos_boundary_state!(::Rusanov, bc::DYCOMS_BC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, state1::Vars, aux1::Vars)
  # stateM is the 𝐘⁻ state while stateP is the 𝐘⁺ state at an interface.
  # at the boundaries the ⁻, minus side states are the interior values
  # state1 is 𝐘 at the first interior nodes relative to the bottom wall
  FT = eltype(stateP)
  # Get values from minus-side state
  ρM = stateM.ρ
  UM, VM, WM = stateM.ρu
  EM = stateM.ρe
  QTM = stateM.moisture.ρq_tot
  uM, vM, wM  = UM/ρM, VM/ρM, WM/ρM
  q_totM = QTM/ρM
  UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM

  # Assign reflection wall boundaries (top wall)
  stateP.ρu = SVector(UM - 2 * nM[1] * UnM,
                      VM - 2 * nM[2] * UnM,
                      WM - 2 * nM[3] * UnM)

  # Assign scalar values at the boundaries
  stateP.ρ = ρM
  stateP.moisture.ρq_tot = QTM

end
function atmos_boundary_state!(::CentralNumericalFluxDiffusive, bc::DYCOMS_BC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, state1::Vars, diff1::Vars,
                               aux1::Vars)
  # stateM is the 𝐘⁻ state while stateP is the 𝐘⁺ state at an interface.
  # at the boundaries the ⁻, minus side states are the interior values
  # state1 is 𝐘 at the first interior nodes relative to the bottom wall
  FT = eltype(stateP)
  # Get values from minus-side state
  ρM = stateM.ρ
  UM, VM, WM = stateM.ρu
  EM = stateM.ρe
  QTM = stateM.moisture.ρq_tot
  uM, vM, wM  = UM/ρM, VM/ρM, WM/ρM
  q_totM = QTM/ρM
  UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM

  # Assign reflection wall boundaries (top wall)
  stateP.ρu = SVector(UM - 2 * nM[1] * UnM,
                      VM - 2 * nM[2] * UnM,
                      WM - 2 * nM[3] * UnM)

  # Assign scalar values at the boundaries
  stateP.ρ = ρM
  stateP.moisture.ρq_tot = QTM
  # Assign diffusive fluxes at boundaries
  diffP = diffM
  xvert = auxM.coord[3]

  if bctype == 1 # bctype identifies bottom wall
    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    z_FN             = aux1.coord[3]
    ρ_FN             = state1.ρ
    U_FN, V_FN, W_FN = state1.ρu
    E_FN             = state1.ρe
    u_FN, v_FN, w_FN = U_FN/ρ_FN, V_FN/ρ_FN, W_FN/ρ_FN
    windspeed_FN     = sqrt(u_FN^2 + v_FN^2 + w_FN^2)
    q_tot_FN         = state1.moisture.ρq_tot / ρ_FN
    e_int_FN         = E_FN/ρ_FN - windspeed_FN^2/2 - grav*z_FN
    TS_FN            = PhaseEquil(e_int_FN, ρ_FN, q_tot_FN)
    T_FN             = air_temperature(TS_FN)
    q_vap_FN         = q_tot_FN - PhasePartition(TS_FN).liq
    # --------------------------
    # Bottom boundary quantities
    # --------------------------
    zM          = auxM.coord[3]
    q_totM      = QTM/ρM
    windspeed   = sqrt(uM^2 + vM^2 + wM^2)
    e_intM      = EM/ρM - windspeed^2/2 - grav*zM
    TSM         = PhaseEquil(e_intM, ρM, q_totM)
    q_vapM      = q_totM - PhasePartition(TSM).liq
    TM          = air_temperature(TSM)
    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    ρτM = diffM.ρτ
    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is replaced by the appropriate flux terms
    C_drag = bc.C_drag
    ρτ13P  = -ρM * C_drag * windspeed_FN * u_FN
    ρτ23P  = -ρM * C_drag * windspeed_FN * v_FN
    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)
    diffP.ρτ = SHermitianCompact{3,FT,6}(SVector(FT(0),ρτM[2,1],ρτ13P, FT(0), ρτ23P,FT(0)))
    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    diffP.moisture.ρd_q_tot  = SVector(FT(0),
                                       FT(0),
                                       bc.LHF/(LH_v0))
    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms)
    diffP.ρd_h_tot  = SVector(FT(0),
                              FT(0),
                              bc.LHF + bc.SHF)
  end
end
boundary_state!(nf, m::AtmosModel, x...) =
  atmos_boundary_state!(nf, m.boundarycondition, m, x...)
boundary_state!(::CentralGradPenalty, bl::AtmosModel, _...) = nothing

# -------------------- Radiation Model -------------------------- # 
vars_state(::RadiationModel, FT) = @vars()
vars_aux(::RadiationModel, FT) = @vars()
vars_integrals(::RadiationModel, FT) = @vars()

function atmos_nodal_update_aux!(::RadiationModel, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function preodefun!(::RadiationModel, aux::Vars, state::Vars, t::Real) end
function integrate_aux!(::RadiationModel, integ::Vars, state::Vars, aux::Vars) end
function flux_radiation!(::RadiationModel, flux::Grad, state::Vars, aux::Vars, t::Real) end
"""
  DYCOMSRadiation

Stevens et. al (2005) version of the δ-four stream model used to represent radiative transfer. 
Analytical description as a function of the liquid water path and inversion height zᵢ
"""
struct DYCOMSRadiation{FT}  <: RadiationModel
  "κ [m^2/s] "
  κ::FT
  "α_z Troposphere cooling parameter [m^(-4/3)]"
  α_z::FT
  "z_i Inversion height [m]"
  z_i::FT
  "ρ_i Density"
  ρ_i::FT
  "D_subsidence Large scale divergence [s^(-1)]"
  D_subsidence::FT
  "F₀ Radiative flux parameter [W/m^2]"
  F_0::FT
  "F₁ Radiative flux parameter [W/m^2]"
  F_1::FT
end
vars_integrals(m::DYCOMSRadiation, FT) = @vars(∂κLWP::FT)
function integrate_aux!(m::DYCOMSRadiation, integrand::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  integrand.radiation.∂κLWP = state.ρ * m.κ * aux.moisture.q_liq
end
function flux_radiation!(m::DYCOMSRadiation, flux::Grad, state::Vars,
                         aux::Vars, t::Real)
  FT = eltype(flux)
  z = aux.orientation.Φ/grav
  Δz_i = max(z - m.z_i, -zero(FT))
  # Constants
  cloud_top_cooling = m.F_0 * exp(-aux.∫dnz.radiation.∂κLWP)
  cloud_base_warming = m.F_1 * exp(-aux.∫dz.radiation.∂κLWP)
  free_troposphere_cooling = m.ρ_i * FT(cp_d) * m.D_subsidence * m.α_z * ((cbrt(Δz_i))^4 / 4 + m.z_i * cbrt(Δz_i))
  F_rad = cloud_top_cooling + cloud_base_warming + free_troposphere_cooling
  flux.ρe += SVector(FT(0), 
                     FT(0), 
                     F_rad)
end
function preodefun!(m::DYCOMSRadiation, aux::Vars, state::Vars, t::Real) end

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
  exner_dry = exner_given_pressure(P, PhasePartition(FT(0)))
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


function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, zmax, zsponge)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{FT}(C_smag),
                     EquilMoist(),
                     DYCOMSRadiation{FT}(85, 1, 840, 1.22, 3.75e-6, 70, 22),
                     (Gravity(),
                      RayleighSponge{FT}(zmax, zsponge, 1),
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

  @tic solve
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))
  @toc solve

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
  tictoc()
  @tic dycoms
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @testset begin
    # Problem type
    FloatType = (Float32, Float64)
    for FT in FloatType
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
      zsponge = FT(0.75 * zmax)

      topl = StackedBrickTopology(mpicomm, brickrange,
                                  periodicity = (true, true, false),
                                  boundary=((0,0),(0,0),(1,2)))
      dt = 0.01
      timeend = 100
      dim = 3
      @info (ArrayType, FT, dim)
      result = run(mpicomm, ArrayType, dim, topl,
                   N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, zmax, zsponge)
      @test result ≈ FT(1.0002404228252337) atol=1e-4
    end
  end
  @toc dycoms
end

#nothing
