# Load modules used here
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
using CLIMA.VTK
using CLIMA.IOstrings
using CLIMA.Atmos: vars_state, vars_aux
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using DelimitedFiles
using Random
using GPUifyLoops

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

# Random number seed
const seed = MersenneTwister(0)


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
  FT            = eltype(state)
  xvert::FT     = z
  #These constants are those used by Stevens et al. (2005)
  qref::FT      = 7.75e-3
  q_tot_sfc::FT = qref
  q_pt_sfc      = PhasePartition(q_tot_sfc)
  Rm_sfc        = gas_constant_air(q_pt_sfc)
  T_sfc::FT     = 292.5
  P_sfc::FT     = MSLP
  ρ_sfc::FT     = P_sfc / Rm_sfc / T_sfc
  # Specify moisture profiles 
  q_liq::FT      = 0
  q_ice::FT      = 0
  zb::FT         = 600    # initial cloud bottom
  zi::FT         = 840    # initial cloud top
  dz_cloud       = zi - zb
  q_liq_peak::FT = 0.00045 #cloud mixing ratio at z_i    
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

  # --------------------------------------------------
  # perturb initial state to break the symmetry and
  # trigger turbulent convection
  # --------------------------------------------------
  randnum1   = rand(seed, FT) / 100
  randnum2   = rand(seed, FT) / 100
  #randnum1   = rand(FT,1)/100
  #randnum2   = rand(FT,1)/100
  if xvert <= 200.0
    θ_liq += randnum1 * θ_liq 
    q_tot += randnum2 * q_tot
  end
  # --------------------------------------------------
  # END perturb initial state
  # --------------------------------------------------
    
  # Calculate PhasePartition object for vertical domain extent
  q_pt  = PhasePartition(q_tot, q_liq, q_ice) 
  #Pressure
  H     = Rm_sfc * T_sfc / grav;
  p     = P_sfc * exp(-xvert/H);
  #Density, Temperature
  TS    = LiquidIcePotTempSHumEquil_no_ρ(θ_liq, q_pt, p)
  ρ     = air_density(TS)
  T     = air_temperature(TS)
  #Assign State Variables
  u, v, w     = FT(7), FT(-5.5), FT(0)
  e_kin       = FT(1/2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(ρ*u, ρ*v, ρ*w) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
end


# TODO: temporary; move to new CLIMA module
# TODO: add an option to reduce communication: compute averages
# locally only
function gather_diagnostics(dg, Q, grid_resolution, OUTPATH)
  mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
  nranks = MPI.Comm_size(MPI.COMM_WORLD)

  grid = dg.grid
  topology = grid.topology
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dimensionality(grid) == 2 ? 1 : Nq
  nrealelem = length(topology.realelems)
  nvertelem = topology.stacksize
  nhorzelem = div(nrealelem, nvertelem)

  nstate = 6
  nthermo = 5
  host_array = Array ∈ typeof(Q).parameters
  localQ = host_array ? Q.realdata : Array(Q.realdata)
  thermoQ = zeros(Nq * Nq * Nqk, nthermo, nrealelem)
  vgeo = grid.vgeo
  localvgeo = host_array ? vgeo : Array(vgeo)

  for e in 1:nrealelem  
    for i in 1:Nq*Nq*Nqk
      z = localvgeo[i,grid.x3id,e]

      rho_node  = localQ[i,1,e]
      u_node    = localQ[i,2,e] / localQ[i,1,e]
      v_node    = localQ[i,3,e] / localQ[i,1,e]
      w_node    = localQ[i,4,e] / localQ[i,1,e]
      etot_node = localQ[i,5,e]/localQ[i,1,e]
      qt_node   = localQ[i,6,e]/localQ[i,1,e]

      e_int = etot_node - 1//2 * (u_node^2 + v_node^2 + w_node^2) - grav * z

      ts = PhaseEquil(e_int, qt_node, rho_node)
      Phpart = PhasePartition(ts)
      thermoQ[i,1,e] = Phpart.liq
      thermoQ[i,2,e] = Phpart.ice
      thermoQ[i,3,e] = qt_node-Phpart.liq-Phpart.ice
      thermoQ[i,4,e] = ts.T
      thermoQ[i,5,e] = liquid_ice_pottemp(ts)
    end
  end

  fluctT = zeros(Nq * Nq * Nqk, nthermo, nrealelem)    
  fluctQ = zeros(Nq * Nq * Nqk, nstate, nrealelem)
  varQ   = zeros(Nq * Nq * Nqk, nstate, nrealelem)

  rho_localtot = sum(localQ[:, 1, :])
  U_localtot   = sum(localQ[:, 2, :])
  V_localtot   = sum(localQ[:, 3, :])
  W_localtot   = sum(localQ[:, 4, :])
  e_localtot   = sum(localQ[:, 5, :])
  qt_localtot  = sum(localQ[:, 6, :])

  qliq_localtot  = sum(thermoQ[:, 1, :])
  qice_localtot  = sum(thermoQ[:, 2, :])
  qvap_localtot  = sum(thermoQ[:, 3, :])
  T_localtot     = sum(thermoQ[:, 4, :])
  theta_localtot = sum(thermoQ[:, 5, :])

  rho_tot   = MPI.Reduce(rho_localtot, +, 0, MPI.COMM_WORLD)
  U_tot     = MPI.Reduce(U_localtot, +, 0, MPI.COMM_WORLD)
  V_tot     = MPI.Reduce(V_localtot, +, 0, MPI.COMM_WORLD)
  W_tot     = MPI.Reduce(W_localtot, +, 0, MPI.COMM_WORLD)
  e_tot     = MPI.Reduce(e_localtot, +, 0, MPI.COMM_WORLD)
  qt_tot    = MPI.Reduce(qt_localtot, +, 0, MPI.COMM_WORLD)
  qliq_tot  = MPI.Reduce(qliq_localtot, +, 0, MPI.COMM_WORLD)
  qice_tot  = MPI.Reduce(qice_localtot, +, 0, MPI.COMM_WORLD)
  qvap_tot  = MPI.Reduce(qvap_localtot, +, 0, MPI.COMM_WORLD)
  T_tot     = MPI.Reduce(T_localtot, +, 0, MPI.COMM_WORLD)
  theta_tot = MPI.Reduce(theta_localtot, +, 0, MPI.COMM_WORLD)

#  if mpirank == 0
    rho_avg   = rho_tot / (size(localQ, 1) * size(localQ, 3) * nranks)
    U_avg     = (U_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    V_avg     = (V_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    W_avg     = (W_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    e_avg     = (e_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    qt_avg    = (qt_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    qliq_avg  = (qliq_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
    qice_avg  = (qice_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
    qvap_avg  = (qvap_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
    T_avg     = (T_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
    theta_avg = (theta_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
#=
    @debug "ρ average = $(rho_avg)"
    @debug "U average = $(U_avg)"
    @debug "V average = $(V_avg)"
    @debug "W average = $(W_avg)"
    @debug "e average = $(e_avg)"
    @debug "qt average = $(qt_avg)"
    @debug "qliq average = $(qliq_avg)"
    @debug "qice average = $(qice_avg)"
    @debug "qvap average = $(qvap_avg)"
    @debug "T average = $(T_avg)"
    @debug "theta average = $(theta_avg)"
=#
 # end

  AVG = SVector(rho_avg, U_avg, V_avg, W_avg, e_avg, qt_avg)
  AVG_T = SVector(qliq_avg, qice_avg, qvap_avg, T_avg, theta_avg)

  #fluctuations
  for s in 1:6  
    for e in 1:nrealelem
      for i in 1:Nq * Nqk * Nq
        if s == 1
          fluctQ[i,s,e] = localQ[i,s,e] - AVG[s]
          varQ[i,s,e]   = fluctQ[i,s,e]^2
          fluctT[i,s,e] = thermoQ[i,s,e] - AVG_T[s]
        elseif s < 6
          fluctQ[i,s,e] = localQ[i,s,e] / localQ[i,1,e] - AVG[s]
          varQ[i,s,e]   = fluctQ[i,s,e]^2
          fluctT[i,s,e] = thermoQ[i,s,e] - AVG_T[s]
        else
          fluctQ[i,s,e] = localQ[i,s,e] / localQ[i,1,e] - AVG[s]
          varQ[i,s,e]   = fluctQ[i,s,e]^2
        end
      end
    end
  end

  #standard_deviation
  rho_local_flucttot = sum(varQ[:,1,:])
  U_local_flucttot   = sum(varQ[:,2,:])
  V_local_flucttot   = sum(varQ[:,3,:])
  W_local_flucttot   = sum(varQ[:,4,:])
  e_local_flucttot   = sum(varQ[:,5,:])
  qt_local_flucttot  = sum(varQ[:,6,:])
#=
  rho_flucttot = MPI.Reduce(rho_local_flucttot, +, 0, MPI.COMM_WORLD)
  U_flucttot   = MPI.Reduce(U_local_flucttot, +, 0, MPI.COMM_WORLD)
  V_flucttot   = MPI.Reduce(V_local_flucttot, +, 0, MPI.COMM_WORLD)
  W_flucttot   = MPI.Reduce(W_local_flucttot, +, 0, MPI.COMM_WORLD)
  e_flucttot   = MPI.Reduce(e_local_flucttot, +, 0, MPI.COMM_WORLD)
  qt_flucttot  = MPI.Reduce(qt_local_flucttot, +, 0, MPI.COMM_WORLD)

  if mpirank == 0
    rho_standard_dev = (rho_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
    U_standard_dev   = (U_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
    V_standard_dev   = (V_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
    W_standard_dev   = (W_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
    e_standard_dev   = (e_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
    qt_standard_dev  = (qt_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
  
    #Variance
    global_variance_rho = rho_standard_dev^2
    global_variance_U = U_standard_dev^2
    global_variance_V = V_standard_dev^2
    global_variance_W = W_standard_dev^2
    global_variance_e = e_standard_dev^2
    global_variance_qt = qt_standard_dev^2
  end
=#
#=
 @debug "ρ standard_deviation = $(rho_standard_dev)" 
 @debug "ρ Variance = $(Global_Variance_rho)"
 @debug "U standard_deviation = $(U_standard_dev)"
 @debug "U Variance = $(Global_Variance_U)"
 @debug "V standard_deviation = $(V_standard_dev)"
 @debug "V Variance = $(Global_Variance_V)"
 @debug "W standard_deviation = $(W_standard_dev)"
 @debug "W Variance = $(Global_Variance_W)"
 @debug "e standard_deviation = $(e_standard_dev)"
 @debug "e Variance = $(Global_Variance_e)"
 @debug "qt standard_deviation = $(qt_standard_dev)"
@debug "qt Variance = $(Global_Variance_qt)"
=#
  #Horizontal averages we might need
  S = zeros(Nqk, nvertelem,11)
  for eh in 1:nhorzelem
    for ev in 1:nvertelem
      e = ev + (eh - 1) * nvertelem

      for k in 1:Nqk
        for j in 1:Nq
          for i in 1:Nq
            ijk = i + Nq * ((j-1) + Nq * (k-1)) 
            S[k,ev,1] += fluctQ[ijk,4,e] * fluctT[ijk,5,e]
            S[k,ev,2] += fluctQ[ijk,4,e] * fluctT[ijk,3,e]
            S[k,ev,3] += fluctQ[ijk,4,e] * fluctQ[ijk,2,e]
            S[k,ev,4] += fluctQ[ijk,4,e] * fluctQ[ijk,3,e]
            S[k,ev,5] += fluctQ[ijk,4,e] * fluctQ[ijk,4,e]
            S[k,ev,6] += fluctQ[ijk,4,e] * fluctQ[ijk,1,e]
            S[k,ev,7] += thermoQ[ijk,1,e]
            S[k,ev,8] += fluctQ[ijk,4,e] * fluctT[ijk,1,e]
            S[k,ev,9] += fluctQ[ijk,4,e] * fluctQ[ijk,4,e] * fluctQ[ijk,4,e]
            S[k,ev,10] += fluctQ[ijk,2,e] * fluctQ[ijk,2,e]
            S[k,ev,11] += fluctQ[ijk,3,e] * fluctQ[ijk,3,e]
          end
        end
      end
    end
  end

  S_avg = zeros(Nqk,nvertelem,11)
  for s in 1:11
    for ev in 1:nvertelem
      for k in 1:Nqk
        S_avg[k,ev,s] = MPI.Reduce(S[k,ev,s], +, 0, MPI.COMM_WORLD)
     
        if mpirank == 0
          S_avg[k,ev,s] = S_avg[k,ev,s] / (Nq * Nq * nhorzelem * nranks)
        end
      end
    end
  end

  OutputHF = zeros(nvertelem * Nqk)
  OutputWQVAP = zeros(nvertelem * Nqk)
  OutputWU = zeros(nvertelem * Nqk)
  OutputWV = zeros(nvertelem * Nqk)
  OutputWW = zeros(nvertelem * Nqk)
  OutputWRHO = zeros(nvertelem * Nqk)
  OutputQLIQ = zeros(nvertelem * Nqk)
  OutputWQLIQ = zeros(nvertelem * Nqk)
  OutputWWW = zeros(nvertelem * Nqk )
  OutputUU = zeros(nvertelem * Nqk )
  OutputVV = zeros(nvertelem * Nqk )

  for ev in 1:nvertelem
    for k in 1:Nqk
      i=k + Nqk * (ev - 1)
      OutputHF[i] = S_avg[k,ev,1]
      OutputWQVAP[i] = S_avg[k,ev,2]
      OutputWU[i] = S_avg[k,ev,3]
      OutputWV[i] = S_avg[k,ev,4]
      OutputWW[i] = S_avg[k,ev,5]
      OutputWRHO[i] = S_avg[k,ev,6]
      OutputQLIQ[i] = S_avg[k,ev,7]
      OutputWQLIQ[i] = S_avg[k,ev,8]
      OutputWWW[i] = S_avg[k,ev,9]
      OutputUU[i] = S_avg[k,ev,10]
      OutputVV[i] = S_avg[k,ev,11]
    end
  end


    Δx, Δy, Δz = grid_resolution[1], grid_resolution[2], grid_resolution[end]
                
    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_HF.txt")
    io = open(fileout, "a")
      writedlm(io, OutputHF)
    close(io)

    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_WQVAP.txt")
    io = open(fileout, "a")
         writedlm(io, OutputWQVAP)
    close(io)

    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_UU.txt")
    io = open(fileout, "a")
      writedlm(io, OutputUU)
    close(io)

    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_VV.txt")
    io = open(fileout, "a")
      writedlm(io, OutputVV)
    close(io)

    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_WW.txt")
    io = open(fileout, "a")
      writedlm(io, OutputWW)
    close(io)

#=        
    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_WU.txt")
    io = open(fileout, "a")
      writedlm(io, OutputWU)
    close(io)

    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_WV.txt")
    io = open(fileout, "a")
      writedlm(io, OutputWV)
    close(io)

    fileout = string(OUTPATH, "/dx", Δx, "mXdy", Δy, "mXdz", Δz, "_WRHO.txt")
    io = open(fileout, "a")
      writedlm(io, OutputWRHO)
    close(io)

=#
end

function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, grid_resolution, domain_size, zmax, zsponge, problem_name, OUTPATH)
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
  u_geostrophic = FT(7)
  v_geostrophic = FT(-5.5)
  
  # Model definition
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{FT}(C_smag),
                     EquilMoist(),
                     StevensRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1),
                     (Gravity(), 
                      RayleighSponge{FT}(zmax, zsponge, 1), 
                      Subsidence(), 
                      GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)), 
                     DYCOMS_BC{FT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)
  # Balancelaw description
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())
  Q = init_ode_state(dg, FT(0); device=CPU())
    
  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
  # Calculating initial condition norm 
#=  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0
=#
  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
    #=  energy = norm(Q) =#
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     """, ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"))
    end
  end
  
  # Setup VTK output callbacks
  step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(5000) do (init=false)
    mkpath(OUTPATH)
    outprefix = @sprintf("%s/dycoms_%dD_mpirank%04d_step%04d", OUTPATH, dim,
                           MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)), 
             dg.auxstate, flattenednames(vars_aux(model,FT)))
        
    step[1] += 1
    nothing
  end
    
  
  #Get statistics during run:
  cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(10000) do (init=false)
    gather_diagnostics(dg, Q, grid_resolution, OUTPATH)
  end
    
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk, cbdiagnostics))

  #Get statistics at the end of the run:
  gather_diagnostics(dg, Q, grid_resolution, OUTPATH)

    
  # Print some end of the simulation information
#=  engf = norm(Q)
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
=#
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
 #   @testset "$(@__FILE__)" for ArrayType in ArrayTypes
  for ArrayType in ArrayTypes
    # Problem type
    FT = Float32
    # DG polynomial order 
    N = 4
    # SGS Filter constants
    C_smag = FT(0.15)
    LHF    = FT(115)
    SHF    = FT(15)
    C_drag = FT(0.0011)
    # User defined domain parameters
    Δx, Δy, Δz = 50, 50, 10
    xmin, xmax = 0, 1500
    ymin, ymax = 0, 1500
    zmin, zmax = 0, 1500

    grid_resolution = [Δx, Δy, Δz]
    domain_size     = [xmin, xmax, ymin, ymax, zmin, zmax]
    dim = length(grid_resolution)

    if(dim == 2)
        brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                      grid1d(ymin, ymax, elemsize=FT(grid_resolution[end])*N))
    elseif (dim == 3)
        brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                      grid1d(ymin, ymax, elemsize=FT(grid_resolution[2])*N),
                      grid1d(zmin, zmax, elemsize=FT(grid_resolution[end])*N))
    end
    zmax = brickrange[dim][end]
    zsponge = FT(0.75 * zmax)
    
    topl = StackedBrickTopology(mpicomm, brickrange,
                                periodicity = (true, true, false),
                                boundary=((0,0),(0,0),(1,2)))

    problem_name = "dycoms_IOstrings"
    dt = 0.002
    timeend = 14400

    #Create unique output path directory:
    OUTPATH = IOstrings_outpath_name(problem_name, grid_resolution)
      
    @info (ArrayType, dt, FT, dim)
    result = run(mpicomm, ArrayType, dim, topl, 
                 N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, grid_resolution, domain_size, zmax, zsponge, problem_name, OUTPATH)

  end
end

#nothing
