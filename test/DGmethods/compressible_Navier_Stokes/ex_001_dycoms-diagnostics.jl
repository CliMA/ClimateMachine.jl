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
using DelimitedFiles

using CLIMA.Atmos: vars_state, vars_aux

# TODO: for diagnostics; move to new CLIMA module
using MPI

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
  DT         = eltype(state)
  xvert::DT  = z
  
  epsdv::DT     = molmass_ratio
  q_tot_sfc::DT = 8.1e-3
  Rm_sfc::DT    = gas_constant_air(PhasePartition(q_tot_sfc))
  ρ_sfc::DT     = 1.22
  P_sfc::DT     = 1.0178e5
  T_BL::DT      = 285.0
  T_sfc::DT     = P_sfc/(ρ_sfc * Rm_sfc);
  
  q_liq::DT      = 0
  q_ice::DT      = 0
  zb::DT         = 600   
  zi::DT         = 840 
  dz_cloud       = zi - zb
  q_liq_peak::DT = 4.5e-4
  
  if xvert > zb && xvert <= zi        
    q_liq = (xvert - zb)*q_liq_peak/dz_cloud
  end
  if ( xvert <= zi)
    θ_liq  = DT(289)
    q_tot  = DT(8.1e-3)
  else
    θ_liq = DT(297.5) + (xvert - zi)^(DT(1/3))
    q_tot = DT(1.5e-3)
  end

  q_pt = PhasePartition(q_tot, q_liq, DT(0))
  Rm    = gas_constant_air(q_pt)
  cpm   = cp_m(q_pt)
  #Pressure
  H = Rm_sfc * T_BL / grav;
  P = P_sfc * exp(-xvert/H);
  #Exner
  exner_dry = exner(P, PhasePartition(DT(0)))
  #Temperature 
  T             = exner_dry*θ_liq + LH_v0*q_liq/(cpm*exner_dry);
  #Density
  ρ             = P/(Rm*T);
  #Potential Temperature
  θv     = virtual_pottemp(T, P, q_pt)
  # energy definitions
  u, v, w     = DT(7), DT(-5.5), DT(0)
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = DT(1//2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(U, V, W) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
end   

# TODO: temporary; move to new CLIMA module
function gather_diags(dg, Q)
  N=polynomialorder(dg.grid)
  Nq=N+1
  Nqk=dimensionality(dg.grid) == 2 ? 1 : Nq
  grid = dg.grid
  topology = grid.topology
  nstate = 6
  nthermo = 5
  nrealelems = length(topology.realelems)
  nvertelems = topology.stacksize
  nhorzelems = div(nrealelems, nvertelems)
  host_array = Array ∈ typeof(Q).parameters
  localQ = host_array ? Q.realQ : Array(Q.realQ)
  thermoQ = zeros(Nq*Nq*Nqk,nthermo,nrealelems)
  vgeo = grid.vgeo
  h_vgeo = host_array ? vgeo : Array(vgeo)
  Xid = grid.x3id
  for e in 1:nrealelems	
	for i in 1:Nq*Nqk*Nq
		z =h_vgeo[i,Xid,e]
  		rho_node = localQ[i,1,e]
		u_node = localQ[i,2,e]
		w_node = localQ[i,4,e]
		etot_node = localQ[i,5,e]
		qt_node = localQ[i,6,e]
		e_int = etot_node-1//2*(u_node^2+w_node^2)-grav*z
		ts = PhaseEquil(e_int, qt_node, rho_node)
		Phpart = PhasePartition(ts)
		thermoQ[i,1,e] = Phpart.liq
		thermoQ[i,2,e] = Phpart.ice
		thermoQ[i,3,e] = qt_node-Phpart.liq-Phpart.ice
		thermoQ[i,4,e] = ts.T
		thermoQ[i,5,e] = liquid_ice_pottemp(ts)
	end
  end
  fluctT = zeros(Nq*Nq*Nqk,nthermo,nrealelems)		
  fluctQ = zeros(Nq*Nq*Nqk,nstate,nrealelems)
  VarQ = zeros(Nq*Nq*Nqk,nstate,nrealelems)
  rho_localtot = sum(localQ[:, 1, :])
  U_localtot = sum(localQ[:, 2, :])
  V_localtot = sum(localQ[:, 3, :])
  W_localtot = sum(localQ[:, 4, :])
  e_localtot = sum(localQ[:, 5, :])
  qt_localtot = sum(localQ[:, 6, :])
  qliq_localtot = sum(thermoQ[:, 1, :])
  qice_localtot = sum(thermoQ[:, 2, :])
  qvap_localtot = sum(thermoQ[:, 3, :])
  T_localtot = sum(thermoQ[:, 4, :])
  theta_localtot = sum(thermoQ[:, 5, :])
  mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
  nranks = MPI.Comm_size(MPI.COMM_WORLD)
  rho_tot = MPI.Reduce(rho_localtot, +, 0, MPI.COMM_WORLD)
  U_tot = MPI.Reduce(U_localtot, +, 0, MPI.COMM_WORLD)
  V_tot = MPI.Reduce(V_localtot, +, 0, MPI.COMM_WORLD)
  W_tot = MPI.Reduce(W_localtot, +, 0, MPI.COMM_WORLD)
  e_tot = MPI.Reduce(e_localtot, +, 0, MPI.COMM_WORLD)
  qt_tot = MPI.Reduce(qt_localtot, +, 0, MPI.COMM_WORLD)
  qliq_tot =MPI.Reduce(qliq_localtot, +, 0, MPI.COMM_WORLD)
  qice_tot =MPI.Reduce(qice_localtot, +, 0, MPI.COMM_WORLD)
  qvap_tot =MPI.Reduce(qvap_localtot, +, 0, MPI.COMM_WORLD)
  T_tot =MPI.Reduce(T_localtot, +, 0, MPI.COMM_WORLD)
  theta_tot =MPI.Reduce(theta_localtot, +, 0, MPI.COMM_WORLD)
  if mpirank == 0
    rho_avg = rho_tot / (size(localQ, 1) * size(localQ, 3) * nranks)
    U_avg = (U_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    V_avg = (V_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    W_avg = (W_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    e_avg = (e_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    qt_avg = (qt_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    qliq_avg = (qliq_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    qice_avg = (qice_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    qvap_avg = (qvap_tot / (size(localQ, 1) * size(localQ, 3) * nranks))/rho_avg
    T_avg = (T_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
    theta_avg = (theta_tot / (size(localQ, 1) * size(localQ, 3) * nranks))
    @info "ρ average = $(rho_avg)"
    @info "U average = $(U_avg)"
    @info "V average = $(V_avg)"
    @info "W average = $(W_avg)"
    @info "e average = $(e_avg)"
    @info "qt average = $(qt_avg)"
    @info "qliq average = $(qliq_avg)"
    @info "qice average = $(qice_avg)"
    @info "qvap average = $(qvap_avg)"
    @info "T average = $(T_avg)"
    @info "theta average = $(theta_avg)"
  end
  AVG=SVector(rho_avg,U_avg,V_avg,W_avg,e_avg,qt_avg)
  AVG_T=SVector(qliq_avg,qice_avg,qvap_avg,T_avg,theta_avg)
  #fluctuations
  for s in 1:6	
	for e in 1:nrealelems
		for i in 1:Nq*Nqk*Nq
			if s==1
				fluctQ[i,s,e] = localQ[i,s,e]-AVG[s]
				VarQ[i,s,e]=fluctQ[i,s,e]^2
				fluctT[i,s,e] = thermoQ[i,s,e]/localQ[i,s,e]-AVG_T[s]
			elseif s<6
				fluctQ[i,s,e] = localQ[i,s,e]/localQ[i,1,e]-AVG[s]
                                VarQ[i,s,e]=fluctQ[i,s,e]^2
				fluctT[i,s,e] = thermoQ[i,s,e]/localQ[i,s,e]-AVG_T[s]
			else
				fluctQ[i,s,e] = localQ[i,s,e]/localQ[i,1,e]-AVG[s]
                                VarQ[i,s,e]=fluctQ[i,s,e]^2
			end
		end
	end
  end
  #standard_deviation
  rho_local_flucttot = sum(VarQ[:,1,:])
  rho_flucttot = MPI.Reduce(rho_local_flucttot, +, 0, MPI.COMM_WORLD)
  U_local_flucttot = sum(VarQ[:,2,:])
  U_flucttot = MPI.Reduce(U_local_flucttot, +, 0, MPI.COMM_WORLD)
  V_local_flucttot = sum(VarQ[:,3,:])
  V_flucttot = MPI.Reduce(V_local_flucttot, +, 0, MPI.COMM_WORLD)
  W_local_flucttot = sum(VarQ[:,4,:])
  W_flucttot = MPI.Reduce(W_local_flucttot, +, 0, MPI.COMM_WORLD)
  e_local_flucttot = sum(VarQ[:,5,:])
  e_flucttot = MPI.Reduce(e_local_flucttot, +, 0, MPI.COMM_WORLD)
  qt_local_flucttot = sum(VarQ[:,6,:])
  qt_flucttot = MPI.Reduce(qt_local_flucttot, +, 0, MPI.COMM_WORLD)
  if mpirank == 0
  	rho_standard_dev = (rho_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
	U_standard_dev = (U_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
	V_standard_dev = (V_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
	W_standard_dev = (W_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
	e_standard_dev = (e_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
	qt_standard_dev = (qt_flucttot / (size(fluctQ, 1) * size(fluctQ, 3) * nranks) )^(0.5)
  #Variance
	Global_Variance_rho = rho_standard_dev^2
	Global_Variance_U = U_standard_dev^2
	Global_Variance_V = V_standard_dev^2
	Global_Variance_W = W_standard_dev^2
	Global_Variance_e = e_standard_dev^2
	Global_Variance_qt = qt_standard_dev^2
  end
 @info "ρ standard_deviation = $(rho_standard_dev)" 
 @info "ρ Variance = $(Global_Variance_rho)"
 @info "U standard_deviation = $(U_standard_dev)"
 @info "U Variance = $(Global_Variance_U)"
 @info "V standard_deviation = $(V_standard_dev)"
 @info "V Variance = $(Global_Variance_V)"
 @info "W standard_deviation = $(W_standard_dev)"
 @info "W Variance = $(Global_Variance_W)"
 @info "e standard_deviation = $(e_standard_dev)"
 @info "e Variance = $(Global_Variance_e)"
 @info "qt standard_deviation = $(qt_standard_dev)"
 @info "qt Variance = $(Global_Variance_qt)"
 S = zeros(Nqk, nvertelems,6)
for eh in 1:nhorzelems
  for ev in 1:nvertelems
    e = ev + (eh - 1) * nvertelems

    for k in 1:Nqk
      for j in 1:Nq
        for i in 1:Nq
          ijk = i + Nq * ((j-1) + Nq * (k-1)) 
          S[k,ev,1] += fluctQ[ijk, 4, e]*fluctT[ijk,5,e]
	  S[k,ev,2] += fluctQ[ijk, 4, e]*fluctT[ijk,3,e]
	  S[k,ev,3] += fluctQ[ijk, 4, e]*fluctQ[ijk,2,e]
	  S[k,ev,4] += fluctQ[ijk, 4, e]*fluctQ[ijk,3,e]
	  S[k,ev,5] += fluctQ[ijk, 4, e]*fluctQ[ijk,4,e]
	  S[k,ev,6] += fluctQ[ijk, 4, e]*fluctQ[ijk,1,e]
        end
      end
    end
  end
end
S_avg=zeros(Nqk,nvertelems,6)
for s in 1:6
 for ev in 1:nvertelems
   for k in 1:Nqk
     S_avg[k,ev,s]=MPI.Reduce(S[k,ev,s], +, 0, MPI.COMM_WORLD)
     if mpirank == 0
       S_avg[k,ev,s] = S_avg[k,ev,s]/(Nq*Nq*nhorzelems*nranks)
     end
   end
 end
end
OutputHF = zeros(nvertelems*Nqk)
OutputWQVAP = zeros(nvertelems*Nqk)
OutputWU = zeros(nvertelems*Nqk)
OutputWV = zeros(nvertelems*Nqk)
OutputWW = zeros(nvertelems*Nqk)
OutputWRHO = zeros(nvertelems*Nqk)
for ev in 1:nvertelems
	for k in 1:Nqk
		i=k+Nqk*(ev-1)
		OutputHF[i] = S_avg[k,ev,1]
		OutputWQVAP[i] = S_avg[k,ev,2]
		OutputWU[i] = S_avg[k,ev,3]
		OutputWV[i] = S_avg[k,ev,4]
		OutputWW[i] = S_avg[k,ev,5]
		OutputWRHO[i] = S_avg[k,ev,6]
	end
end
open("/home/yassine/yt-2T-drive/HF.txt", "w") do f
writedlm("/home/yassine/yt-2T-drive/HF.txt", OutputHF)
end
open("/home/yassine/yt-2T-drive/WQVAP.txt", "w") do f
writedlm("/home/yassine/yt-2T-drive/WQVAP.txt", OutputWQVAP)
end
open("/home/yassine/yt-2T-drive/WU.txt", "w") do f
writedlm("/home/yassine/yt-2T-drive/WU.txt", OutputWU)
end
open("/home/yassine/yt-2T-drive/WV.txt", "w") do f
writedlm("/home/yassine/yt-2T-drive/WV.txt", OutputWV)
end
open("/home/yassine/yt-2T-drive/WW.txt", "w") do f
writedlm("/home/yassine/yt-2T-drive/WW.txt", OutputWW)
end
open("/home/yassine/yt-2T-drive/WRHO.txt", "w") do f
writedlm("/home/yassine/yt-2T-drive/WRHO.txt", OutputWRHO)
end
end

function run(mpicomm, ArrayType, dim, topl, N, timeend, DT, dt, C_smag, LHF, SHF, C_drag, zmax, zsponge)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N)
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{DT}(C_smag),
                     EquilMoist(),
                     StevensRadiation{DT}(85, 1, 840, 1.22, 3.75e-6, 70, 22),
                     (Gravity(), 
                      RayleighSponge{DT}(zmax, zsponge, 1), 
                      Subsidence(), 
                      GeostrophicForcing{DT}(7.62e-5, 7, -5.5)), 
                     DYCOMS_BC{DT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  Q = init_ode_state(dg, DT(0))

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
    writevtk(outprefix, Q, dg, flattenednames(vars_state(model,DT)), 
             dg.auxstate, flattenednames(vars_aux(model,DT)))
        
    step[1] += 1
    nothing
  end

  cbdiags = GenericCallbacks.EveryXSimulationSteps(300000) do (init=false)
    gather_diags(dg, Q)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk, cbdiags))

  gather_diags(dg, Q)

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, DT(timeend))

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
    DT = Float64
    # DG polynomial order 
    N = 4
    # SGS Filter constants
    C_smag = DT(0.15)
    LHF    = DT(115)
    SHF    = DT(15)
    C_drag = DT(0.0011)
    # User defined domain parameters
    brickrange = (grid1d(0, 2000, elemsize=DT(50)*N),
                  grid1d(0, 2000, elemsize=DT(50)*N),
                  grid1d(0, 1500, elemsize=DT(20)*N))
    zmax = brickrange[3][end]
    zsponge = DT(0.75 * zmax)
    
    topl = StackedBrickTopology(mpicomm, brickrange,
                                periodicity = (true, true, false),
                                boundary=((0,0),(0,0),(1,2)))
    dt = 0.02
    timeend = 14400
    dim = 3
    @info (ArrayType, DT, dim)
    result = run(mpicomm, ArrayType, dim, topl, 
                 N, timeend, DT, dt, C_smag, LHF, SHF, C_drag, zmax, zsponge)
    @test result ≈ DT(0.9999737848359238)
  end
end

#nothing
