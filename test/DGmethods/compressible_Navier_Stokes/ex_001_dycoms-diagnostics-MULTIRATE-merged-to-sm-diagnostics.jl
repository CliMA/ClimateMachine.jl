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
using CLIMA.AdditiveRungeKuttaMethods
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
using DelimitedFiles
using GPUifyLoops
using Random
using CLIMA.IOstrings
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
  ziplus::FT     = 875
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
  #randnum1   = rand(seed, FT) / 100
  #randnum2   = rand(seed, FT) / 1000
  #randnum1   = rand(Uniform(-0.02,0.02), 1, 1)
  #randnum2   = rand(Uniform(-0.000015,0.000015), 1, 1)
  #=if xvert <= 25.0    
    θ_liq += randnum1 * θ_liq 
    #q_tot += randnum2 * q_tot      
  end=#
  # --------------------------------------------------
  # END perturb initial state
  # --------------------------------------------------

  # Calculate PhasePartition object for vertical domain extent
  q_pt  = PhasePartition(q_tot, q_liq, q_ice) 
  #Pressure
  H     = Rm_sfc * T_sfc / grav;
  p     = P_sfc * exp(-xvert/H);
  #Density, Temperature
  #TS    = LiquidIcePotTempSHumEquil_no_ρ(θ_liq, q_pt, p)
  TS    = LiquidIcePotTempSHumNonEquil_no_ρ(θ_liq, q_pt, p)
  ρ     = air_density(TS)
  T     = air_temperature(TS)

  #Assign State Variables
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
  e_kin       = FT(1/2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(ρ*u, ρ*v, ρ*w) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot

  #writing initialized thermodynamic quantities
  q_vap=q_tot-q_liq-q_ice
  p   = air_pressure(T,ρ,q_pt)
  θ   = dry_pottemp(T,p,q_pt)
  θ_v = virtual_pottemp(T,p,q_pt)
  if x == 0 && y == 0
    io = open("./output/ICs.dat", "a")
      writedlm(io, [z state.ρ θ θ_v θ_liq q_tot q_liq q_vap])
    close(io)
  end
    
end


# TODO: temporary; move to new CLIMA module
# TODO: add an option to reduce communication: compute averages
# locally only
function gather_diagnostics(dg, Q, grid_resolution, current_time_string, diagnostics_fileout,κ,LWP_fileout)
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
  aux1=dg.auxstate
  nstate = 6
  nthermo = 7
  host_array = Array ∈ typeof(Q).parameters
  localQ = host_array ? Q.realdata : Array(Q.realdata)
  host_array1 = Array ∈ typeof(dg.auxstate).parameters
  localaux = host_array1 ? aux1.realdata : Array(aux1.realdata)
  thermoQ = zeros(Nq * Nq * Nqk, nthermo, nrealelem)
  vgeo = grid.vgeo
  localvgeo = host_array ? vgeo : Array(vgeo)
  Zvals = zeros(Nqk,nvertelem)
  for e in 1:nrealelem  
    for i in 1:Nq * Nq * Nqk
      z = localvgeo[i,grid.x3id,e]

      ρ_node    = localQ[i,1,e]
      u_node    = localQ[i,2,e] / localQ[i,1,e]
      v_node    = localQ[i,3,e] / localQ[i,1,e]
      w_node    = localQ[i,4,e] / localQ[i,1,e]
      etot_node = localQ[i,5,e] / localQ[i,1,e]
      qt_node   = localQ[i,6,e] / localQ[i,1,e]
      
      #e_int = etot_node - 1//2 * (u_node^2 + v_node^2 + w_node^2) - grav * z
      ρu_node = SVector(ρ_node*u_node, ρ_node*v_node, ρ_node*w_node)
      e_int = internal_energy(ρ_node, ρ_node*etot_node, ρu_node, grav*z*ρ_node)

      #Phpart = PhasePartition(qt_node)
      #T      = air_temperature(e_int, Phpart)
      #p      = air_pressure(T, ρ_node, Phpart)
      #θ_l    = liquid_ice_pottemp(T, p, Phpart)
      #θ_v    = virtual_pottemp(T, p, Phpart)
      #θ      = dry_pottemp(T, p, Phpart)
      
      ts = PhaseEquil(e_int, qt_node, ρ_node)
      Phpart = PhasePartition(ts)
        
      thermoQ[i,1,e] = Phpart.liq
      thermoQ[i,2,e] = Phpart.ice
      thermoQ[i,3,e] = qt_node - Phpart.liq - Phpart.ice
      thermoQ[i,4,e] = ts.T
      thermoQ[i,5,e] = liquid_ice_pottemp(ts)
      thermoQ[i,6,e] = dry_pottemp(ts)
      thermoQ[i,7,e] = virtual_pottemp(ts)
    end
  end
    
  #Horizontal averages we might need
  #Horizontal averages we might need
  Horzavgs = zeros(Nqk, nvertelem,10)
  LWP_local = 0
  for eh in 1:nhorzelem
    for ev in 1:nvertelem
      e = ev + (eh - 1) * nvertelem

      for k in 1:Nqk
        for j in 1:Nq
          for i in 1:Nq
            ijk = i + Nq * ((j-1) + Nq * (k-1)) 
            Horzavgs[k,ev,1]  += localQ[ijk,1,e]  #ρ
            Horzavgs[k,ev,2]  += localQ[ijk,2,e]  #ρu
            Horzavgs[k,ev,3]  += localQ[ijk,3,e]  #ρv
            Horzavgs[k,ev,4]  += localQ[ijk,4,e]  #ρw
            Horzavgs[k,ev,5]  += thermoQ[ijk,5,e] #θl
            Horzavgs[k,ev,6]  += thermoQ[ijk,1,e] #ql
            Horzavgs[k,ev,7]  += thermoQ[ijk,3,e] #qv
            Horzavgs[k,ev,8]  += thermoQ[ijk,6,e] #θ
            Horzavgs[k,ev,9]  += localQ[ijk,6,e]  #qt
            Horzavgs[k,ev,10] += thermoQ[ijk,7,e] #θv
            #The next three lines are for the liquid water path
            if ev == floor(nvertelem/2) && k==floor(Nqk/2)
                LWP_local += (localaux[ijk,1,e] + localaux[ijk,2,e])/κ / (Nq * Nq * nhorzelem)
            end
          end
        end
      end
    end
  end
  Horzavgstot = zeros(Nqk,nvertelem,10)
for s in 1:10
  for ev in 1:nvertelem
       for k in 1:Nqk
       
           Horzavgs[k,ev,s] = Horzavgs[k,ev,s] /  (Nq * Nq * nhorzelem)
           Horzavgstot[k,ev,s] = MPI.Reduce(Horzavgs[k,ev,s], +, 0, MPI.COMM_WORLD)
           if mpirank == 0
               Horzavgstot[k,ev,s] = Horzavgstot[k,ev,s] / (nranks)
           end
       end
  end
end
if mpirank == 0
 LWP=MPI.Reduce(LWP_local, +, 0, MPI.COMM_WORLD) / (nranks) 
end
  S=zeros(Nqk,nvertelem,18)
  for eh in 1:nhorzelem
    for ev in 1:nvertelem
      e = ev + (eh - 1) * nvertelem

      for k in 1:Nqk
        for j in 1:Nq
          for i in 1:Nq
              ijk = i + Nq * ((j-1) + Nq * (k-1))
              wfluct = (localQ[ijk,4,e]/localQ[ijk,1,e] - Horzavgstot[k,ev,4] / Horzavgstot[k,ev,1])
              
              S[k,ev,1] += wfluct * (thermoQ[ijk,6,e] - Horzavgstot[k,ev,8]) #w'θ'
              S[k,ev,2] += wfluct * (thermoQ[ijk,3,e] - Horzavgstot[k,ev,7]) #w'qv'
              S[k,ev,3] += wfluct * (localQ[ijk,2,e] / localQ[ijk,1,e] - Horzavgstot[k,ev,2] / Horzavgstot[k,ev,1]) #w'u'
              S[k,ev,4] += wfluct * (localQ[ijk,3,e] / localQ[ijk,1,e] - Horzavgstot[k,ev,3] / Horzavgstot[k,ev,1]) #w'v'
              S[k,ev,5] += wfluct * wfluct  #w'w'
              S[k,ev,6] += wfluct * (localQ[ijk,1,e] - Horzavgstot[k,ev,1]) #w'\rho'
              S[k,ev,7] += Horzavgstot[k,ev,6] #ql
              S[k,ev,8] += wfluct * (thermoQ[ijk,1,e] - Horzavgstot[k,ev,6]) #w'ql'
              S[k,ev,9] += wfluct * wfluct * wfluct #w'w'w'
              
              S[k,ev,10] += (localQ[ijk,2,e]/localQ[ijk,1,e] - Horzavgstot[k,ev,2]/Horzavgstot[k,ev,1])^2  #u'u'
              S[k,ev,11] += (localQ[ijk,3,e]/localQ[ijk,1,e] - Horzavgstot[k,ev,3]/Horzavgstot[k,ev,1])^2  #v'v'
              
              S[k,ev,12] += (Horzavgstot[k,ev,8]) #<θ>
              S[k,ev,13] += (Horzavgstot[k,ev,9]/Horzavgstot[k,ev,1]) #<ρqt>/<ρ>
              S[k,ev,14] += wfluct * (localQ[ijk,6,e]/localQ[ijk,1,e] - Horzavgstot[k,ev,9]/Horzavgstot[k,ev,1]) #w'qt'
              S[k,ev,15] += (Horzavgstot[k,ev,5]) #<θl>
              S[k,ev,16] += wfluct * (thermoQ[ijk,7,e] - Horzavgstot[k,ev,10]) #w'θv'
              S[k,ev,17] += 0.5 * (S[k,ev,5] + S[k,ev,10] + S[k,ev,11]) #TKE
              S[k,ev,18] += wfluct * (thermoQ[ijk,5,e] - Horzavgstot[k,ev,5]) #w'θl'
              
              Zvals[k,ev] = localvgeo[ijk,grid.x3id,e] #z
          end
        end
      end
    end
  end

#See Outputs below for what S[k,ev,:] are respectively.
  S_avg = zeros(Nqk,nvertelem,18)
  for s in 1:18
    for ev in 1:nvertelem
      for k in 1:Nqk
        S[k,ev,s] = S[k,ev,s] /  (Nq * Nq * nhorzelem)
        S_avg[k,ev,s] = MPI.Reduce(S[k,ev,s], +, 0, MPI.COMM_WORLD)
     
        if mpirank == 0
          S_avg[k,ev,s] = S_avg[k,ev,s] / (nranks)
        end
      end
    end
  end

  OutputWtheta = zeros(nvertelem * Nqk)
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
  OutputZ = zeros(nvertelem * Nqk)
  OutputU = zeros(nvertelem * Nqk)
  OutputV = zeros(nvertelem * Nqk)
  Outputtheta = zeros(nvertelem * Nqk)
  Outputqt = zeros(nvertelem * Nqk)
  OutputWqt = zeros(nvertelem * Nqk)
  Outputthetaliq = zeros(nvertelem * Nqk)
  OutputWthetav = zeros(nvertelem * Nqk)
  OutputTKE = zeros(nvertelem * Nqk)
  OutputWthetal = zeros(nvertelem * Nqk)
  
  for ev in 1:nvertelem
    for k in 1:Nqk
      i=k + Nqk * (ev - 1)
      OutputWtheta[i] = S_avg[k,ev,1] # <w'theta'>
      OutputWQVAP[i] = S_avg[k,ev,2] # <w'qvap'>
      OutputWU[i] = S_avg[k,ev,3] # <w'u'>
      OutputWV[i] = S_avg[k,ev,4] # <w'v'>
      OutputWW[i] = S_avg[k,ev,5] # <w'w'>
      OutputWRHO[i] = S_avg[k,ev,6] #<w'rho'>
      OutputQLIQ[i] = S_avg[k,ev,7] # qliq 
      OutputWQLIQ[i] = S_avg[k,ev,8] # <w'qliq'>
      OutputWWW[i] = S_avg[k,ev,9]  #<w'w'w'>
      OutputUU[i] = S_avg[k,ev,10] #<u'u'>
      OutputVV[i] = S_avg[k,ev,11] #<v'v'>
      OutputU[i] =  Horzavgstot[k,ev,2] / Horzavgstot[k,ev,1] # <u> 
      OutputV[i] =  Horzavgstot[k,ev,3] / Horzavgstot[k,ev,1] # <v>
      Outputtheta[i] = S_avg[k,ev,12] # <theta> 
      Outputqt[i] = S_avg[k,ev,13]  # <qt>
      OutputWqt[i] = S_avg[k,ev,14] #<w'qt'> 
      Outputthetaliq[i] = S_avg[k,ev,15] # <thetaliq>
      OutputWthetav[i] = S_avg[k,ev,16] # <w'thetav'>
      OutputTKE[i] = S_avg[k,ev,17] # <TKE>
      OutputWthetal[i] = S_avg[k,ev,18] # <w'thetal'>
      OutputZ[i] = Zvals[k,ev] # Height
    end
  end
if mpirank == 0

  #Write <stats>
  io = open(diagnostics_fileout, "a")
     current_time_str = string(current_time_string, "\n")
     write(io, current_time_str)
     writedlm(io, [Outputtheta Outputthetaliq Outputqt OutputQLIQ OutputU OutputV OutputWtheta OutputWthetav OutputUU OutputVV OutputWW OutputWU OutputWV OutputWWW OutputWRHO OutputWQVAP OutputWQLIQ OutputWqt OutputZ])    
  close(io)

  #Write <LWP>
  io = open(LWP_fileout, "a")
  current_time_str = string(current_time_string)
  LWP_string = string(current_time_str, " ", LWP, "\n")
  write(io, LWP_string)
  close(io)

  #Write ICs file
  ICs_fileout = string("./output/ICs.dat")
  io = open(ICs_fileout, "w")
     writedlm(io, ["z" "rho" "theta" "theta_v" "theta_liq" "q_tot" "q_liq" "q_vap"])
  close(io) 
end
end


function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, grid_resolution, domain_size, zmax, zsponge, problem_name, diagnostics_fileout, OUTPATH,LWP_fileout)
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
  Q = init_ode_state(dg, FT(0); device=CPU())
  #lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  slow_ode_solver = LSRK144NiegemannDiehlBusch(slow_dg, Q; dt = slow_dt)
  # check if FastMethod is ARK, is there a better way ?
  fast_ode_solver = FastMethod(fast_dg, Q; dt = fast_dt)

  ode_solver = MultirateRungeKutta((slow_ode_solver, fast_ode_solver))

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      #= energy = norm(Q) =#
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
    cbvtk = GenericCallbacks.EveryXSimulationSteps(10000) do (init=false)
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
    current_time_str = string(gettime(ode_solver))
      gather_diagnostics(dg, Q, grid_resolution, current_time_str, diagnostics_fileout,κ,LWP_fileout)
  end

  solve!(Q, ode_solver; timeend=timeend, callbacks=(cbinfo, cbvtk, cbdiagnostics))

  #Get statistics at the end of the run:
  current_time_str = string(ODESolvers.gettime(lsrk))
  gather_diagnostics(dg, Q, grid_resolution, current_time_str, diagnostics_fileout,κ,LWP_fileout)

    
  # Print some end of the simulation information
 #= engf = norm(Q)
  Qe = init_ode_state(dg, gettime(ode_solver))

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
 # @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for ArrayType in ArrayTypes
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
    Δx, Δy, Δz = 50, 50, 25
    xmin, xmax = 0, 1500
    ymin, ymax = 0, 1500
    zmin, zmax = 0, 1500

    grid_resolution = [Δx, Δy, Δz]
    domain_size     = [xmin, xmax, ymin, ymax, zmin, zmax]
    dim = length(grid_resolution)
        
     brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                   grid1d(ymin, ymax, elemsize=FT(grid_resolution[2])*N),
                   grid1d(zmin, zmax, elemsize=FT(grid_resolution[end])*N))
    
    zmax = brickrange[dim][end]
    zsponge = FT(1200.0) #FT(0.75 * zmax)
    
    topl = StackedBrickTopology(mpicomm, brickrange,
                                periodicity = (true, true, false),
                                boundary=((0,0),(0,0),(1,2)))

    problem_name = "dycoms_IOstrings"
    dt = 0.01
    timeend = 14400

    #Create unique output path directory:
    OUTPATH = IOstrings_outpath_name(problem_name, grid_resolution)


    #open diagnostics file and write header:
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
    if mpirank == 0

      # Write diagnostics file:
      diagnostics_fileout = string(OUTPATH, "/statistic_diagnostics.dat")
      io = open(diagnostics_fileout, "w")
        diags_header_str = string("<theta>  <thetal>  <qt>  <ql>  <u>  <v>  <th.w>  <thv.w>  <u.u>  <v.v>  <w.w>  <u.w>  <v.w>  <w.w.w>  <rho.w>  <qv.w>  <ql.w>  <qt.w>  z\n")
        write(io, diags_header_str)
      close(io)

      # Write LWP file:  
      LWP_fileout = string(OUTPATH, "/LWP_calc.dat")
        io = open(LWP_fileout, "w")
        write(io, "LWP \n")
      close(io)
    end
      
    @info (ArrayType, dt, FT, dim)
    result = run(mpicomm, ArrayType, dim, topl, 
                 N, timeend, FT, FastMethod, C_smag, LHF, SHF, C_drag, grid_resolution, domain_size, zmax, zsponge, problem_name, diagnostics_fileout, OUTPATH, LWP_fileout)

  end
end

#nothing
