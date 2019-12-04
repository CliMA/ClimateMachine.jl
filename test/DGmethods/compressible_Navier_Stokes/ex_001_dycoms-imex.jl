using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Filters
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Diagnostics
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VTK

using CLIMA.Atmos: vars_state, vars_aux

using LinearAlgebra
using Random
using StaticArrays
using Logging
using Printf
using Dates
using CLIMA.ColumnwiseLUSolver: SingleColumnLU, ManyColumnLU, banded_matrix,
                                banded_matrix_vector_product!
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
  Rd::FT        = R_d
  Rv::FT        = R_v
  Rm::FT        = Rd
  ϵdv::FT       = Rv/Rd
  cpd::FT       = cp_d

  # These constants are those used by Stevens et al. (2005)
  qref::FT      = FT(9.0e-3)
  q_tot_sfc::FT = qref
  q_pt_sfc      = PhasePartition(q_tot_sfc)
  Rm_sfc::FT    = 461.5 #gas_constant_air(q_pt_sfc) # 461.5
  T_sfc::FT     = 290.4
  P_sfc::FT     = MSLP
  ρ_sfc::FT     = P_sfc / Rm_sfc / T_sfc
  # Specify moisture profiles
  q_liq::FT      = 0
  q_ice::FT      = 0
  q_c::FT        = 0
  zb::FT         = 600         # initial cloud bottom
  zi::FT         = 840         # initial cloud top
  ziplus::FT     = 875
  dz_cloud       = zi - zb
  θ_liq::FT      = 289
  if xvert <= zi
    θ_liq = FT(289.0)
    q_tot = qref
  else
    θ_liq = FT(297.0) + (xvert - zi)^(FT(1/3))
    q_tot = FT(1.5e-3)
  end
  q_c = q_liq + q_ice
  #Rm  = Rd*(FT(1) + (ϵdv - FT(1))*q_tot - ϵdv*q_c)

  # --------------------------------------------------
  # perturb initial state to break the symmetry and
  # trigger turbulent convection
  # --------------------------------------------------
  #randnum1   = rand(seed, FT) / 100
  #randnum2   = rand(seed, FT) / 1000
  #randnum1   = rand(Uniform(-0.02,0.02), 1, 1)
  #randnum2   = rand(Uniform(-0.000015,0.000015), 1, 1)
  #if xvert <= 25.0
  #  θ_liq += randnum1 * θ_liq
  #  q_tot += randnum2 * q_tot
  #end
  # --------------------------------------------------
  # END perturb initial state
  # --------------------------------------------------

    # Calculate PhasePartition object for vertical domain extent
    q_pt  = PhasePartition(q_tot, q_liq, q_ice)
    Rm    = gas_constant_air(q_pt)

    # Pressure
    H     = Rm_sfc * T_sfc / grav;
    p     = P_sfc * exp(-xvert/H);
    # Density, Temperature
    # TODO: temporary fix
    TS    = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, q_tot, p)
    ρ     = air_density(TS)
    T     = air_temperature(TS)
    q_pt  = PhasePartition_equil(T, ρ, q_tot)

  # Assign State Variables
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

end

function run(mpicomm,
             ArrayType,
             dim,
             topl,
             N,
             timeend,
             FT,
             C_smag,
             LHF,
             SHF,
             C_drag,
             xmax, ymax, zmax,
             zsponge,
             dt_exp, 
             dt_imex,
             explicit, 
             LinearModel,
             SolverMethod,
             out_dir)
    
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
  #Sponge:
  c_sponge = 1

  T_min = FT(289)
  T_s = FT(290.4)
  Γ_lapse = FT(grav/cp_d)
  Temp = LinearTemperatureProfile(T_min,T_s,Γ_lapse)
  RelHum = FT(0)
    
  # Model definition
  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(Temp,RelHum),
                     SmagorinskyLilly{}(C_smag),
                     EquilMoist(),
                     StevensRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1),
                     (Gravity(),
                      RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation),
                      GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)),
                     DYCOMS_BC{FT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)
  
  # Balancelaw description
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty(),
               direction=EveryDirection())

  linmodel = LinearModel(model)
  
  vdg = DGModel(linmodel,
                grid,
                Rusanov(),
                CentralNumericalFluxDiffusive(),
                CentralGradPenalty(),
                auxstate=dg.auxstate,
                direction=VerticalDirection())

    
  #Q = init_ode_state(dg, FT(0); device=CPU())
  Q = init_ode_state(dg, FT(0))

  cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
      Filters.apply!(Q, 6, dg.grid, TMARFilter())
      nothing
  end
    
  ###lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
  # Calculating initial condition norm
 #= eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0
 =#
  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(5, mpicomm) do (s=false)
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
  out_interval = 5000
  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(out_interval) do (init=false)
    fprefix = @sprintf("dycoms_%dD_mpirank%04d_step%04d", dim,
                       MPI.Comm_rank(mpicomm), step[1])
    outprefix = joinpath(out_dir, fprefix)
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)),
             dg.auxstate, flattenednames(vars_aux(model,FT)))

    step[1] += 1
    nothing
  end

       
    if explicit == 1
        numberofsteps = convert(Int64, cld(timeend, dt_exp))
        dt_exp = timeend / numberofsteps
        @info "EXP timestepper" dt_exp numberofsteps dt_exp*numberofsteps timeend
        solver = LSRK54CarpenterKennedy(dg, Q; dt = dt_exp, t0 = 0)
        #@timeit to "solve! EX DYCOMS- $LinearModel $SolverMethod $aspectratio $dt_exp $timeend" solve!(Q, solver; timeend=timeend, callbacks=(cbfilter,))


        # Get statistics during run
        diagnostics_time_str = string(now())
        cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(out_interval) do (init=false)
            sim_time_str = string(ODESolvers.gettime(solver))
            gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                               xmax, ymax, out_dir)
        end
          
        solve!(Q, solver; timeend=timeend, callbacks=(cbfilter, cbvtk, cbinfo, cbdiagnostics))

        
    else
        numberofsteps = convert(Int64, cld(timeend, dt_imex))
        dt_imex = timeend / numberofsteps
        @info "1DIMEX timestepper" dt_imex numberofsteps dt_imex*numberofsteps timeend

                
        solver = SolverMethod(dg, vdg, SingleColumnLU(), Q;
                              dt = dt_imex, t0 = 0,
                              split_nonlinear_linear=false)
        #@timeit to "solve! IMEX DYCOMS - $LinearModel $SolverMethod $aspectratio $dt_imex $timeend" solve!(Q, solver; numberofsteps=numberofsteps, callbacks=(cbfilter,),adjustfinalstep=false)

        # Get statistics during run
        diagnostics_time_str = string(now())
        cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(out_interval) do (init=false)
            sim_time_str = string(ODESolvers.gettime(solver))
            gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                               xmax, ymax, out_dir)
        end

        
        solve!(Q, solver; numberofsteps=numberofsteps, callbacks=(cbfilter, cbvtk, cbinfo, cbdiagnostics), adjustfinalstep=false)
    end
 
###  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk, cbdiagnostics))

  # Get statistics at the end of the run
#=  sim_time_str = string(ODESolvers.gettime(lsrk))
  gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                     xmax, ymax, out_dir)
=#
  # Print some end of the simulation information
 #= engf = norm(Q)
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

  out_dir = get(ENV, "OUT_DIR", "output")
  mkpath(out_dir)

  @static if haspkg("CUDAnative")
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  # @testset "$(@__FILE__)" for ArrayType in ArrayTypes
  for ArrayType in ArrayTypes
      #aspectratios = (1,3.5,7,)
      exp_step = 0
      linearmodels      = (AtmosAcousticGravityLinearModel,)
      IMEXSolverMethods = (ARK2GiraldoKellyConstantinescu,) #, ARK548L2SA2KennedyCarpenter)
      for SolverMethod in IMEXSolverMethods
          for LinearModel in linearmodels 
              for explicit in exp_step

                  # Problem type
                  FT = Float64
                  
                  # DG polynomial order
                  N = 4
                  
                  # SGS Filter constants
                  C_smag = FT(0.23)
                  LHF    = FT(115)
                  SHF    = FT(15)
                  C_drag = FT(0.0011)
                  
                  # User defined domain parameters
                  Δh, Δv = 40, 20
                  aspectratio = Δh/Δv
                  xmin, xmax = 0, 1000
                  ymin, ymax = 0, 1000
                  zmin, zmax = 0, 1500
                  
                  grid_resolution = [Δh, Δh, Δv]
                  domain_size     = [xmin, xmax, ymin, ymax, zmin, zmax]
                  dim = length(grid_resolution)
                  
                  brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                                grid1d(ymin, ymax, elemsize=FT(grid_resolution[2])*N),
                                grid1d(zmin, zmax, elemsize=FT(grid_resolution[end])*N))
                  zmax = brickrange[dim][end]
                  
                  zsponge = FT(1000.0)
                  topl = StackedBrickTopology(mpicomm, brickrange,
                                              periodicity = (true, true, false),
                                              boundary=((0,0),(0,0),(1,2)))
                  #dt = 0.0075
                  safety_fac = FT(0.5)
                  dt_exp = min(Δv/soundspeed_air(FT(330))/N * safety_fac, safety_fac * Δh/soundspeed_air(FT(330))/N) 
                  dt_imex = Δv/soundspeed_air(FT(330))/N * safety_fac
                  timeend = 14400
                  
                  @info @sprintf """Starting
                          ArrayType                 = %s
                          ODE_Solver                = %s
                          LinearModel               = %s
                          dt_exp                    = %.5e
                          dt_imp                    = %.5e
                          dt_ratio                  = %.3e
                          Δhoriz/Δvert              = %.5e
                          """ ArrayType SolverMethod LinearModel dt_exp dt_imex dt_imex/dt_exp aspectratio
                  
                  result = run(mpicomm,
                               ArrayType,
                               dim,
                               topl,
                               N,
                               timeend,
                               FT,
                               C_smag,
                               LHF, SHF,
                               C_drag,
                               xmax, ymax, zmax,
                               zsponge,
                               dt_exp, 
                               dt_imex,
                               explicit, 
                               LinearModel,
                               SolverMethod,
                               out_dir)
                  
              end
          end
      end
  end
    
#  @show LH_v0
#  @show R_d
#  @show MSLP
#  @show cp_d
end

###include(joinpath("..","..","..","src","Diagnostics","graph_diagnostic.jl"))
nothing
