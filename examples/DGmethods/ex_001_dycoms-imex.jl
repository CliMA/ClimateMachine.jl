using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection
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
#using CLIMA.Courant

using CLIMA.Atmos: vars_state, vars_aux

using LinearAlgebra
using Random
using Distributions
using StaticArrays
using Logging
using Printf
using Dates
using CLIMA.ColumnwiseLUSolver: SingleColumnLU, ManyColumnLU, banded_matrix,
                                banded_matrix_vector_product!
using CLIMA.DGmethods: EveryDirection, HorizontalDirection, VerticalDirection

const ArrayType = CLIMA.array_type()

const seed = MersenneTwister(0)

function global_max(A::MPIStateArray, states=1:size(A, 2))
  host_array = Array ∈ typeof(A).parameters
  h_A = host_array ? A : Array(A)
  locmax = maximum(view(h_A, :, states, A.realelems))
  MPI.Allreduce([locmax], MPI.MAX, A.mpicomm)[1]
end

function global_max_scalar(A, mpicomm)
  MPI.Allreduce(A, MPI.MAX, mpicomm)[1]
end


function extract_state(dg, localQ, ijk, e)
    bl = dg.balancelaw
    FT = eltype(localQ)
    nstate = num_state(bl, FT)
    l_Q = MArray{Tuple{nstate},FT}(undef)
    for s in 1:nstate
        l_Q[s] = localQ[ijk,s,e]
    end
    return Vars{vars_state(bl, FT)}(l_Q)
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
  Rm_sfc::FT    = gas_constant_air(q_pt_sfc)
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

  ugeo = FT(7)
  vgeo = FT(-5.5)
  u, v, w = ugeo, vgeo, FT(0)

  # --------------------------------------------------
  # perturb initial state to break the symmetry and
  # trigger turbulent convection
    # --------------------------------------------------
    randnum1   = rand(Uniform(-0.002,0.002))
    randnum2   = rand(Uniform(-0.00001,0.00001))
    randnum3   = rand(Uniform(-0.001,0.001))
    randnum4   = rand(Uniform(-0.001,0.001))
    if xvert <= 400.0
        θ_liq += randnum1 * θ_liq
        q_tot += randnum2 * q_tot
        u     += randnum3 * u
        v     += randnum4 * v
    end
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
    TS    = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, p, q_tot)
    ρ     = air_density(TS)
    T     = air_temperature(TS)
    q_pt  = PhasePartition_equil(T, ρ, q_tot)

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
             explicit,
             LinearModel,
             SolverMethod,
             out_dir)

  # Grid setup (topl contains brickrange information)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                          )


  # Problem constants
  # Radiation model
  κ             = FT(85)
  α_z           = FT(1)
  z_i           = FT(840)
  D_subsidence  = FT(0) # 0 for stable testing, 3.75e-6 in practice
  ρ_i           = FT(1.13)
  F_0           = FT(70)
  F_1           = FT(22)
  # Geostrophic forcing
  f_coriolis    = FT(1.03e-4) #FT(7.62e-5)
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
                     EquilMoist(5),
                     NoPrecipitation(),
                     DYCOMSRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1),
                     ConstantSubsidence{FT}(D_subsidence),
                     (Gravity(),
                      RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2),
                      GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)),
                     DYCOMS_BC{FT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)

  # Balancelaw description
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient(),
               direction=EveryDirection())

  linmodel = LinearModel(model)

  vdg = DGModel(linmodel,
                grid,
                Rusanov(),
                CentralNumericalFluxDiffusive(),
                CentralNumericalFluxGradient(),
                auxstate=dg.auxstate,
                direction=VerticalDirection())

  Q = init_ode_state(dg, FT(0); forcecpu=true)

  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(2) do (init=false)
      Filters.apply!(Q, 6, dg.grid, TMARFilter())
      nothing
  end
#=
  filterorder = 4
  filter = ExponentialFilter(grid, 0, filterorder)
  cbfilter = GenericCallbacks.EveryXSimulationSteps(50) do
      Filters.apply!(Q, 1:size(Q, 2), grid, filter)
      nothing
  end
    =#

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else

        @info @sprintf("""Update
                         dt      = %.5e
                         simtime = %.16e
                         runtime = %s""",
                       dt,
                       ODESolvers.gettime(solver),
                       Dates.format(convert(Dates.DateTime,
                                            Dates.now()-starttime[]),
                                    Dates.dateformat"HH:MM:SS"))

    end
  end

    # Setup VTK output callbacks
    out_interval = 10000
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
        Courant_number = 0.2
        dt             = Courant_number * min_node_distance(dg.grid, VerticalDirection())/soundspeed_air(FT(340))
        numberofsteps = convert(Int64, cld(timeend, dt))
        dt = timeend / numberofsteps #dt_exp
        @info "EXP timestepper" dt numberofsteps dt*numberofsteps timeend Courant_number
        solver = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
        #
        # Get statistics during run
        #
        out_interval_diags = 10000
        diagnostics_time_str = string(now())
        cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(out_interval_diags) do (init=false)
            sim_time_str = string(ODESolvers.gettime(solver))
            gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str, out_dir, ODESolvers.gettime(solver))

            #=
            #Calcualte Courant numbers:
            Dx = min_node_distance(grid, HorizontalDirection())
            Dz = min_node_distance(grid, VerticalDirection())

            dt_inout = Ref(dt)
            gather_Courant(mpicomm, dg, Q, xmax, ymax, Courant_number, out_dir,Dx,Dx,Dz,dt_inout)
            =#
        end
        #End get statistcs

        solve!(Q, solver; timeend=timeend, callbacks=(cbtmarfilter, cbinfo, cbdiagnostics))

    else
        #
        # 1D IMEX
        #
        Courant_number = 0.4
        dt             = Courant_number * min_node_distance(dg.grid, HorizontalDirection())/soundspeed_air(FT(290))
        # dt = 0.01
        numberofsteps = convert(Int64, cld(timeend, dt))
        dt = timeend / numberofsteps #dt_imex
        @info "1DIMEX timestepper" dt numberofsteps dt*numberofsteps timeend Courant_number

        #=solver = SolverMethod(dg, vdg, SingleColumnLU(), Q;
                              dt = dt, t0 = 0,
                              split_nonlinear_linear=false)
        =#
        solver = SolverMethod(dg, Q;
                              dt = dt, t0 = 0)
        
        # Get statistics during run
        out_interval_diags = 10000
        diagnostics_time_str = string(now())
        cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(out_interval_diags) do (init=false)
            sim_time_str = string(ODESolvers.gettime(solver))
            gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str, out_dir, ODESolvers.gettime(solver))
            #=
            #Calcualte Courant numbers:
            Dx = min_node_distance(grid, HorizontalDirection())
            Dz = min_node_distance(grid, VerticalDirection())
            dt_inout = Ref(dt)
            #@info " Ref(dt): " dt_inout
            gather_Courant(mpicomm, dg, Q,xmax, ymax, Courant_number, out_dir, Dx, Dx, Dz, dt_inout)
            #dt = dt_inout[]
            #@info " dt::::: " dt, Ref(dt)
            =#
        end
        #End get statistcs

        solve!(Q, solver; numberofsteps=numberofsteps, callbacks=(cbtmarfilter, cbdiagnostics, cbinfo), adjustfinalstep=false)
    end

    @info " END of the simulation!"

end

function main()
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

    #aspectratios = (1,3.5,7,)
    exp_step = 0
    linearmodels      = (AtmosAcousticGravityLinearModel,)
    #IMEXSolverMethods = (ARK548L2SA2KennedyCarpenter,)
    #IMEXSolverMethods = (ARK2GiraldoKellyConstantinescu,)
    IMEXSolverMethods = (LSRK144NiegemannDiehlBusch,)
    for SolverMethod in IMEXSolverMethods
        for LinearModel in linearmodels
            for explicit in exp_step

                # Problem type
                FT = Float64

                # DG polynomial order
                N = 4

                # SGS Filter constants
                C_smag = FT(0.21) # for stable testing, 0.18 in practice
                LHF    = FT(115)
                SHF    = FT(15)
                C_drag = FT(0.0011)

                # User defined domain parameters
                #Δh = FT(40)
                aspectratio = FT(2)
                Δv = FT(20)
                Δh = Δv * aspectratio
                #aspectratio = Δh/Δv

                xmin, xmax = 0, 1000
                ymin, ymax = 0, 1000
                zmin, zmax = 0, 2500

                grid_resolution = [Δh, Δh, Δv]
                domain_size     = [xmin, xmax, ymin, ymax, zmin, zmax]
                dim = length(grid_resolution)

                brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                              grid1d(ymin, ymax, elemsize=FT(grid_resolution[2])*N),
                              grid1d(zmin, zmax, elemsize=FT(grid_resolution[end])*N))
                zmax = brickrange[dim][end]

                zsponge = FT(1500.0)
                topl = StackedBrickTopology(mpicomm, brickrange,
                                            periodicity = (true, true, false),
                                            boundary=((0,0),(0,0),(1,2)))

                timeend = 14400

                @info @sprintf """Starting
                        ArrayType                 = %s
                        ODE_Solver                = %s
                        LinearModel               = %s
                        Δhoriz/Δvert              = %.5e
                        """ ArrayType SolverMethod LinearModel aspectratio
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
                             explicit,
                             LinearModel,
                             SolverMethod,
                             out_dir)
            end
        end
    end

    nothing
end

main()

