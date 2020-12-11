using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.Atmos: AtmosModel
using ClimateMachine.BalanceLaws: vars_state
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using ClimateMachine.Atmos: PressureGradientModel
using ClimateMachine.BalanceLaws
using ClimateMachine.Mesh.Filters: apply!
import ClimateMachine.DGMethods: custom_filter!
import ClimateMachine.DGMethods: custom_filter!, rhs_prehook_filters
using ClimateMachine.DGMethods: RemBL


ENV["CLIMATEMACHINE_SETTINGS_FIX_RNG_SEED"] = true
include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
include(joinpath(clima_dir, "docs", "plothelpers.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

"""
    init_state_prognostic!(
            turbconv::EDMF{FT},
            m::AtmosModel{FT},
            state::Vars,
            aux::Vars,
            localgeo,
            t::Real,
        ) where {FT}

Initialize EDMF state variables.
This method is only called at `t=0`.
"""
function init_state_prognostic!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
) where {FT}
    # Aliases:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    N_up = n_updrafts(turbconv)
    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e_int,q_tot,u,v,w)
    z = altitude(m, aux)

    # SCM setting - need to have separate cases coded and called from a folder - see what LES does
    # a thermo state is used here to convert the input θ to e_int profile
    e_int = internal_energy(m, state, aux)

    ts = PhaseDry(m.param_set, e_int, state.ρ)
    T = air_temperature(ts)
    p = air_pressure(ts)
    q = PhasePartition(ts)
    θ_liq = liquid_ice_pottemp(ts)

    a_min = turbconv.subdomains.a_min
    @unroll_map(N_up) do i
        up[i].ρa = gm.ρ * a_min
        up[i].ρaw = gm.ρu[3] * a_min
        up[i].ρaθ_liq = gm.ρ * a_min * θ_liq
        up[i].ρaq_tot = FT(0)
    end

    # initialize environment covariance with zero for now
    if z <= FT(250)
        en.ρatke =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
        en.ρaθ_liq_cv =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
    else
        en.ρatke = FT(0)
        en.ρaθ_liq_cv = FT(0)
    end
    en.ρaq_tot_cv = FT(0)
    en.ρaθ_liq_q_tot_cv = FT(0)
    return nothing
end;
using ClimateMachine.DGMethods: AbstractCustomFilter, apply!

struct ZeroVerticalVelocityFilter <: AbstractCustomFilter end
function custom_filter!(::ZeroVerticalVelocityFilter, bl, state, aux)
    # state.ρu = SVector(state.ρu[1], state.ρu[2], 0)
end

rhs_prehook_filters(atmos::BalanceLaw) = EDMFFilter()
rhs_prehook_filters(atmos::PressureGradientModel) = nothing

struct EDMFFilter <: AbstractCustomFilter end
function custom_filter!(::EDMFFilter, bl, state, aux)
    if hasproperty(bl, :turbconv)
        # FT = eltype(state)
        # # this ρu[3]=0 is only for single_stack
        # # state.ρu = SVector(state.ρu[1],state.ρu[2],0)
        # up = state.turbconv.updraft
        # en = state.turbconv.environment
        # N_up = n_updrafts(bl.turbconv)
        # ρ_gm = state.ρ
        # ρa_min = ρ_gm * bl.turbconv.subdomains.a_min
        # ρa_max = ρ_gm-ρa_min
        # ts = recover_thermo_state(bl, state, aux)
        # θ_liq_gm    = liquid_ice_pottemp(ts)
        # ρaθ_liq_ups = sum(vuntuple(i->up[i].ρaθ_liq, N_up))
        # ρa_ups      = sum(vuntuple(i->up[i].ρa, N_up))
        # ρaw_ups     = sum(vuntuple(i->up[i].ρaw, N_up))
        # ρa_en       = ρ_gm - ρa_ups
        # ρaw_en      = - ρaw_ups
        # θ_liq_en    = (θ_liq_gm - ρaθ_liq_ups) / ρa_en
        # # if !(θ_liq_en > FT(0))
        # z = altitude(bl, aux)
        # println("negative θ_liq_en")
        # @show(θ_liq_en)
        # @show(ρa_ups)
        # @show(up[1].ρa)
        # @show(up[1].ρaθ_liq)
        # @show(z)
        # @show(θ_liq_gm)
        # @show(ρaθ_liq_ups)
        # # end
        # w_en        = ρaw_en / ρa_en
        # @unroll_map(N_up) do i
        #     if !(ρa_min <= up[i].ρa <= ρa_max)
        #         up[i].ρa = min(max(up[i].ρa,ρa_min),ρa_max)
        #         up[i].ρaθ_liq = up[i].ρa * θ_liq_gm
        #         up[i].ρaw     = FT(0)
        #     end
        # end
        # en.ρatke = max(en.ρatke,FT(0))
        # en.ρaθ_liq_cv = max(en.ρaθ_liq_cv,FT(0))


        FT = eltype(state)
        # this ρu[3]=0 is only for single_stack
        # state.ρu = SVector(state.ρu[1],state.ρu[2],0)
        up = state.turbconv.updraft
        en = state.turbconv.environment
        N_up = n_updrafts(bl.turbconv)
        ρ_gm = state.ρ
        ρa_min = ρ_gm * bl.turbconv.subdomains.a_min
        ρa_max = ρ_gm-ρa_min
        ts = recover_thermo_state(bl, state, aux)
        θ_liq_gm    = liquid_ice_pottemp(ts)
        ρaθ_liq_ups = sum(vuntuple(i->up[i].ρaθ_liq, N_up))
        ρa_ups      = sum(vuntuple(i->up[i].ρa, N_up))
        ρaw_ups     = sum(vuntuple(i->up[i].ρaw, N_up))
        ρa_en       = ρ_gm - ρa_ups
        ρaw_en      = - ρaw_ups
        θ_liq_en    = (θ_liq_gm - ρaθ_liq_ups) / ρa_en
        # if !(θ_liq_en > FT(0))
        w_en        = ρaw_en / ρa_en
        @unroll_map(N_up) do i
            up[i].ρa = ρa_min
            up[i].ρaθ_liq = up[i].ρa * θ_liq_gm
            up[i].ρaw     = FT(0)
        end
        en.ρatke = max(en.ρatke,FT(0))
        en.ρaθ_liq_cv = max(en.ρaθ_liq_cv,FT(0))
        # en.ρaq_tot_cv = max(en.ρaq_tot_cv,FT(0))
        # en.ρaθ_liq_q_tot_cv = max(en.ρaθ_liq_q_tot_cv,FT(0))
        validate_variables(bl, state, aux, "custom_filter!")
    end
end

function main(::Type{FT}) where {FT}
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    sbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(sbl_args, "StableBoundaryLayer")
    @add_arg_table! sbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk"
        arg_type = String
        default = "bulk"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    config_type = SingleStackConfigType

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(400)
    t0 = FT(0)
    # Simulation time
    timeend = FT(3600*0.1)

    # Charlie's solver
    use_explicit_stepper_with_small_Δt = true
    if use_explicit_stepper_with_small_Δt
        CFLmax = FT(0.9)
        # ode_solver_type = ClimateMachine.IMEXSolverType()

         ode_solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    else
        CFLmax = FT(25)
        ode_solver_type = ClimateMachine.IMEXSolverType(
            implicit_model = AtmosAcousticGravityLinearModel,
            implicit_solver = SingleColumnLU,
            solver_method = ARK2GiraldoKellyConstantinescu,
            split_explicit_implicit = true,
            # split_explicit_implicit = false,
            discrete_splitting = false,
            # discrete_splitting = true,
        )
    end

    # old version
    # ode_solver_type = ClimateMachine.ExplicitSolverType(
    #     solver_method = LSRK144NiegemannDiehlBusch,
    # )

    N_updrafts = 1
    N_quad = 3 # Using N_quad = 1 leads to norm(Q) = NaN at init.
    turbconv = EDMF(FT, N_updrafts, N_quad)

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        # turbconv = turbconv,
        turbconv = NoTurbConv(),
    )

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_EDMF",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = zmax,
        solver_type = ode_solver_type,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
        # ode_dt = 2.64583e-01,
    )

    # #################### Change the ode_solver to implicit solver
    # CFLmax = FT(5)
    # dg = solver_config.dg
    # Q = solver_config.Q


    # vdg = DGModel(
    #     driver_config;
    #     state_auxiliary = dg.state_auxiliary,
    #     direction = VerticalDirection(),
    # )


    # # linear solver relative tolerance rtol which should be slightly smaller than the nonlinear solver tol
    # linearsolver = BatchedGeneralizedMinimalResidual(
    #     dg,
    #     Q;
    #     max_subspace_size = 30,
    #     atol = -1.0,
    #     rtol = 5e-5,
    # )

    # """
    # N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat
    # F(Q) == 0
    # ||F(Q^i) || / ||F(Q^0) || < tol
    # """
    # # ϵ is a sensity parameter for this problem, it determines the finite difference Jacobian dF = (F(Q + ϵdQ) - F(Q))/ϵ
    # # I have also try larger tol, but tol = 1e-3 does not work
    # nonlinearsolver =
    #     JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4, ϵ = 1.e-10)

    # # this is a second order time integrator, to change it to a first order time integrator
    # # change it ARK1ForwardBackwardEuler, which can reduce the cost by half at the cost of accuracy 
    # # and stability
    # # preconditioner_update_freq = 50 means updating the preconditioner every 50 Newton solves, 
    # # update it more freqent will accelerate the convergence of linear solves, but updating it 
    # # is very expensive
    # ode_solver = ARK2ImplicitExplicitMidpoint(
    #     dg,
    #     vdg,
    #     NonLinearBackwardEulerSolver(
    #         nonlinearsolver;
    #         isadjustable = true,
    #         preconditioner_update_freq = 50,
    #     ),
    #     Q;
    #     dt = solver_config.dt,
    #     t0 = 0,
    #     split_explicit_implicit = false,
    #     variant = NaiveVariant(),
    # )

    # solver_config.solver = ode_solver

    # #######################################



    # --- Zero-out horizontal variations:
    vsp = vars_state(model, Prognostic(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :turbconv),
    )
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :ρe),
    )
    vsa = vars_state(model, Auxiliary(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        varsindex(vsa, :turbconv),
    )
    # ---

    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (),
            solver_config.dg.grid,
            TMARFilter(),
        )
        # Filters.apply!( # comment this for NoTurbConv
        #     EDMFFilter(),
        #     solver_config.dg.grid,
        #     solver_config.dg.balance_law,
        #     solver_config.Q,
        #     solver_config.dg.state_auxiliary,
        # )
        nothing
    end

    # State variable
    Q = solver_config.Q
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρ₀ = Q.ρ
    ρe₀ = Q.ρe
    # DG variable sums
    Σρ₀ = sum(ρ₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)

    grid = driver_config.grid

    # state_types = (Prognostic(), Auxiliary(), GradientFlux())
    state_types = (Prognostic(), Auxiliary())
    dons_arr = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 5
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            push!(
                dons_arr,
                dict_of_nodal_states(solver_config, state_types; interp = true),
            )
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    cb_check_cons = GenericCallbacks.EveryXSimulationSteps(3000) do
        Q = solver_config.Q
        δρ = (sum(Q.ρ .* M) - Σρ₀) / Σρ₀
        δρe = (sum(Q.ρe .* M) .- Σρe₀) ./ Σρe₀
        @show (abs(δρ))
        @show (abs(δρe))
        @test (abs(δρ) <= 0.001)
        @test (abs(δρe) <= 0.0025)
        nothing
    end

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (
            cbtmarfilter,
            cb_check_cons,
            cb_data_vs_time,
            cb_print_step,
        ),
        check_euclidean_distance = true,
    )

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)
