using JLD2, FileIO
using ClimateMachine
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.MPIStateArrays
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.BalanceLaws: vars_state
import ClimateMachine.BalanceLaws: projection
import ClimateMachine.DGMethods
using ClimateMachine.Atmos
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
import CLIMAParameters

include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

projection(
    ::AtmosModel,
    ::TendencyDef{Flux{O}, PV},
    x,
) where {O, PV <: Momentum} =
    x .* SArray{Tuple{3, 3}}(1, 1, 1, 1, 1, 1, 0, 0, 0)

projection(::AtmosModel, ::TendencyDef{Source, PV}, x) where {PV <: Momentum} =
    x .* SVector(1, 1, 0)

# CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 290.0 # default
CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 265

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

function main(::Type{FT}) where {FT}
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    sbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(sbl_args, "StableBoundaryLayer")
    @add_arg_table! sbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk|custom_sbl"
        arg_type = String
        default = "custom_sbl"
        # default = "prescribed"
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(400)

    t0 = FT(0)

    # Simulation time
    timeend = FT(1800)
    CFLmax  = FT(100)

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    N_updrafts = 1
    N_quad = 3
    # turbconv = NoTurbConv()
    turbconv = EDMF(FT, N_updrafts, N_quad, param_set)
    # compressibility = Compressible()
    compressibility = Anelastic1D()

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        # turbulence = ConstantKinematicViscosity(FT(0)),
        turbulence = SmagorinskyLilly{FT}(0.21),
        turbconv = turbconv,
        compressibility = compressibility,
    )

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_ANELASTIC_1D",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = FT(40),
        solver_type = ode_solver_type,
        # Ncutoff = 3,
        # numerical_flux_first_order = RoeNumericalFlux(),
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
        # fixed_number_of_steps = 25
    )

    # #################### Change the ode_solver to implicit solver

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
    #     max_subspace_size = 50,
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
        varsindex(vsp, :energy, :ρe),
    )
    vsa = vars_state(model, Auxiliary(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        varsindex(vsa, :turbconv),
    )
    # ---

    dgn_config = config_diagnostics(driver_config)

    # TMAR filter
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (turbconv_filters(turbconv)...,),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # boyd vandeven filter
    cb_boyd = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            # ("energy.ρe",turbconv_filters(turbconv)...),
            ("energy.ρe",),
            solver_config.dg.grid,
            BoydVandevenFilter(
                solver_config.dg.grid,
                1, #default=0
                4, #default=32
            ),
        )
        nothing
    end

    diag_arr = [single_stack_diagnostics(solver_config)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 10
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            diag_vs_z = single_stack_diagnostics(solver_config)

            nstep = getsteps(solver_config.solver)

            push!(diag_arr, diag_vs_z)
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    # Mass tendencies = 0 for Anelastic1D model,
    # so mass should be completely conserved:
    check_cons =
        (ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.00000001)),)

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        user_callbacks = (cb_boyd, cb_data_vs_time, cb_print_step),
        check_euclidean_distance = true,
    )

    diag_vs_z = single_stack_diagnostics(solver_config)
    push!(diag_arr, diag_vs_z)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, diag_arr, time_data
end

function config_diagnostics(driver_config, timeend)
    FT = eltype(driver_config.grid)
    info = driver_config.config_info
    interval = "$(cld(timeend, 2) + 10)ssecs"
    interval = "10steps"

    boundaries = [
        FT(0) FT(0) FT(0)
        FT(info.hmax) FT(info.hmax) FT(info.zmax)
    ]
    axes = (
        [FT(1)],
        [FT(1)],
        collect(range(boundaries[1, 3], boundaries[2, 3], step = FT(50)),),
    )
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries;
        axes = axes,
    )
    ds_dgngrp = setup_dump_state_diagnostics(
        SingleStackConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    dt_dgngrp = setup_dump_tendencies_diagnostics(
        SingleStackConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    return ClimateMachine.DiagnosticsConfiguration([ds_dgngrp, dt_dgngrp])
end

solver_config, diag_arr, time_data = main(Float64)

# include(joinpath(@__DIR__, "report_mse_sbl_anelastic.jl"))

nothing
