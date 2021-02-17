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
using JLD2, FileIO
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

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
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 1
    nelem_vert = 50

    # Prescribe domain parameters
    zmax = FT(400)

    t0 = FT(0)

    # Simulation time
    timeend = FT(3600 * 6)
    CFLmax = FT(40.0)

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    N_updrafts = 1
    N_quad = 3 # Using N_quad = 1 leads to norm(Q) = NaN at init.
    turbconv = NoTurbConv()

    C_smag = FT(0.23)

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        turbulence = SmagorinskyLilly{FT}(C_smag),
        turbconv = turbconv,
    )

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_EDMF",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = FT(40),
        solver_type = ode_solver_type,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )

    #################### Change the ode_solver to implicit solver

    dg = solver_config.dg
    Q = solver_config.Q


    vdg = DGModel(
        driver_config;
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
    )


    # linear solver relative tolerance rtol which should be slightly smaller than the nonlinear solver tol
    linearsolver = BatchedGeneralizedMinimalResidual(
        dg,
        Q;
        max_subspace_size = 30,
        atol = -1.0,
        rtol = 5e-5,
    )

    """
    N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat

    F(Q) == 0
    ||F(Q^i) || / ||F(Q^0) || < tol

    """
    # ϵ is a sensity parameter for this problem, it determines the finite difference Jacobian dF = (F(Q + ϵdQ) - F(Q))/ϵ
    # I have also try larger tol, but tol = 1e-3 does not work
    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4, ϵ = 1.e-10)

    # this is a second order time integrator, to change it to a first order time integrator
    # change it ARK1ForwardBackwardEuler, which can reduce the cost by half at the cost of accuracy
    # and stability
    # preconditioner_update_freq = 50 means updating the preconditioner every 50 Newton solves,
    # update it more freqent will accelerate the convergence of linear solves, but updating it
    # is very expensive
    ode_solver = ARK2ImplicitExplicitMidpoint(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = 50,
        ),
        Q;
        dt = solver_config.dt,
        t0 = 0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )

    solver_config.solver = ode_solver

    #######################################

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

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (turbconv_filters(turbconv)...,),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    diag_arr = [single_stack_diagnostics(solver_config)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 5
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            diag_vs_z = single_stack_diagnostics(solver_config)

            nstep = getsteps(solver_config.solver)
            # Save to disc (for debugging):
            # @save "bomex_edmf_nstep=$nstep.jld2" diag_vs_z

            push!(diag_arr, diag_vs_z)
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    check_cons =
        (ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.001)),)

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        user_callbacks = (cbtmarfilter, cb_data_vs_time, cb_print_step),
        check_euclidean_distance = true,
    )

    diag_vs_z = single_stack_diagnostics(solver_config)
    push!(diag_arr, diag_vs_z)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, diag_arr, time_data
end

solver_config, diag_arr, time_data = main(Float64)

nothing
