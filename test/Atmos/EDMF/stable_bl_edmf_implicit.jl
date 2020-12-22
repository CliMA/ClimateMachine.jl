
include("stable_bl_init.jl")
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

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 1
    nelem_vert = 50

    # Prescribe domain parameters
    zmax = FT(400)

    t0 = FT(0)

    # Simulation time
    timeend = FT(3600)
    # CFLmax = FT(0.50)
    CFLmax = FT(20)

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.HEVISolverType(
        FT;
        solver_method = ARK2ImplicitExplicitMidpoint,
        linear_max_subspace_size = 30,
        linear_atol = FT(-1.0),
        linear_rtol = FT(5e-5),
        nonlinear_max_iterations = 10,
        nonlinear_rtol = FT(1e-4),
        nonlinear_ϵ = FT(1.e-10),
        preconditioner_update_freq = Int(50),
    )

    N_updrafts = 1
    N_quad = 3 # Using N_quad = 1 leads to norm(Q) = NaN at init.
    turbconv = EDMF(FT, N_updrafts, N_quad)

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
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
            (turbconv_filters(turbconv)...,),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end


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

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.001)),
        ClimateMachine.ConservationCheck("ρe", "3000steps", FT(0.1)),
    )

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

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)
# include(joinpath(@__DIR__, "report_mse_sbl.jl"))
nothing
