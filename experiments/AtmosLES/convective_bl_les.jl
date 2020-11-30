include("convective_bl_model.jl")
function main()

    # TODO: this will move to the future namelist functionality
    cbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(cbl_args, "ConvectiveBoundaryLayer")
    @add_arg_table! cbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk"
        arg_type = String
        default = "bulk"
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = cbl_args)

    surface_flux = cl_args["surface_flux"]

    FT = Float64
    config_type = AtmosLESConfigType

    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(80)
    Δv = FT(80)

    resolution = (Δh, Δh, Δv)

    # Prescribe domain parameters
    xmax = Δh * FT(N+1) # FT(4800)
    ymax = Δh * FT(N+1) # FT(4800)
    zmax = FT(3200)

    t0 = FT(0)

    # Full simulation requires 16+ hours of simulated time
    timeend = FT(3600 * 0.1)
    CFLmax = FT(0.4)

    # Choose default Explicit solver
    ode_solver_type = ClimateMachine.ExplicitSolverType()

    model = convective_bl_model(FT, config_type, zmax, surface_flux)
    ics = model.problem.init_state_prognostic

    # Assemble configuration
    driver_config = ClimateMachine.AtmosLESConfiguration(
        "ConvectiveBoundaryLayer",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        ics,
        solver_type = ode_solver_type,
        model = model,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
        CFL_direction = HorizontalDirection(),
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end
    state_types = (Prognostic(), Auxiliary())
    all_data = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]
    n_outputs = 10
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            push!(all_data,
                dict_of_nodal_states(solver_config, state_types; interp = true)
            )
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "1mins", FT(0.0001)),
        ClimateMachine.ConservationCheck("ρe", "1mins", FT(0.0025)),
    )

    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cbtmarfilter,),
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        check_euclidean_distance = true,
    )
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver))
    return solver_config, all_data, time_data, state_types
end
solver_config, all_data, time_data, state_types = main()
