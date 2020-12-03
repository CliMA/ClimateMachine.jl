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
    xmax = FT(4800)
    ymax = FT(4800)
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
    )
    dgn_config = config_diagnostics(driver_config)

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "1mins", FT(0.0001)),
        ClimateMachine.ConservationCheck("ρe", "1mins", FT(0.0025)),
    )

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        check_euclidean_distance = true,
    )
end

main()
