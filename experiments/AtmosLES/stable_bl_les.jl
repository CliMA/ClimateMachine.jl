include("stable_bl_model.jl")

function main()

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

    FT = Float64
    config_type = AtmosLESConfigType

    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(20)
    Δv = FT(20)

    resolution = (Δh, Δh, Δv)

    # Prescribe domain parameters
    xmax = FT(100)
    ymax = FT(100)
    zmax = FT(400)

    t0 = FT(0)

    # Required simulation time == 9hours
    timeend = FT(3600 * 0.1)
    CFLmax = FT(0.4)

    # Choose default IMEX solver
    ode_solver_type = ClimateMachine.ExplicitSolverType()

    C_smag = FT(0.23)
    
    model = stable_bl_model(FT,
        config_type,
        zmax,
        surface_flux;
        turbulence = SmagorinskyLilly{FT}(C_smag))

    ics = model.problem.init_state_prognostic

    # Assemble configuration
    driver_config = ClimateMachine.AtmosLESConfiguration(
        "StableBoundaryLayer",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_problem!,
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
