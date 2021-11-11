function init_constants!(pysdm, varvals)
    pkg_constants = pyimport("PySDM.physics.constants")
    pkg_constants.Mv = CP.Planet.molmass_water(param_set)

    init!(pysdm, varvals)
end

function main()
    # Working precision
    FT = Float64
    # Domain resolution and size
    Δx = FT(20)
    Δy = FT(1)
    Δz = FT(20)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = 1500
    ymax = 10
    zmax = 1500

    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]

    t_end = FT(10)
    dt = 10

    driver_config, solver_config =
        set_up_machine((xmax, ymax, zmax), (Δx, Δy, Δz), t_end, dt, qt_0)

    mpicomm = MPI.COMM_WORLD

    MPI.Barrier(mpicomm)

    # output for netcdf
    boundaries = [
        FT(0) FT(0) FT(0)
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    pysdm_cb = set_up_pysdm(
        solver_config,
        interpol,
        (xmax, zmax),
        (Δx, Δz),
        init_constants!,
        do_step!,
        nothing,
    )

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = nothing,
        user_callbacks = (pysdm_cb,),
        check_euclidean_distance = true,
    )

    pkg_constants = pyimport("PySDM.physics.constants")
    @test pkg_constants.Mv == CP.Planet.molmass_water(param_set)
end

main()
