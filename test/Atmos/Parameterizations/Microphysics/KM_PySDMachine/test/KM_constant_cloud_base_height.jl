module CloudBaseHeightTest

include("./utils/KM_CliMa_no_saturation_adjustment.jl")
include("./utils/KM_PySDM.jl")

function test_height_diff(A, tolerance)
    row_means = var(A, corrected = false, dims = 1)
    return maximum(row_means) < tolerance
end

function test_cloud_base_height(pysdm, varvals, t)
    do_step!(pysdm, varvals, t)
    println("[TEST] Cloud base test")
    @test test_height_diff(pysdm.particulator.products["radius_m1"].get(), 30)
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

    t_end = FT(10 * 30)
    dt = 10

    output_freq = 9
    interval = "90steps"

    driver_config, solver_config =
        set_up_machine((xmax, ymax, zmax), (Δx, Δy, Δz), t_end, dt, qt_0)

    mpicomm = MPI.COMM_WORLD

    # output for paraview
    # initialize base prefix directory from rank 0
    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    model = driver_config.bl
    vtkstep = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(output_freq) do
        out_dirname = @sprintf(
            "new_ex_1_mpirank%04d_step%04d",
            MPI.Comm_rank(mpicomm),
            vtkstep[1]
        )
        out_path_prefix = joinpath(vtkdir, out_dirname)
        @info "doing VTK output" out_path_prefix
        writevtk(
            out_path_prefix,
            solver_config.Q,
            solver_config.dg,
            flattenednames(vars_state(model, Prognostic(), FT)),
            solver_config.dg.state_auxiliary,
            flattenednames(vars_state(model, Auxiliary(), FT)),
        )
        vtkstep[1] += 1
        nothing
    end

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
    dgngrps = [
        setup_dump_state_diagnostics(
            AtmosLESConfigType(),
            interval,
            driver_config.name,
            interpol = interpol,
        ),
        setup_dump_aux_diagnostics(
            AtmosLESConfigType(),
            interval,
            driver_config.name,
            interpol = interpol,
        ),
    ]
    dgn_config = ClimateMachine.DiagnosticsConfiguration(dgngrps)

    pysdm_cb = set_up_pysdm(
        solver_config,
        interpol,
        (xmax, zmax),
        (Δx, Δz),
        init!,
        test_cloud_base_height,
        nothing,
    )

    # get aux variables indices for testing
    q_tot_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_tot)
    q_vap_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_vap)
    q_liq_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_liq)
    q_ice_ind = varsindex(vars_state(model, Auxiliary(), FT), :q_ice)

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbvtk, pysdm_cb),
        check_euclidean_distance = true,
    )

    # qt is conserved
    max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    min_q_tot = minimum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    @test isapprox(max_q_tot, qt_0; rtol = 1e-3)
    @test isapprox(min_q_tot, qt_0; rtol = 1e-3)

    # q_vap + q_liq = q_tot
    max_water_diff = maximum(abs.(
        solver_config.dg.state_auxiliary[:, q_tot_ind, :] .-
        solver_config.dg.state_auxiliary[:, q_vap_ind, :] .-
        solver_config.dg.state_auxiliary[:, q_liq_ind, :],
    ))
    @test isequal(max_water_diff, FT(0))

    # no ice
    max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    @test isequal(max_q_ice, FT(0))

    # q_liq ∈ reference range
    max_q_liq = maximum(solver_config.dg.state_auxiliary[:, q_liq_ind, :])
    min_q_liq = minimum(solver_config.dg.state_auxiliary[:, q_liq_ind, :])
    @test max_q_liq < FT(1e-3)
    @test isequal(min_q_liq, FT(0))
end

main()

end #module
