using PyCall

mutable struct MyCallback
    initialized::Bool
    calls::Int
    finished::Bool
end


function GenericCallbacks.init!(cb::MyCallback, solver, Q, param, t)
    py"""
    def change_state_var(Q):
        Q.fill(-1.)
    """

    py"change_state_var($(parent(Q.ρ)))"
end

GenericCallbacks.call!(cb::MyCallback, _...) = (cb.calls += 1; nothing)
GenericCallbacks.fini!(cb::MyCallback, _...) = cb.finished = true

function main()
    # Working precision
    FT = Float64
    # Domain resolution and size
    Δx = FT(20)
    Δy = FT(1)
    Δz = FT(20)
    # Domain extents
    xmax = 1500
    ymax = 10
    zmax = 1500

    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]

    t_end = FT(0)
    dt = 10

    driver_config, solver_config =
        set_up_machine((xmax, ymax, zmax), (Δx, Δy, Δz), t_end, dt, qt_0)

    mpicomm = MPI.COMM_WORLD
    MPI.Barrier(mpicomm)

    testcb =
        GenericCallbacks.EveryXSimulationSteps(MyCallback(false, 0, false), 1)

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = nothing,
        user_callbacks = (testcb,),
        check_euclidean_distance = true,
    )

    cb_test_max = maximum(solver_config.Q.ρ)
    cb_test_min = minimum(solver_config.Q.ρ)

    @test isequal(cb_test_max, FT(-1)) && isequal(cb_test_min, FT(-1))
end

main()
