# ClimateMachine solver configurations
#
# Contains helper functions to establish solver configurations to be
# used with the ClimateMachine driver.

"""
    ClimateMachine.SolverConfiguration

Parameters needed by `ClimateMachine.solve!()` to run a simulation.
"""
struct SolverConfiguration{FT}
    name::String
    mpicomm::MPI.Comm
    param_set::AbstractParameterSet
    dg::DGModel
    Q::MPIStateArray
    t0::FT
    timeend::FT
    dt::FT
    init_on_cpu::Bool
    numberofsteps::Int
    init_args
    solver
end

"""
    DGMethods.courant(local_cfl, solver_config::SolverConfiguration;
                      Q=solver_config.Q, dt=solver_config.dt)

Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain with the model defined by `solver_config`. The
keyword arguments `Q` and `dt` can be used to call the courant method with a
different state `Q` or time step `dt` than are defined in `solver_config`.
"""
DGMethods.courant(
    f,
    sc::SolverConfiguration;
    Q = sc.Q,
    dt = sc.dt,
    simtime = gettime(sc.solver),
    direction = EveryDirection(),
) = DGMethods.courant(f, sc.dg, sc.dg.balance_law, Q, dt, simtime, direction)

"""
    ClimateMachine.SolverConfiguration(
        t0::FT,
        timeend::FT,
        driver_config::DriverConfiguration,
        init_args...;
        init_on_cpu=false,
        ode_solver_type=driver_config.solver_type,
        ode_dt=nothing,
        modeldata=nothing,
        Courant_number=0.4,
        diffdir=EveryDirection(),
    )

Set up the DG model per the specified driver configuration, set up
the ODE solver, and return a `SolverConfiguration` to be used with
`ClimateMachine.invoke!()`.

# Arguments:
# - `t0::FT`: simulation start time.
# - `timeend::FT`: simulation end time.
# - `driver_config::DriverConfiguration`: from `AtmosLESConfiguration()`, etc.
# - `init_args...`: passed through to `init_state_conservative!()`.
# - `init_on_cpu=false`: run `init_state_conservative!()` on CPU?
# - `ode_solver_type=driver_config.solver_type`: override solver choice.
# - `ode_dt=nothing`: override timestep computation.
# - `modeldata=nothing`: passed through to `DGModel`.
# - `Courant_number=0.4`: for the simulation.
# - `diffdir=EveryDirection()`: restrict diffusivity direction.
# - `timeend_dt_adjust=true`: should `dt` be adjusted to hit `timeend` exactly
# - `CFL_direction=EveryDirection()`: direction for `calculate_dt`
"""
function SolverConfiguration(
    t0::FT,
    timeend::FT,
    driver_config::DriverConfiguration,
    init_args...;
    init_on_cpu = false,
    ode_solver_type = driver_config.solver_type,
    ode_dt = nothing,
    modeldata = nothing,
    Courant_number = nothing,
    diffdir = EveryDirection(),
    timeend_dt_adjust = true,
    CFL_direction = EveryDirection(),
) where {FT <: AbstractFloat}
    @tic SolverConfiguration

    bl = driver_config.bl
    grid = driver_config.grid
    numerical_flux_first_order = driver_config.numerical_flux_first_order
    numerical_flux_second_order = driver_config.numerical_flux_second_order
    numerical_flux_gradient = driver_config.numerical_flux_gradient

    # Create the DG model and initialize the ODE state. If we're restarting,
    # use state data from the checkpoint.
    if Settings.restart_from_num > 0
        s_Q, s_aux, t0 = Callbacks.read_checkpoint(
            Settings.checkpoint_dir,
            driver_config.name,
            driver_config.array_type,
            driver_config.mpicomm,
            Settings.restart_from_num,
        )

        state_auxiliary = restart_auxiliary_state(bl, grid, s_aux)

        dg = DGModel(
            bl,
            grid,
            numerical_flux_first_order,
            numerical_flux_second_order,
            numerical_flux_gradient,
            state_auxiliary = state_auxiliary,
            diffusion_direction = diffdir,
            modeldata = modeldata,
        )

        @info @sprintf(
            "Initializing %s from time %8.2f",
            driver_config.name,
            t0
        )
        Q = restart_ode_state(dg, s_Q; init_on_cpu = init_on_cpu)
    else
        dg = DGModel(
            bl,
            grid,
            numerical_flux_first_order,
            numerical_flux_second_order,
            numerical_flux_gradient,
            diffusion_direction = diffdir,
            modeldata = modeldata,
        )

        @info @sprintf("Initializing %s", driver_config.name,)
        Q = init_ode_state(dg, FT(0), init_args...; init_on_cpu = init_on_cpu)
    end
    update_auxiliary_state!(dg, bl, Q, FT(0), dg.grid.topology.realelems)

    # default Courant number
    # TODO: Think about revising this or drop it entirely.
    # This is difficult to determine/approximate
    # for MIS and general multirate methods.
    if Courant_number === nothing
        if isa(ode_solver_type, ExplicitSolverType)
            if ode_solver_type.solver_method == LSRK144NiegemannDiehlBusch
                Courant_number = FT(1.7)
            else
                @assert ode_solver_type.solver_method == LSRK54CarpenterKennedy
                Courant_number = FT(0.3)
            end
        else
            Courant_number = FT(0.5)
        end
    end

    # initial Δt specified or computed
    if ode_dt === nothing
        dtmodel = getdtmodel(ode_solver_type, bl)
        ode_dt = ClimateMachine.DGMethods.calculate_dt(
            dg,
            dtmodel,
            Q,
            Courant_number,
            t0,
            CFL_direction,
        )
    end
    numberofsteps = convert(Int, cld(timeend - t0, ode_dt))
    timeend_dt_adjust && (ode_dt = (timeend - t0) / numberofsteps)

    # create the solver
    solver = solversetup(ode_solver_type, dg, Q, ode_dt, t0, diffdir)

    @toc SolverConfiguration

    return SolverConfiguration(
        driver_config.name,
        driver_config.mpicomm,
        driver_config.param_set,
        dg,
        Q,
        t0,
        timeend,
        ode_dt,
        init_on_cpu,
        numberofsteps,
        init_args,
        solver,
    )
end
