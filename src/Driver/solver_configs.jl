# CLIMA solver configurations
#
# Contains helper functions to establish solver configurations to be
# used with the CLIMA driver.

"""
    CLIMA.SolverConfiguration

Parameters needed by `CLIMA.solve!()` to run a simulation.
"""
struct SolverConfiguration{FT}
    name::String
    mpicomm::MPI.Comm
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
    DGmethods.courant(local_cfl, solver_config::SolverConfiguration;
                      Q=solver_config.Q, dt=solver_config.dt)

Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain with the model defined by `solver_config`. The
keyword arguments `Q` and `dt` can be used to call the courant method with a
different state `Q` or time step `dt` than are defined in `solver_config`.
"""
DGmethods.courant(
    f,
    sc::SolverConfiguration;
    Q = sc.Q,
    dt = sc.dt,
    simtime = gettime(sc.solver),
    direction = EveryDirection(),
) = DGmethods.courant(f, sc.dg, sc.dg.balancelaw, Q, dt, simtime, direction)

"""
    CLIMA.SolverConfiguration(
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
`CLIMA.invoke!()`.

# Arguments:
# - `t0::FT`: simulation start time.
# - `timeend::FT`: simulation end time.
# - `driver_config::DriverConfiguration`: from `AtmosLESConfiguration()`, etc.
# - `init_args...`: passed through to `init_state!()`.
# - `init_on_cpu=false`: run `init_state!()` on CPU?
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
    numfluxnondiff = driver_config.numfluxnondiff
    numfluxdiff = driver_config.numfluxdiff
    gradnumflux = driver_config.gradnumflux

    # create DG model, initialize ODE state
    dg = DGModel(
        bl,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
        modeldata = modeldata,
        diffusion_direction = diffdir,
    )
    @info @sprintf("Initializing %s", driver_config.name)
    Q = init_ode_state(dg, FT(0), init_args...; init_on_cpu = init_on_cpu)
    update_aux!(dg, bl, Q, FT(0))

    # create the linear model for IMEX solvers
    linmodel = nothing
    if isa(ode_solver_type, ExplicitSolverType)
        dtmodel = bl
    else # ode_solver_type === IMEXSolverType
        linmodel = ode_solver_type.linear_model(bl)
        dtmodel = linmodel
    end

    # default Courant number
    if Courant_number == nothing
        if ode_solver_type.solver_method == LSRK144NiegemannDiehlBusch
            Courant_number = FT(1.7)
        elseif ode_solver_type.solver_method == LSRK54CarpenterKennedy
            Courant_number = FT(0.3)
        else
            Courant_number = FT(0.5)
        end
    end

    # initial Δt specified or computed
    simtime = FT(0) # TODO: needs to be more general to account for restart:
    if ode_dt === nothing
        ode_dt =
            calculate_dt(dg, dtmodel, Q, Courant_number, simtime, CFL_direction)
    end
    numberofsteps = convert(Int, cld(timeend, ode_dt))
    timeend_dt_adjust && (ode_dt = timeend / numberofsteps)

    # create the solver
    if isa(ode_solver_type, ExplicitSolverType)
        solver = ode_solver_type.solver_method(dg, Q; dt = ode_dt, t0 = t0)
    elseif isa(ode_solver_type, MultirateSolverType)
        fast_dg = DGModel(
            linmodel,
            grid,
            numfluxnondiff,
            numfluxdiff,
            gradnumflux,
            auxstate = dg.auxstate,
        )
        slow_model = RemainderModel(bl, (linmodel,))
        slow_dg = DGModel(
            slow_model,
            grid,
            numfluxnondiff,
            numfluxdiff,
            gradnumflux,
            auxstate = dg.auxstate,
        )
        slow_solver = ode_solver_type.slow_method(slow_dg, Q; dt = ode_dt)
        fast_dt = ode_dt / ode_solver_type.timestep_ratio
        fast_solver = ode_solver_type.fast_method(fast_dg, Q; dt = fast_dt)
        solver = ode_solver_type.solver_method((slow_solver, fast_solver))
    else # solver_type === IMEXSolverType
        vdg = DGModel(
            linmodel,
            grid,
            numfluxnondiff,
            numfluxdiff,
            gradnumflux,
            auxstate = dg.auxstate,
            direction = VerticalDirection(),
        )
        solver = ode_solver_type.solver_method(
            dg,
            vdg,
            ode_solver_type.linear_solver(),
            Q;
            dt = ode_dt,
            t0 = t0,
        )
    end

    @toc SolverConfiguration

    return SolverConfiguration(
        driver_config.name,
        driver_config.mpicomm,
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
