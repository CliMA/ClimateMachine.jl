using ArgParse
using CUDAapi
using Dates
using LinearAlgebra
using Logging
using MPI
using Printf
using CLIMAParameters

using ..Atmos
using ..Callbacks
using ..ColumnwiseLUSolver
using ..ConfigTypes
using ..Diagnostics
using ..DGmethods
using ..DGmethods:
    vars_state_conservative, vars_state_auxiliary, update_auxiliary_state!
using ..DGmethods.NumericalFluxes
using ..HydrostaticBoussinesq
using ..Mesh.Grids
using ..Mesh.Topologies
using ..MoistThermodynamics
using ..MPIStateArrays
using ..ODESolvers
using ..TicToc
using ..VariableTemplates

using CuArrays, CuArrays.CUDAdrv, CuArrays.CUDAnative

function _init_array(::Type{CuArray})
    comm = MPI.COMM_WORLD
    # allocate GPUs among MPI ranks
    local_comm =
        MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    # we intentionally oversubscribe GPUs for testing: may want to disable this for production
    CUDAnative.device!(MPI.Comm_rank(local_comm) % length(devices()))
    CuArrays.allowscalar(false)
    return nothing
end

_init_array(::Type{Array}) = nothing

const cuarray_pkgid =
    Base.PkgId(Base.UUID("3a865a2d-5b23-5a0f-bc46-62713ec82fae"), "CuArrays")
function gpu_allowscalar(val)
    if haskey(Base.loaded_modules, ClimateMachine.cuarray_pkgid)
        Base.loaded_modules[ClimateMachine.cuarray_pkgid].allowscalar(val)
    end
    return nothing
end

# Note that the initial values specified here are overwritten by the
# command line argument defaults in `parse_commandline()`.
Base.@kwdef mutable struct ClimateMachine_Settings
    disable_gpu::Bool = false
    show_updates::String = "60secs"
    diagnostics::String = "never"
    vtk::String = "never"
    monitor_timestep_duration::String = "never"
    monitor_courant_numbers::String = "never"
    checkpoint::String = "never"
    checkpoint_keep_one::Bool = true
    checkpoint_at_end::Bool = false
    checkpoint_dir::String = "checkpoint"
    restart_from_num::Int = -1
    log_level::String = "INFO"
    output_dir::String = "output"
    integration_testing::Bool = false
    array_type
end
const Settings = ClimateMachine_Settings(array_type = Array)


"""
    parse_commandline()
"""
function parse_commandline(custom_settings)
    exc_handler =
        isinteractive() ? ArgParse.debug_handler : ArgParse.default_handler
    s = ArgParseSettings(
        prog = PROGRAM_FILE,
        description = "Climate Machine: an Earth System Model that automatically learns from data\n",
        preformatted_description = true,
        epilog = """
            Any <interval> unless otherwise stated may be specified as:
                - 2hours or 10mins or 30secs => wall-clock time
                - 9.5smonths or 3.3sdays or 1.5shours => simulation time
                - 1000steps => simulation steps
                - never => disable
                - default => use experiment specified interval (only for diagnostics at present)
            """,
        preformatted_epilog = true,
        version = string(CLIMATEMACHINE_VERSION),
        exc_handler = exc_handler,
    )
    add_arg_group!(s, "ClimateMachine")

    @add_arg_table! s begin
        "--disable-gpu"
        help = "do not use the GPU"
        action = :store_true
        "--show-updates"
        help = "interval at which to show simulation updates"
        metavar = "<interval>"
        arg_type = String
        default = "60secs"
        "--diagnostics"
        help = "interval at which to collect diagnostics"
        metavar = "<interval>"
        arg_type = String
        default = "never"
        "--vtk"
        help = "interval at which to output VTK"
        metavar = "<interval>"
        arg_type = String
        default = "never"
        "--monitor-timestep-duration"
        help = "interval in time-steps at which to output wall-clock time per time-step"
        metavar = "<interval>"
        arg_type = String
        default = "never"
        "--monitor-courant-numbers"
        help = "interval at which to output acoustic, advective, and diffusive Courant numbers"
        metavar = "<interval>"
        arg_type = String
        default = "never"
        "--checkpoint"
        help = "interval at which to create a checkpoint"
        metavar = "<interval>"
        arg_type = String
        default = "never"
        "--checkpoint-keep-all"
        help = "keep all checkpoints (instead of just the most recent)"
        action = :store_true
        "--checkpoint-at-end"
        help = "create a checkpoint at the end of the simulation"
        action = :store_true
        "--checkpoint-dir"
        help = "the directory in which to store checkpoints"
        metavar = "<path>"
        arg_type = String
        default = "checkpoint"
        "--restart-from-num"
        help = "checkpoint number from which to restart (in <checkpoint-dir>)"
        metavar = "<number>"
        arg_type = Int
        default = -1
        "--log-level"
        help = "set the log level to one of debug/info/warn/error"
        metavar = "<level>"
        arg_type = String
        default = "info"
        "--output-dir"
        help = "directory for output data"
        metavar = "<path>"
        arg_type = String
        default = "output"
        "--integration-testing"
        help = "enable integration testing"
        action = :store_true
    end

    if custom_settings !== nothing
        import_settings!(s, custom_settings)
    end

    return parse_args(s)
end

"""
    ClimateMachine.array_type()

Return the array type used by ClimateMachine. This defaults to (CPU-based) `Array`
and is only correctly set (based on choice from the command line, from
an environment variable, or from experiment code) after `ClimateMachine.init()`
is called.
"""
array_type() = Settings.array_type

"""
    ClimateMachine.init(
        ;
        disable_gpu = false,
        arg_settings = nothing,
    )

Perform necessary initializations for ClimateMachine:
- Initialize MPI.
- Parse command line arguments. To support experiment-specific arguments,
`arg_settings` may be specified (it is an `ArgParse.ArgParseSettings`);
it will be imported into ClimateMachine's settings.
- Determine whether GPU(s) is available and should be used (pass
`disable-gpu = true` if not) and set the ClimateMachine array type appropriately.
- Set up the global logger.

Returns a `Dict` containing non-ClimateMachine command-line arguments.
"""
function init(; disable_gpu = false, arg_settings = nothing)
    # set up timing mechanism
    tictoc()

    # parse command line arguments
    parsed_args = nothing
    try
        parsed_args = parse_commandline(arg_settings)
        Settings.disable_gpu = disable_gpu || parsed_args["disable-gpu"]
        delete!(parsed_args, "disable-gpu")
        Settings.show_updates = parsed_args["show-updates"]
        delete!(parsed_args, "show-updates")
        Settings.diagnostics = parsed_args["diagnostics"]
        delete!(parsed_args, "diagnostics")
        Settings.vtk = parsed_args["vtk"]
        delete!(parsed_args, "vtk")
        Settings.monitor_timestep_duration =
            parsed_args["monitor-timestep-duration"]
        delete!(parsed_args, "monitor-timestep-duration")
        Settings.monitor_courant_numbers =
            parsed_args["monitor-courant-numbers"]
        delete!(parsed_args, "monitor-courant-numbers")
        Settings.log_level = uppercase(parsed_args["log-level"])
        delete!(parsed_args, "log-level")
        Settings.checkpoint = parsed_args["checkpoint"]
        delete!(parsed_args, "checkpoint")
        Settings.checkpoint_keep_one = !parsed_args["checkpoint-keep-all"]
        delete!(parsed_args, "checkpoint-keep-all")
        Settings.checkpoint_at_end = parsed_args["checkpoint-at-end"]
        delete!(parsed_args, "checkpoint-at-end")
        Settings.checkpoint_dir = parsed_args["checkpoint-dir"]
        delete!(parsed_args, "checkpoint-dir")
        Settings.restart_from_num = parsed_args["restart-from-num"]
        delete!(parsed_args, "restart-from-num")
        Settings.output_dir = parsed_args["output-dir"]
        delete!(parsed_args, "output-dir")
        Settings.integration_testing = parsed_args["integration-testing"]
        delete!(parsed_args, "integration-testing")
    catch
        Settings.disable_gpu = disable_gpu
    end

    # initialize MPI
    if !MPI.Initialized()
        MPI.Init()
    end

    # set up the array type appropriately depending on whether we're using GPUs
    if get(ENV, "CLIMATEMACHINE_GPU", "") != "false" &&
       !Settings.disable_gpu &&
       CUDAapi.has_cuda_gpu()
        atyp = CuArrays.CuArray
    else
        atyp = Array
    end
    _init_array(atyp)
    Settings.array_type = atyp

    # create the output directory if needed
    if Settings.diagnostics !== "never" || Settings.vtk !== "never"
        mkpath(Settings.output_dir)
    end
    if Settings.checkpoint !== "never" || Settings.checkpoint_at_end
        mkpath(Settings.checkpoint_dir)
    end

    # set up logging
    loglevel = Settings.log_level == "DEBUG" ? Logging.Debug :
        Settings.log_level == "WARN" ? Logging.Warn :
        Settings.log_level == "ERROR" ? Logging.Error : Logging.Info
    # TODO: write a better MPI logging back-end and also integrate Dlog for large scale
    logger_stream = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    return parsed_args
end

include("driver_configs.jl")
include("solver_configs.jl")
include("diagnostics_configs.jl")

"""
    ClimateMachine.invoke!(
        solver_config::SolverConfiguration;
        diagnostics_config = nothing,
        user_callbacks = (),
        check_euclidean_distance = false,
        adjustfinalstep = false,
        user_info_callback = (init) -> nothing,
    )

Run the simulation defined by `solver_config`.

Keyword Arguments:

The `user_callbacks` are passed to the ODE solver as callback functions;
see [`ODESolvers.solve!]@ref().

If `check_euclidean_distance` is `true, then the Euclidean distance
between the final solution and initial condition function evaluated with
`solver_config.timeend` is reported.

The value of 'adjustfinalstep` is passed to the ODE solver; see
[`ODESolvers.solve!]@ref().

The function `user_info_callback` is called after the default info
callback (which is called every `Settings.show_updates` interval). The
single input argument `init` is `true` when the callback is called for
initialization (before time stepping begins) and `false` when called
during the actual ODE solve; see [`GenericCallbacks`](@ref) and
[`ODESolvers.solve!]@ref().
"""
function invoke!(
    solver_config::SolverConfiguration;
    diagnostics_config = nothing,
    user_callbacks = (),
    check_euclidean_distance = false,
    adjustfinalstep = false,
    user_info_callback = (init) -> nothing,
)
    mpicomm = solver_config.mpicomm
    dg = solver_config.dg
    bl = dg.balance_law
    Q = solver_config.Q
    FT = eltype(Q)
    timeend = solver_config.timeend
    init_on_cpu = solver_config.init_on_cpu
    init_args = solver_config.init_args
    solver = solver_config.solver

    # set up callbacks
    callbacks = ()

    # info callback
    cb_updates = Callbacks.show_updates(
        Settings.show_updates,
        solver_config,
        user_info_callback,
    )
    if cb_updates !== nothing
        callbacks = (callbacks..., cb_updates)
    end

    # diagnostics callback(s)
    if Settings.diagnostics !== "never" && diagnostics_config !== nothing
        dgn_starttime = replace(string(now()), ":" => ".")
        Diagnostics.init(mpicomm, dg, Q, dgn_starttime, Settings.output_dir)

        dgncbs = Callbacks.diagnostics(
            Settings.diagnostics,
            solver_config,
            dgn_starttime,
            diagnostics_config,
        )
        callbacks = (callbacks..., dgncbs...)
    end

    # vtk callback
    cb_vtk = Callbacks.vtk(Settings.vtk, solver_config, Settings.output_dir)
    if cb_vtk !== nothing
        callbacks = (callbacks..., cb_vtk)
    end

    # timestep duration monitor
    cb_mtd = Callbacks.monitor_timestep_duration(
        Settings.monitor_timestep_duration,
        Settings.array_type,
        mpicomm,
    )
    if cb_mtd !== nothing
        callbacks = (callbacks..., cb_mtd)
    end

    # Courant number monitor
    cb_mcn = Callbacks.monitor_courant_numbers(
        Settings.monitor_courant_numbers,
        solver_config,
    )
    if cb_mcn !== nothing
        callbacks = (callbacks..., cb_mcn)
    end

    # checkpointing callback
    cb_checkpoint = Callbacks.checkpoint(
        Settings.checkpoint,
        Settings.checkpoint_keep_one,
        solver_config,
        Settings.checkpoint_dir,
    )
    if cb_checkpoint !== nothing
        callbacks = (callbacks..., cb_checkpoint)
    end

    # user callbacks
    callbacks = (user_callbacks..., callbacks...)

    # initial condition norm
    eng0 = norm(Q)
    @info @sprintf(
        """
Starting %s
    dt              = %.5e
    timeend         = %8.2f
    number of steps = %d
    norm(Q)         = %.16e""",
        solver_config.name,
        solver_config.dt,
        solver_config.timeend,
        solver_config.numberofsteps,
        eng0
    )

    # run the simulation
    @tic solve!
    solve!(
        Q,
        solver;
        timeend = timeend,
        callbacks = callbacks,
        adjustfinalstep = adjustfinalstep,
    )
    @toc solve!

    # write end checkpoint if requested
    if Settings.checkpoint_at_end
        Callbacks.write_checkpoint(
            solver_config,
            Settings.checkpoint_dir,
            solver_config.name,
            mpicomm,
            solver_config.numberofsteps,
        )
    end

    # fini diagnostics groups
    if Settings.diagnostics !== "never" && diagnostics_config !== nothing
        currtime = ODESolvers.gettime(solver)
        for dgngrp in diagnostics_config.groups
            dgngrp(currtime, fini = true)
        end
    end

    engf = norm(Q)
    @info @sprintf(
        """
Finished
    norm(Q)            = %.16e
    norm(Q) / norm(Q₀) = %.16e
    norm(Q) - norm(Q₀) = %.16e""",
        engf,
        engf / eng0,
        engf - eng0
    )

    if check_euclidean_distance
        Qe =
            init_ode_state(dg, timeend, init_args...; init_on_cpu = init_on_cpu)
        engfe = norm(Qe)
        errf = euclidean_distance(Q, Qe)
        @info @sprintf(
            """
Euclidean distance
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e""",
            errf,
            errf / engfe
        )
    end

    return engf / eng0
end
