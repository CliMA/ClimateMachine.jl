using Base.Threads

using ArgParse
using CUDAapi
using Dates
using LinearAlgebra
using Logging
using MPI
using Printf
using Random

using CLIMAParameters

using ..Atmos
using ..Callbacks
using ..SystemSolvers
using ..ConfigTypes
using ..Diagnostics
using ..DGMethods
using ..DGMethods:
    vars_state_conservative, vars_state_auxiliary, update_auxiliary_state!
using ..DGMethods.NumericalFluxes
using ..HydrostaticBoussinesq
using ..Mesh.Grids
using ..Mesh.Topologies
using ..Thermodynamics
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

function gpu_allowscalar(val::Bool)
    if haskey(Base.loaded_modules, ClimateMachine.cuarray_pkgid)
        Base.loaded_modules[ClimateMachine.cuarray_pkgid].allowscalar(val)
    end
    return
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
    fix_rng_seed::Bool = false
    log_level::String = "INFO"
    output_dir::String = "output"
    integration_testing::Bool = false
    array_type::Type = Array
end

const Settings = ClimateMachine_Settings()


"""
    ClimateMachine.array_type()

Return the array type used by ClimateMachine. This defaults to (CPU-based) `Array`
and is only correctly set (based on choice from the command line, from
an environment variable, or from experiment code) after `ClimateMachine.init()`
is called.
"""
array_type() = Settings.array_type


"""
    get_setting(setting_name::Symbol, settings, defaults)

Define fallback behavior for driver settings, first accessing overloaded `settings`
if defined, followed by constructed global ENV variable `CLIMATEMACHINE_SETTINGS_<SETTING_NAME>`,
then (global) defaults.

Returns setting value.
"""
function get_setting(setting_name::Symbol, settings, defaults)
    if !haskey(defaults, setting_name)
        error("setting $setting_name is not defined in `defaults`")
    end
    setting_type = typeof(defaults[setting_name])
    setting_env = "CLIMATEMACHINE_SETTINGS_" * uppercase(String(setting_name))
    if haskey(settings, setting_name)
        return convert(setting_type, settings[setting_name])
    elseif haskey(ENV, setting_env)
        env_val = ENV[setting_env]
        v = tryparse(setting_type, env_val)
        if isnothing(v)
            error("Cannot parse ENV $setting_env value $env_val, to setting type $setting_type")
        end
        return v
    elseif haskey(defaults, setting_name)
        return defaults[setting_name]
    else
        error("setting $setting_name is not contained in either settings or defaults")
    end
end

"""
    parse_commandline(defaults::Union{Nothing, Dict{Symbol,Any}),
                      custom_settings::Union{Nothing,ArgParseSettings}=nothing)

Parse process command line ARGS values.
If `defaults` is not nothing, override default values for cli argument defaults.
If `custom_settings` arg is not nothing, add ArgParseSettings to the parsing step.

Returns a `Dict` containing parsed process ARGS values.
"""
function parse_commandline(
    defaults::Union{Nothing, Dict{Symbol, Any}} = nothing,
    custom_settings::Union{Nothing, ArgParseSettings} = nothing,
)
    if isnothing(defaults)
        defaults = Dict{Symbol, Any}()
    end
    exc_handler = ArgParse.default_handler
    if Base.isinteractive()
        exc_handler = ArgParse.debug_handler
    end
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
        autofix_names = true,  # switches --flag-name to 'flag_name'
    )
    add_arg_group!(s, "ClimateMachine")

    global_defaults = Dict{Symbol, Any}(
        (n, getproperty(Settings, n)) for n in propertynames(Settings)
    )

    @add_arg_table! s begin
        "--disable-gpu"
        help = "do not use the GPU"
        action = :store_true
        "--show-updates"
        help = "interval at which to show simulation updates"
        metavar = "<interval>"
        arg_type = String
        default = get_setting(:show_updates, defaults, global_defaults)
        "--diagnostics"
        help = "interval at which to collect diagnostics"
        metavar = "<interval>"
        arg_type = String
        default = get_setting(:diagnostics, defaults, global_defaults)
        "--vtk"
        help = "interval at which to output VTK"
        metavar = "<interval>"
        arg_type = String
        default = get_setting(:vtk, defaults, global_defaults)
        "--monitor-timestep-duration"
        help = "interval in time-steps at which to output wall-clock time per time-step"
        metavar = "<interval>"
        arg_type = String
        default =
            get_setting(:monitor_timestep_duration, defaults, global_defaults)
        "--monitor-courant-numbers"
        help = "interval at which to output acoustic, advective, and diffusive Courant numbers"
        metavar = "<interval>"
        arg_type = String
        default =
            get_setting(:monitor_courant_numbers, defaults, global_defaults)
        "--checkpoint"
        help = "interval at which to create a checkpoint"
        metavar = "<interval>"
        arg_type = String
        default = get_setting(:checkpoint, defaults, global_defaults)
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
        default = get_setting(:checkpoint_dir, defaults, global_defaults)
        "--restart-from-num"
        help = "checkpoint number from which to restart (in <checkpoint-dir>)"
        metavar = "<number>"
        arg_type = Int
        default = get_setting(:restart_from_num, defaults, global_defaults)
        "--fix-rng-seed"
        help = "set RNG seed to a fixed value for reproducibility"
        action = :store_true
        "--log-level"
        help = "set the log level to one of debug/info/warn/error"
        metavar = "<level>"
        arg_type = String
        default = uppercase(get_setting(:log_level, defaults, global_defaults))
        "--output-dir"
        help = "directory for output data"
        metavar = "<path>"
        arg_type = String
        default = get(defaults, :output_dir) do
            get(ENV, "CLIMATEMACHINE_OUTPUT_DIR") do
                get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", Settings.output_dir)
            end
        end
        "--integration-testing"
        help = "enable integration testing"
        action = :store_true
    end
    # add custom cli argparse settings if provided
    if !isnothing(custom_settings)
        import_settings!(s, custom_settings)
    end
    return parse_args(s)
end


"""
    ClimateMachine.cli(;arg_settings=nothing, init_driver=true, kwargs...)

Initialize the ClimateMachine runtime with cli argument parsing.

- Additional `ArgParseSettings` behavior can be injected into the default
ClimateMachine `ArgParseSettings` cofiguration by setting the `custom_settings`
keyword value.

- Setting `init_driver = false` will set the `ClimateMachine.Settings` singleton
values without initializing the ClimateMachine driver runtime.

- ClimateMachine.init key value pairs can be supplied to overload
default system defaults at runtime, default values will be merged with the
parsed argument settings, with parsed cli argument values taking precedent
over runtime defined default values.


Returns a `Dict` containing parsed process ARGS values.
"""
function cli(;
    custom_settings::Union{Nothing, ArgParseSettings} = nothing,
    init_driver::Bool = true,
    kwargs...,
)
    kw_defaults = Dict{Symbol, Any}(kwargs)
    parsed_args = parse_commandline(kw_defaults, custom_settings)
    # we need to munge the parsed arg dict a bit as parsed arg keys
    # and climatemachine initialization keywords are not 1:1
    parsed_args["checkpoint_keep_one"] = !parsed_args["checkpoint_keep_all"]
    parsed_kwargs = Dict{Symbol, Any}((Symbol(k), v) for (k, v) in parsed_args)
    # allow for setting cli arguments as hard defaults that override parsed process ARGS
    init_kwargs = merge(kw_defaults, parsed_kwargs)
    # call init with munged kw arguments
    ClimateMachine.init(; init_driver = init_driver, init_kwargs...)
    return parsed_args
end


"""
    ClimateMachine.init(;init_driver::Bool=true, kwargs...)

Perform necessary initializations for ClimateMachine:
- Initialize MPI.
- Parse command line arguments. To support experiment-specific arguments,
`arg_settings` may be specified (it is an `ArgParse.ArgParseSettings`);
it will be imported into ClimateMachine's settings.
- Determine whether GPU(s) is available and should be used (pass
`disable-gpu = true` if not) and set the ClimateMachine array type appropriately.
- Set up the global logger.

Setting `init_driverd = false` will set the `ClimateMachine.Settings` singleton
values without initializing the ClimateMachine driver runtime.

`ClimateMachine.Settings` values can be overloaded at runtime upon initialization.
If keyword argument overloads are not supplied, the `init` routine will try and
fallback on any `CLIMATEMACHINE_SETTINGS_<VALUE>` `ENV` variables defined for
the process, otherwise the defaulting to `ClimateMachine.Settings`.

# Keyword Arguments
- `disable_gpu::Bool = false`:
        do not use the GPU
- `show_updates::String = "60secs"`:
        interval at which to show simulation updates
- `diagnostics::String = "never"`:
        interval at which to collect diagnostics"
- `vtk::String = "never"`:
        inteverval at which to write simulation vtk output
- `monitor_timestep_duration::String = "never"`:
        interval in time-steps at which to output wall-clock time per time-step
- `monitor_courant_numbers::String = "never"`:
        interval at which to output acoustic, advective, and diffusive Courant numbers"
- `checkpoint::String = "never"`:
        interval at which to output a checkpoint
- `checkpoint_keep_one::Bool = true`: (interval)
        keep all checkpoints (instead of just the most recent)"
- `checkpoint_at_end::Bool = false`:
        create a checkpoint at the end of the simulation"
- `checkpoint_dir::String = "checkpoint"`:
        absolute or relative path to checkpoint directory
- `restart_from_num::Int = -1`:
        checkpoint number from which to restart (in `checkpoint_dir`)
- `fix_rng_seed::Bool = false`:
        set RNG seed to a fixed value for reproducibility
- `log_level::String = "INFO"`:
        log level for ClimateMachine global default runtime logger
- `output_dir::String = "output"`: (path)
        absolute or relative path to output data directory
- `integration_testing::Bool = false`:
        enable integration_testing
"""
function init(; init_driver::Bool = true, kwargs...)
    # init global setting values
    # TODO: add validation for initialization values

    if haskey(kwargs, :disable_gpu)
        Settings.disable_gpu = kwargs[:disable_gpu]
    elseif haskey(ENV, "CLIMATEMACHINE_GPU")
        @warn(
            "CLIMATEMACHINE_GPU will be deprecated; " *
            "use CLIMATEMACHINE_SETTINGS_DISABLE_GPU"
        )
        Settings.disable_gpu = ENV["CLIMATEMACHINE_GPU"] == "false"
    elseif haskey(ENV, "CLIMATEMACHINE_SETTINGS_DISABLE_GPU")
        Settings.disable_gpu =
            parse(Bool, ENV["CLIMATEMACHINE_SETTINGS_DISABLE_GPU"])
    end

    if haskey(kwargs, :output_dir)
        Settings.output_dir = kwargs[:output_dir]
    elseif haskey(ENV, "CLIMATEMACHINE_OUTPUT_DIR")
        @warn(
            "CLIMATEMACHINE_OUTPUT_DIR will be deprecated; " *
            "use CLIMATEMACHINE_SETTINGS_OUTPUT_DIR"
        )
        Settings.output_dir = ENV["CLIMATEMACHINE_OUTPUT_DIR"]
    elseif haskey(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR")
        Settings.output_dir = ENV["CLIMATEMACHINE_SETTINGS_OUTPUT_DIR"]
    end

    global_defaults = Dict{Symbol, Any}(
        (n, getproperty(Settings, n)) for n in propertynames(Settings)
    )

    for n in propertynames(Settings)
        # skip over the special backwards compat cases defined above
        if n == :disable_gpu || n == :output_dir
            continue
        end
        setproperty!(Settings, n, get_setting(n, kwargs, global_defaults))
    end

    # set up the array type appropriately depending on whether we're using GPUs
    if !Settings.disable_gpu && CUDAapi.has_cuda_gpu()
        Settings.array_type = CuArrays.CuArray
    end

    if init_driver
        _init_driver(Settings)
    end
    return
end


function _init_driver(settings::ClimateMachine_Settings)
    # set up timing mechanism
    tictoc()

    # initialize MPI
    if !MPI.Initialized()
        MPI.Init()
    end

    # initialize the array GPU backend if appropriate
    _init_array(settings.array_type)

    # fix the RNG seeds if requested
    if settings.fix_rng_seed
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        for tid in 1:nthreads()
            s = 1000 * rank + tid
            Random.seed!(Random.default_rng(tid), s)
        end
    end

    # create the output directory if needed on delegated rank
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if settings.diagnostics != "never" || settings.vtk != "never"
            mkpath(settings.output_dir)
        end
        if settings.checkpoint != "never" || settings.checkpoint_at_end
            mkpath(settings.checkpoint_dir)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)

    # set up logging
    log_level_str = uppercase(settings.log_level)
    loglevel = log_level_str == "DEBUG" ? Logging.Debug :
        log_level_str == "WARN" ? Logging.Warn :
        log_level_str == "ERROR" ? Logging.Error : Logging.Info
    # TODO: write a better MPI logging back-end and also integrate Dlog for large scale
    logger_stream = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))
    return
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
    if !isnothing(cb_updates)
        callbacks = (callbacks..., cb_updates)
    end

    # diagnostics callback(s)
    if Settings.diagnostics != "never" && !isnothing(diagnostics_config)
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
    if !isnothing(cb_vtk)
        callbacks = (callbacks..., cb_vtk)
    end

    # timestep duration monitor
    cb_mtd = Callbacks.monitor_timestep_duration(
        Settings.monitor_timestep_duration,
        Settings.array_type,
        mpicomm,
    )
    if !isnothing(cb_mtd)
        callbacks = (callbacks..., cb_mtd)
    end

    # Courant number monitor
    cb_mcn = Callbacks.monitor_courant_numbers(
        Settings.monitor_courant_numbers,
        solver_config,
    )
    if !isnothing(cb_mcn)
        callbacks = (callbacks..., cb_mcn)
    end

    # checkpointing callback
    cb_checkpoint = Callbacks.checkpoint(
        Settings.checkpoint,
        Settings.checkpoint_keep_one,
        solver_config,
        Settings.checkpoint_dir,
    )
    if !isnothing(cb_checkpoint)
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
    if Settings.diagnostics != "never" && !isnothing(diagnostics_config)
        currtime = ODESolvers.gettime(solver)
        for dgngrp in diagnostics_config.groups
            dgngrp(currtime, fini = true)
        end
    end

    # fini VTK
    !isnothing(cb_vtk) && cb_vtk()

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
