using Base.Threads

using ArgParse
using CUDA
using Dates
using LinearAlgebra
using Logging
using MPI
using Printf
using Random

using CLIMAParameters

using ..Atmos
using ..Callbacks
using ..Checkpoint
using ..SystemSolvers
using ..ConfigTypes
using ..Diagnostics
using ..DGMethods
using ..BalanceLaws
using ..DGMethods: remainder_DGModel, SpaceDiscretization
using ..DGMethods.NumericalFluxes
using ..DGMethods.FVReconstructions

using ..Mesh.Grids
using ..Mesh.Topologies
using ..Mesh.Filters
using ..Thermodynamics
using ..MPIStateArrays
using ..ODESolvers
using ..TicToc
using ..VariableTemplates
using ..VTK

function _init_array(::Type{CuArray})
    comm = MPI.COMM_WORLD
    # allocate GPUs among MPI ranks
    local_comm =
        MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    # we intentionally oversubscribe GPUs for testing: may want to disable this for production
    CUDA.device!(MPI.Comm_rank(local_comm) % length(devices()))
    CUDA.allowscalar(false)
    return nothing
end

_init_array(::Type{Array}) = nothing

function gpu_allowscalar(val::Bool)
    CUDA.allowscalar(val)
    return
end

# Note that the initial values specified here are overwritten by the
# command line argument defaults in `parse_commandline()`.
Base.@kwdef mutable struct ClimateMachine_Settings
    disable_gpu::Bool = false
    show_updates::String = "60secs"
    diagnostics::String = "never"
    vtk::String = "never"
    vtk_number_sample_points::Int = 0
    monitor_timestep_duration::String = "never"
    monitor_courant_numbers::String = "never"
    adapt_timestep::String = "never"
    checkpoint::String = "never"
    checkpoint_keep_one::Bool = true
    checkpoint_at_end::Bool = false
    checkpoint_dir::String = "checkpoint"
    restart_from_num::Int = -1
    fix_rng_seed::Bool = false
    log_level::String = "INFO"
    disable_custom_logger::Bool = false
    output_dir::String = "output"
    debug_init::Bool = false
    integration_testing::Bool = false
    array_type::Type = Array
    sim_time::Float64 = NaN
    fixed_number_of_steps::Int = -1
    degree::NTuple{2, Int} = (-1, -1)
    cutoff_degree::NTuple{2, Int} = (-1, -1)
    nelems::NTuple{3, Int} = (-1, -1, -1)
    domain_height::Float64 = -1
    resolution::NTuple{3, Float64} = (-1, -1, -1)
    domain_min::NTuple{3, Float64} = (-1, -1, -1)
    domain_max::NTuple{3, Float64} = (-1, -1, -1)
end

const Settings = ClimateMachine_Settings()

"""
    ClimateMachine.array_type()

Return the array type used by ClimateMachine. This defaults to (CPU-based)
`Array` and is only correctly set (based on choice from the command
line, from an environment variable, or from experiment code) after
`ClimateMachine.init()` is called.
"""
array_type() = Settings.array_type

"""
    get_setting(setting_name::Symbol, settings, defaults)

Define fallback behavior for driver settings, first accessing overloaded
`settings` if defined, followed by constructed global ENV variable
`CLIMATEMACHINE_SETTINGS_<SETTING_NAME>`, then (global) defaults.

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
        if setting_type == String
            v = env_val
        elseif setting_type == NTuple{2, Int} ||
               setting_type == NTuple{3, Float64}
            v = ArgParse.parse_item(setting_type, env_val)
        else
            v = tryparse(setting_type, env_val)
        end
        if isnothing(v)
            error("cannot parse ENV $setting_env value $env_val, to setting type $setting_type")
        end
        return v
    elseif haskey(defaults, setting_name)
        return defaults[setting_name]
    else
        error("setting $setting_name is not contained in either settings or defaults")
    end
end

function get_gpu_setting(setting_name::Symbol, settings, defaults)
    # do not override disable_gpu keyword argument setting if it exists
    if !haskey(settings, setting_name)
        # if old GPU ENV keyword exists, overwrite the settings variable if not defined
        if haskey(ENV, "CLIMATEMACHINE_GPU")
            settings[setting_name] = ENV["CLIMATEMACHINE_GPU"] == "false"
        end
    end
    # fallback behavior
    return get_setting(setting_name, settings, defaults)
end

"""
    parse_commandline(
        defaults::Dict{Symbol, Any},
        global_defaults::Dict{Symbol, Any},
        custom_clargs::Union{Nothing, ArgParseSettings} = nothing,
    )

Parse process command line ARGS values. If `defaults` is specified, it
overrides default values for command line argument defaults. If
`custom_clargs` is specified, it is added to the parsing step.

Returns a `Dict` containing parsed process ARGS values.
"""
function parse_commandline(
    defaults::Dict{Symbol, Any},
    global_defaults::Dict{Symbol, Any},
    custom_clargs::Union{Nothing, ArgParseSettings} = nothing,
)
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
        autofix_names = true,     # switches --flag-name to 'flag_name'
        error_on_conflict = true, # don't allow custom_clargs' settings to override these
    )
    add_arg_group!(s, "ClimateMachine")

    @add_arg_table! s begin
        "--disable-gpu"
        help = "do not use the GPU"
        action = :store_const
        constant = true
        default = get_gpu_setting(:disable_gpu, defaults, global_defaults)
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
        "--vtk-number-sample-points"
        help = "number of sampling points in each element for VTK output"
        metavar = "<number>"
        arg_type = Int
        default =
            get_setting(:vtk_number_sample_points, defaults, global_defaults)
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
        "--adapt-timestep"
        help = "interval at which to update the timestep"
        metavar = "<interval>"
        arg_type = String
        default = get_setting(:adapt_timestep, defaults, global_defaults)
        "--checkpoint"
        help = "interval at which to create a checkpoint"
        metavar = "<interval>"
        arg_type = String
        default = get_setting(:checkpoint, defaults, global_defaults)
        "--checkpoint-keep-all"
        help = "keep all checkpoints (instead of just the most recent)"
        action = :store_const
        constant = true
        default = !get_setting(:checkpoint_keep_one, defaults, global_defaults)
        "--checkpoint-at-end"
        help = "create a checkpoint at the end of the simulation"
        action = :store_const
        constant = true
        default = get_setting(:checkpoint_at_end, defaults, global_defaults)
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
        action = :store_const
        constant = true
        default = get_setting(:fix_rng_seed, defaults, global_defaults)
        "--disable-custom-logger"
        help = "do not use a custom logger"
        action = :store_const
        constant = true
        default = get_setting(:disable_custom_logger, defaults, global_defaults)
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
        "--debug-init"
        help = "fill state arrays with NaNs and dump them post-initialization"
        action = :store_const
        constant = true
        default = get_setting(:debug_init, defaults, global_defaults)
        "--integration-testing"
        help = "enable integration testing"
        action = :store_const
        constant = true
        default = get_setting(:integration_testing, defaults, global_defaults)
        "--sim-time"
        help = "run for the specified time (in simulation seconds)"
        metavar = "<number>"
        arg_type = Float64
        default = get_setting(:sim_time, defaults, global_defaults)
        "--fixed-number-of-steps"
        help = "if `≥0` perform specified number of steps"
        metavar = "<number>"
        arg_type = Int
        default = get_setting(:fixed_number_of_steps, defaults, global_defaults)
        "--degree"
        help = "tuple of horizontal and vertical polynomial degrees for spatial discretization order (no space before/after comma)"
        metavar = "<horizontal>,<vertical>"
        arg_type = NTuple{2, Int}
        default = get_setting(:degree, defaults, global_defaults)
        "--cutoff-degree"
        help = "tuple of horizontal and vertical polynomial degrees for cutoff filter (no space before/after comma)"
        metavar = "<horizontal>,<vertical>"
        arg_type = NTuple{2, Int}
        default = get_setting(:cutoff_degree, defaults, global_defaults)
        "--nelems"
        help = "number of elements in each direction: 3 for Ocean GCM, 2 for Atmos GCM or 1 for Atmos single-stack (no space before/after comma)"
        metavar = "<nelem_1>[,<nelem_2>[,<nelem_3>]]"
        arg_type = NTuple{3, Int}
        default = get_setting(:nelems, defaults, global_defaults)
        "--domain-height"
        help = "domain height (in meters) for GCM or single-stack configurations"
        metavar = "<number>"
        arg_type = Float64
        default = get_setting(:domain_height, defaults, global_defaults)
        "--resolution"
        help = "tuple of three element resolutions (in meters) for LES and MultiColumnLandModel configurations"
        metavar = "<Δx>,<Δy>,<Δz>"
        arg_type = NTuple{3, Float64}
        default = get_setting(:resolution, defaults, global_defaults)
        "--domain-min"
        help = "tuple of three minima for the domain size (in meters) for LES and MultiColumnLandModel configurations"
        metavar = "<xmin>,<ymin>,<zmin>"
        arg_type = NTuple{3, Float64}
        default = get_setting(:domain_min, defaults, global_defaults)
        "--domain-max"
        help = "tuple of three maxima for the domain size (in meters) for LES and MultiColumnLandModel configurations"
        metavar = "<xmax>,<ymax>,<zmax>"
        arg_type = NTuple{3, Float64}
        default = get_setting(:domain_max, defaults, global_defaults)
    end
    # add custom cli argparse settings if provided
    if !isnothing(custom_clargs)
        import_settings!(s, custom_clargs)
    end
    return parse_args(s)
end

"""
    ClimateMachine.init(;
        parse_clargs::Bool = false,
        custom_clargs::Union{Nothing, ArgParseSettings} = nothing,
        init_driver::Bool = true,
        keyword_args...,
    )

Initialize the ClimateMachine. If `parse_clargs` is set, parse command line
arguments (additional driver-specific arguments can be added by specifying
`custom_clargs`).

Setting `init_driver = false` will set up the `ClimateMachine.Settings`
singleton values without initializing the ClimateMachine runtime. Otherwise,
the runtime will be initialized (see `init_runtime()`).

Finally, key-value pairs can be supplied to `ClimateMachine.init()` to set
system default settings -- the final settings are decided as follows (in
order of precedence):
1. Command line arguments (if `parse_clargs = true`).
2. Environment variables.
3. Keyword arguments to `init()`.
4. Defaults (in `ClimateMachine_Settings`).

Recognized keyword arguments are:
- `disable_gpu::Bool = false`:
        do not use the GPU
- `show_updates::String = "60secs"`:
        interval at which to show simulation updates
- `diagnostics::String = "never"`:
        interval at which to collect diagnostics"
- `vtk::String = "never"`:
        interval at which to write simulation vtk output
- `vtk-number-sample-points::Int` = 0:
        the number of sampling points in each element for VTK output
- `monitor_timestep_duration::String = "never"`:
        interval in time-steps at which to output wall-clock time per time-step
- `monitor_courant_numbers::String = "never"`:
        interval at which to output acoustic, advective, and diffusive Courant numbers
- `adapt-timestep::String = "never"`:
        interval at which to update the timestep
- `checkpoint::String = "never"`:
        interval at which to write a checkpoint
- `checkpoint_keep_one::Bool = true`: (interval)
        keep all checkpoints (instead of just the most recent)"
- `checkpoint_at_end::Bool = false`:
        create a checkpoint at the end of the simulation
- `checkpoint_dir::String = "checkpoint"`:
        absolute or relative path to checkpoint directory
- `restart_from_num::Int = -1`:
        checkpoint number from which to restart (in `checkpoint_dir`)
- `fix_rng_seed::Bool = false`:
        set RNG seed to a fixed value for reproducibility
- `log_level::String = "INFO"`:
        log level for ClimateMachine global default runtime logger
- `disable_custom_logger::String = false`:
        disable using a global custom logger for ClimateMachine
- `output_dir::String = "output"`: (path)
        absolute or relative path to output data directory
- `debug_init::Bool = false`:
        fill state arrays with NaNs and dump them post-initialization
- `integration_testing::Bool = false`:
        enable integration_testing
- `sim_time::Float64 = NaN`:
        run for the specified time (in simulation seconds)
- `fixed_number_of_steps::Int = -1`:
        if `≥0` perform specified number of steps
- `degree::NTuple{2, Int} = (-1, -1)`:
        tuple of horizontal and vertical polynomial degrees for spatial discretization order
- `cutoff_degree::NTuple{2, Int} = (-1, -1)`:
        tuple of horizontal and vertical polynomial degrees for cutoff filter
- `nelems::NTuple{3, Int} = (-1, -1, -1)`:
        tuple of number of elements in each direction: 3 for Ocean, 2 for GCM or 1 for single-stack
- `domain_height::Float64 = -1`:
        domain height (in meters) for GCM or single-stack configurations
- `resolution::NTuple{3, Float64} = (-1, -1, -1)`:
        tuple of three element resolutions (in meters) for LES and MultiColumnLandModel configurations
- `domain_min::NTuple{3, Float64} = (-1, -1, -1)`:
        tuple of three minima for the domain size (in meters) for LES and MultiColumnLandModel configurations
- `domain_max::NTuple{3, Float64} = (-1, -1, -1)`:
        tuple of three maxima for the domain size (in meters) for LES and MultiColumnLandModel configurations

Returns `nothing`, or if `parse_clargs = true`, returns parsed command line
arguments.
"""
function init(;
    parse_clargs::Bool = false,
    custom_clargs::Union{Nothing, ArgParseSettings} = nothing,
    init_driver::Bool = true,
    keyword_args...,
)
    # `Settings` contains global defaults
    global_defaults = Dict{Symbol, Any}(
        (n, getproperty(Settings, n)) for n in propertynames(Settings)
    )

    # keyword arguments must be applicable to `Settings`
    all_args = Dict{Symbol, Any}(keyword_args)
    for kwarg in keys(all_args)
        if get(global_defaults, kwarg, nothing) === nothing
            throw(ArgumentError(string(kwarg)))
        end
    end

    # if command line arguments should be processed, do so and override
    # keyword arguments
    cl_args = nothing
    if parse_clargs
        cl_args = parse_commandline(all_args, global_defaults, custom_clargs)

        # We need to munge the parsed arg dict a bit as parsed arg keys
        # and initialization keywords are not 1:1
        cl_args["checkpoint_keep_one"] = !cl_args["checkpoint_keep_all"]

        # parsed command line arguments override keyword arguments
        all_args = merge(
            all_args,
            Dict{Symbol, Any}((Symbol(k), v) for (k, v) in cl_args),
        )
    end

    # TODO: also add validation for initialization values

    # Here, `all_args` contains command line arguments and keyword arguments.
    # They must be applied to `Settings`.
    #
    # special cases for backward compatibility
    if haskey(all_args, :disable_gpu)
        Settings.disable_gpu = all_args[:disable_gpu]
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

    if haskey(all_args, :output_dir)
        Settings.output_dir = all_args[:output_dir]
    elseif haskey(ENV, "CLIMATEMACHINE_OUTPUT_DIR")
        @warn(
            "CLIMATEMACHINE_OUTPUT_DIR will be deprecated; " *
            "use CLIMATEMACHINE_SETTINGS_OUTPUT_DIR"
        )
        Settings.output_dir = ENV["CLIMATEMACHINE_OUTPUT_DIR"]
    elseif haskey(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR")
        Settings.output_dir = ENV["CLIMATEMACHINE_SETTINGS_OUTPUT_DIR"]
    end

    # all other settings
    for n in propertynames(Settings)
        # skip over the special backwards compat cases defined above
        if n == :disable_gpu || n == :output_dir
            continue
        end
        setproperty!(Settings, n, get_setting(n, all_args, global_defaults))
    end

    # set up the array type appropriately depending on whether we're using GPUs
    if !Settings.disable_gpu && CUDA.has_cuda_gpu()
        Settings.array_type = CUDA.CuArray
    else
        Settings.array_type = Array
    end

    # initialize the runtime
    if init_driver
        init_runtime(Settings)
    end
    return cl_args
end

"""
    init_runtime(settings::ClimateMachine_Settings)

Initialize the ClimateMachine runtime: initialize MPI, set the default
array type, set RNG seeds for all threads, create output directory and
set up logging.
"""
function init_runtime(settings::ClimateMachine_Settings)
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

    # TODO: write a better MPI logging back-end and also integrate Dlog
    # for large scale

    # set up logging
    log_level_str = uppercase(settings.log_level)
    loglevel =
        log_level_str == "DEBUG" ? Logging.Debug :
        log_level_str == "WARN" ? Logging.Warn :
        log_level_str == "ERROR" ? Logging.Error : Logging.Info
    if !settings.disable_custom_logger
        # cannot use `NullLogger` here because MPI collectives may be
        # used in logging calls!
        logger_stream = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? stderr : devnull
        prev_logger = global_logger(ConsoleLogger(logger_stream, loglevel))
        atexit() do
            global_logger(prev_logger)
        end
    end
    return
end

include("driver_configs.jl")
include("solver_configs.jl")
include("diagnostics_configs.jl")

"""
    ClimateMachine.ConservationCheck

Pass a tuple of these to `ClimateMachine.invoke!` to perform a
conservation check of each `varname` at the specified `interval`. This
computes `Σv = weightedsum(Q.varname)` and `δv = (Σv - Σv₀) / Σv`.
`invoke!` throws an error if `abs(δv)` exceeds `error_threshold. If
`show`, `δv` is displayed.
"""
struct ConservationCheck{FT}
    varname::String
    interval::String
    error_threshold::FT
    show::Bool
end
ConservationCheck(varname::String, interval::String) =
    ConservationCheck(varname, interval, Inf, true)
ConservationCheck(
    varname::String,
    interval::String,
    error_threshold::FT,
) where {FT} = ConservationCheck(varname, interval, error_threshold, true)

"""
    ClimateMachine.invoke!(
        solver_config::SolverConfiguration;
        adjustfinalstep = false,
        diagnostics_config = nothing,
        user_callbacks = (),
        user_info_callback = () -> nothing,
        check_cons = (),
        check_euclidean_distance = false,
    )

Run the simulation defined by `solver_config`.

Keyword Arguments:

The value of 'adjustfinalstep` is passed to the ODE solver; see
[`solve!`](@ref ODESolvers.solve!).

The `user_callbacks` are passed to the ODE solver as callback functions;
see [`solve!`](@ref ODESolvers.solve!).

The function `user_info_callback` is called after the default info
callback (which is called every `Settings.show_updates` interval). The
single input argument `init` is `true` when the callback is called for
initialization (before time stepping begins) and `false` when called
during the actual ODE solve; see [`GenericCallbacks`](@ref) and
[`solve!`](@ref ODESolvers.solve!).

If conservation checks are to be performed, `check_cons` must be a
tuple of [`ConservationCheck`](@ref).

If `check_euclidean_distance` is `true, then the Euclidean distance
between the final solution and initial condition function evaluated with
`solver_config.timeend` is reported.
"""
function invoke!(
    solver_config::SolverConfiguration;
    adjustfinalstep = false,
    diagnostics_config = nothing,
    user_callbacks = (),
    user_info_callback = () -> nothing,
    check_cons = (),
    check_euclidean_distance = false,
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
        Diagnostics.init(
            mpicomm,
            solver_config.param_set,
            dg,
            Q,
            dgn_starttime,
            Settings.output_dir,
        )

        dgncbs = Callbacks.diagnostics(
            Settings.diagnostics,
            solver_config,
            dgn_starttime,
            diagnostics_config,
        )
        callbacks = (callbacks..., dgncbs...)
    end

    # vtk callback
    cb_vtk = Callbacks.vtk(
        Settings.vtk,
        solver_config,
        Settings.output_dir,
        Settings.vtk_number_sample_points,
    )
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

    # Timestep adapter
    cb_adp = Callbacks.adapt_timestep(Settings.adapt_timestep, solver_config)
    if !isnothing(cb_adp)
        callbacks = (callbacks..., cb_adp)
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

    # conservation callbacks
    cccbs = Callbacks.check_cons(check_cons, solver_config)
    callbacks = (callbacks..., cccbs...)

    # user callbacks
    callbacks = (user_callbacks..., callbacks...)

    # initial condition norm
    eng0 = norm(Q)
    @info @sprintf(
        """
        %s %s
            dt              = %.5e
            timeend         = %8.2f
            number of steps = %d
            norm(Q)         = %.16e""",
        Settings.restart_from_num > 0 ? "Restarting" : "Starting",
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
