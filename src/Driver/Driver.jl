using ArgParse
using CUDAapi
using Dates
using LinearAlgebra
using Logging
using MPI
using Printf
using Requires

using ..Atmos
using ..ColumnwiseLUSolver
using ..ConfigTypes
using ..Courant
using ..Diagnostics
using ..DGmethods
using ..DGmethods: vars_state, vars_aux, update_aux!
using ..DGmethods.NumericalFluxes
using ..GenericCallbacks
using ..HydrostaticBoussinesq
using ..Mesh.Grids
using ..Mesh.Topologies
using ..MoistThermodynamics
using ..MPIStateArrays
using ..ODESolvers
using ..PlanetParameters
using ..TicToc
using ..VariableTemplates
using ..VTK

# Note that the initial values specified here are overwritten by the
# command line argument defaults in `parse_commandline()`.
Base.@kwdef mutable struct CLIMA_Settings
    disable_gpu::Bool = false
    mpi_knows_cuda::Bool = false
    show_updates::Bool = true
    update_interval::Int = 60
    enable_diagnostics::Bool = false
    diagnostics_interval::Int = 10000
    enable_vtk::Bool = false
    vtk_interval::Int = 10000
    monitor_courant_numbers::Bool = false
    monitor_courant_interval::Int = 10
    log_level::String = "INFO"
    output_dir::String = "output"
    integration_testing::Bool = false
    array_type
end

const Settings = CLIMA_Settings(array_type = Array)

array_type() = Settings.array_type

const cuarray_pkgid =
    Base.PkgId(Base.UUID("3a865a2d-5b23-5a0f-bc46-62713ec82fae"), "CuArrays")

@init begin
    if get(ENV, "CLIMA_GPU", "") != "false" && CUDAapi.has_cuda_gpu()
        CuArrays = Base.require(cuarray_pkgid)
    end
end

function _init_array(::Type{Array})
    return nothing
end

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays, .CuArrays.CUDAdrv, .CuArrays.CUDAnative
    function _init_array(::Type{CuArray})
        comm = MPI.COMM_WORLD
        # allocate GPUs among MPI ranks
        local_comm = MPI.Comm_split_type(
            comm,
            MPI.MPI_COMM_TYPE_SHARED,
            MPI.Comm_rank(comm),
        )
        # we intentionally oversubscribe GPUs for testing: may want to disable this for production
        CUDAnative.device!(MPI.Comm_rank(local_comm) % length(devices()))
        CuArrays.allowscalar(false)
        return nothing
    end
end

function gpu_allowscalar(val)
    if haskey(Base.loaded_modules, CLIMA.cuarray_pkgid)
        Base.loaded_modules[CLIMA.cuarray_pkgid].allowscalar(val)
    end
    return nothing
end

"""
    parse_commandline()
"""
function parse_commandline()
    exc_handler =
        isinteractive() ? ArgParse.debug_handler : ArgParse.default_handler
    s = ArgParseSettings(exc_handler = exc_handler)

    @add_arg_table! s begin
        "--disable-gpu"
        help = "do not use the GPU"
        action = :store_true
        "--mpi-knows-cuda"
        help = "MPI is CUDA-enabled"
        action = :store_true
        "--no-show-updates"
        help = "do not show simulation updates"
        action = :store_true
        "--update-interval"
        help = "interval in seconds for showing simulation updates"
        arg_type = Int
        default = 60
        "--enable-diagnostics"
        help = "enable the collection of diagnostics to <output-dir>"
        action = :store_true
        "--diagnostics-interval"
        help = "override the interval for gathering diagnostics (in simulation steps)"
        arg_type = Int
        default = 10000
        "--enable-vtk"
        help = "output VTK to <output-dir> every <vtk-interval> simulation steps"
        action = :store_true
        "--vtk-interval"
        help = "interval in simulation steps for VTK output"
        arg_type = Int
        default = 10000
        "--monitor-courant-numbers"
        help = "output acoustic, advective, and diffusive Courant numbers"
        action = :store_true
        "--monitor-courant-interval"
        help = "interval in Courant number calculations"
        arg_type = Int
        default = 10
        "--log-level"
        help = "set the log level to one of debug/info/warn/error"
        arg_type = String
        default = "info"
        "--output-dir"
        help = "directory for output data"
        arg_type = String
        default = "output"
        "--integration-testing"
        help = "enable integration testing"
        action = :store_true
    end


    return parse_args(s)
end

"""
    CLIMA.init(; disable_gpu=false)

Initialize MPI, allocate GPUs among MPI ranks if using GPUs, parse command
line arguments for CLIMA, and return a Dict of any additional command line
arguments.
"""
function init(; disable_gpu = false)
    # initialize MPI
    if !MPI.Initialized()
        MPI.Init()
    end

    # set up timing mechanism
    tictoc()

    # parse command line arguments
    try
        parsed_args = parse_commandline()
        Settings.disable_gpu = disable_gpu || parsed_args["disable-gpu"]
        Settings.mpi_knows_cuda = parsed_args["mpi-knows-cuda"]
        Settings.show_updates = !parsed_args["no-show-updates"]
        Settings.update_interval = parsed_args["update-interval"]
        Settings.enable_diagnostics = parsed_args["enable-diagnostics"]
        Settings.diagnostics_interval = parsed_args["diagnostics-interval"]
        Settings.enable_vtk = parsed_args["enable-vtk"]
        Settings.vtk_interval = parsed_args["vtk-interval"]
        Settings.output_dir = parsed_args["output-dir"]
        Settings.monitor_courant_numbers =
            parsed_args["monitor-courant-numbers"]
        Settings.monitor_courant_interval =
            parsed_args["monitor-courant-interval"]
        Settings.integration_testing = parsed_args["integration-testing"]
        Settings.log_level = uppercase(parsed_args["log-level"])
    catch
        Settings.disable_gpu = disable_gpu
    end

    # set up the array type appropriately depending on whether we're using GPUs
    if !Settings.disable_gpu &&
       get(ENV, "CLIMA_GPU", "") != "false" &&
       CUDAapi.has_cuda_gpu()
        atyp = CuArrays.CuArray
    else
        atyp = Array
    end
    _init_array(atyp)
    Settings.array_type = atyp

    # create the output directory if needed
    if Settings.enable_diagnostics || Settings.enable_vtk
        mkpath(Settings.output_dir)
    end

    # set up logging
    loglevel = Settings.log_level == "DEBUG" ? Logging.Debug :
        Settings.log_level == "WARN" ? Logging.Warn :
        Settings.log_level == "ERROR" ? Logging.Error : Logging.Info
    # TODO: write a better MPI logging back-end and also integrate Dlog for large scale
    logger_stream = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    return nothing
end

include("driver_configs.jl")
include("solver_configs.jl")
include("diagnostics_configs.jl")

"""
    CLIMA.invoke!(solver_config::SolverConfiguration;
                  diagnostics_config       = nothing,
                  user_callbacks           = (),
                  check_euclidean_distance = false,
                  adjustfinalstep          = false,
                  user_info_callback       = (init)->nothing)

Run the simulation defined by the `solver_config`.

Keyword Arguments:

The `user_callbacks` are passed to the ODE solver as callback functions; see
[`ODESolvers.solve!]@ref().

If `check_euclidean_distance` is `true, then the Euclidean distance between the
final solution and initial condition function evaluated with
`solver_config.timeend` is reported.

The value of 'adjustfinalstep` is passed to the ODE solver; see
[`ODESolvers.solve!]@ref().

The function `user_info_callback` is called after the default info callback
(which is called every `Settings.update_interval` seconds of wallclock time).
The single input argument `init` is `true` when the callback is called
called for initialization before time stepping begins and `false` when called
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
    bl = dg.balancelaw
    Q = solver_config.Q
    FT = eltype(Q)
    timeend = solver_config.timeend
    init_on_cpu = solver_config.init_on_cpu
    init_args = solver_config.init_args
    solver = solver_config.solver

    # set up callbacks
    callbacks = ()
    if Settings.show_updates
        # set up the information callback
        upd_starttime = Ref(now())
        cbinfo = GenericCallbacks.EveryXWallTimeSeconds(
            Settings.update_interval,
            mpicomm,
        ) do (init = false)
            if init
                upd_starttime[] = now()
            else
                runtime = Dates.format(
                    convert(Dates.DateTime, Dates.now() - upd_starttime[]),
                    Dates.dateformat"HH:MM:SS",
                )
                energy = norm(Q)
                @info @sprintf(
                    """Update
                    simtime = %8.2f / %8.2f
                    runtime = %s
                    norm(Q) = %.16e""",
                    ODESolvers.gettime(solver),
                    solver_config.timeend,
                    runtime,
                    energy
                )
                isnan(energy) && error("norm(Q) is NaN")
            end
            user_info_callback(init)
        end
        callbacks = (callbacks..., cbinfo)
    end

    dia_starttime = ""
    if Settings.enable_diagnostics && diagnostics_config !== nothing
        dia_starttime = replace(string(now()), ":" => ".")
        Diagnostics.init(mpicomm, dg, Q, dia_starttime, Settings.output_dir)

        # set up a callback for each diagnostics group
        diacbs = ()
        for diagrp in diagnostics_config.groups
            if Settings.diagnostics_interval > 0
                interval = Settings.diagnostics_interval
            else
                interval = diagrp.interval
            end
            fn = GenericCallbacks.EveryXSimulationSteps(
                interval,
            ) do (init = false)
                currtime = ODESolvers.gettime(solver)
                if init
                    diagrp(currtime, init = true)
                end
                @info @sprintf(
                    """Diagnostics: %s
                    collecting at %s""",
                    diagrp.name,
                    string(currtime)
                )
                diagrp(currtime)
                nothing
            end
            diacbs = (diacbs..., fn)
        end
        callbacks = (callbacks..., diacbs...)
    end

    if Settings.enable_vtk
        # set up VTK output callback
        step = [0]
        cbvtk =
            GenericCallbacks.EveryXSimulationSteps(Settings.vtk_interval) do (
                init = false
            )
                vprefix = @sprintf(
                    "%s_mpirank%04d_step%04d",
                    solver_config.name,
                    MPI.Comm_rank(mpicomm),
                    step[1]
                )
                outprefix = joinpath(Settings.output_dir, vprefix)
                statenames = flattenednames(vars_state(bl, FT))
                auxnames = flattenednames(vars_aux(bl, FT))
                writevtk(outprefix, Q, dg, statenames, dg.auxstate, auxnames)
                # Generate the pvtu file for these vtk files
                if MPI.Comm_rank(mpicomm) == 0
                    # name of the pvtu file
                    pprefix =
                        @sprintf("%s_step%04d", solver_config.name, step[1])
                    pvtuprefix = joinpath(Settings.output_dir, pprefix)
                    # name of each of the ranks vtk files
                    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                        @sprintf(
                            "%s_mpirank%04d_step%04d",
                            solver_config.name,
                            i - 1,
                            step[1]
                        )
                    end
                    writepvtu(
                        pvtuprefix,
                        prefixes,
                        (statenames..., auxnames...),
                    )
                end
                step[1] += 1
                nothing
            end
        callbacks = (callbacks..., cbvtk)
    end

    if Settings.monitor_courant_numbers
        # set up the callback for Courant number calculations
        cbcfl =
            GenericCallbacks.EveryXSimulationSteps(Settings.monitor_courant_interval) do (
                init = false
            )
                simtime = ODESolvers.gettime(solver)
                Δt = solver_config.dt
                c_v = DGmethods.courant(
                    nondiffusive_courant,
                    solver_config;
                    direction = VerticalDirection(),
                )
                c_h = DGmethods.courant(
                    nondiffusive_courant,
                    solver_config;
                    direction = HorizontalDirection(),
                )
                ca_v = DGmethods.courant(
                    advective_courant,
                    solver_config;
                    direction = VerticalDirection(),
                )
                ca_h = DGmethods.courant(
                    advective_courant,
                    solver_config;
                    direction = HorizontalDirection(),
                )
                cd_v = DGmethods.courant(
                    diffusive_courant,
                    solver_config;
                    direction = VerticalDirection(),
                )
                cd_h = DGmethods.courant(
                    diffusive_courant,
                    solver_config;
                    direction = HorizontalDirection(),
                )
                @info @sprintf """
                ================================================
                Courant numbers at simtime: %8.2f
                Δt = %8.2f s

                ------------------------------------------------
                Acoustic (vertical) Courant number    = %.2g
                Acoustic (horizontal) Courant number  = %.2g
                ------------------------------------------------
                Advection (vertical) Courant number   = %.2g
                Advection (horizontal) Courant number = %.2g
                ------------------------------------------------
                Diffusion (vertical) Courant number   = %.2g
                Diffusion (horizontal) Courant number = %.2g
                ================================================
                """ simtime Δt c_v c_h ca_v ca_h cd_v cd_h
                return nothing
            end
        callbacks = (callbacks..., cbcfl)
    end

    callbacks = (callbacks..., user_callbacks...)

    # initial condition norm
    eng0 = norm(Q)
    @info @sprintf(
        """Starting %s
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

    # fini diagnostics groups
    if Settings.enable_diagnostics
        currtime = ODESolvers.gettime(solver)
        for diagrp in diagnostics_config.groups
            diagrp(currtime, fini = true)
        end
    end

    engf = norm(Q)
    @info @sprintf(
        """Finished
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
            """Euclidean distance
            norm(Q - Qe)            = %.16e
            norm(Q - Qe) / norm(Qe) = %.16e""",
            errf,
            errf / engfe
        )
    end

    return engf / eng0
end
