using ArgParse
using CUDAapi
using Dates
using LinearAlgebra
using Logging
using MPI
using Printf
using Requires

using ..AdditiveRungeKuttaMethod
using ..Atmos
using ..VTK
using ..ColumnwiseLUSolver
using ..Diagnostics
using ..GenericCallbacks
using ..LowStorageRungeKuttaMethod
using ..Mesh.Grids: EveryDirection, VerticalDirection, HorizontalDirection
using ..MoistThermodynamics
using ..MPIStateArrays
using ..TicToc

Base.@kwdef mutable struct CLIMA_Settings
    disable_gpu::Bool = false
    mpi_knows_cuda::Bool = false
    show_updates::Bool = true
    update_interval::Int = 60
    enable_diagnostics::Bool = true
    diagnostics_interval::Int = 10000
    enable_vtk::Bool = false
    vtk_interval::Int = 10000
    log_level::String = "INFO"
    output_dir::String = "output"
    integration_testing::Bool = false
    array_type
end

const Settings = CLIMA_Settings(array_type = Array)

array_type() = Settings.array_type

include("Configurations.jl")

const cuarray_pkgid = Base.PkgId(Base.UUID("3a865a2d-5b23-5a0f-bc46-62713ec82fae"), "CuArrays")

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
        local_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED,  MPI.Comm_rank(comm))
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
    exc_handler = isinteractive() ? ArgParse.debug_handler : ArgParse.default_handler
    s = ArgParseSettings(exc_handler=exc_handler)

    @add_arg_table s begin
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
        "--disable-diagnostics"
            help = "disable the collection of diagnostics to <output-dir>"
            action = :store_true
        "--diagnostics-interval"
            help = "interval in simulation steps for gathering diagnostics"
            arg_type = Int
            default = 10000
        "--enable-vtk"
            help = "output VTK to <output-dir> every <vtk-interval> simulation steps"
            action = :store_true
        "--vtk-interval"
            help = "interval in simulation steps for VTK output"
            arg_type = Int
            default = 10000
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
function init(; disable_gpu=false)
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
        Settings.enable_diagnostics = !parsed_args["disable-diagnostics"]
        Settings.diagnostics_interval = parsed_args["diagnostics-interval"]
        Settings.enable_vtk = parsed_args["enable-vtk"]
        Settings.vtk_interval = parsed_args["vtk-interval"]
        Settings.output_dir = parsed_args["output-dir"]
        Settings.integration_testing = parsed_args["integration-testing"]
        Settings.log_level = uppercase(parsed_args["log-level"])
    catch
        Settings.disable_gpu = disable_gpu
    end

    # set up the array type appropriately depending on whether we're using GPUs
    if !Settings.disable_gpu && get(ENV, "CLIMA_GPU", "") != "false" && CUDAapi.has_cuda_gpu()
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
        Settings.log_level == "WARN"  ? Logging.Warn  :
        Settings.log_level == "ERROR" ? Logging.Error : Logging.Info
    # TODO: write a better MPI logging back-end and also integrate Dlog for large scale
    logger_stream = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    return nothing
end

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
    forcecpu::Bool
    init_args
    solver
end

"""
    CLIMA.setup_solver(t0, timeend, driver_config)

Set up the DG model per the specified driver configuration and set up the ODE solver.
"""
function setup_solver(t0::FT, timeend::FT,
                      driver_config::DriverConfiguration,
                      init_args...;
                      forcecpu=false,
                      ode_solver_type=nothing,
                      Courant_number=0.4,
                      T=FT(290)
                     ) where {FT<:AbstractFloat}
    @tic setup_solver

    # create DG model, initialize ODE state
    dg = DGModel(driver_config.bl, driver_config.grid, driver_config.numfluxnondiff,
                 driver_config.numfluxdiff, driver_config.gradnumflux)
    @info @sprintf("Initializing %s", driver_config.name)
    Q = init_ode_state(dg, FT(0), init_args...; forcecpu=forcecpu)

    # if solver has been specified, use it
    if ode_solver_type !== nothing
        solver_type = ode_solver_type
    else
        solver_type = driver_config.solver_type
    end

    if isa(solver_type, ExplicitSolverType)

        dt = Courant_number * min_node_distance(dg.grid, VerticalDirection()) / soundspeed_air(T)
        numberofsteps = convert(Int64, cld(timeend, dt))
        dt = timeend / numberofsteps

        solver = solver_type.solver_method(dg, Q; dt=dt, t0=t0)

    else # solver_type === IMEXSolverType

        dt = Courant_number * min_node_distance(dg.grid, HorizontalDirection()) / soundspeed_air(T)
        numberofsteps = convert(Int64, cld(timeend, dt))
        dt = timeend / numberofsteps

        linmodel = solver_type.linear_model(driver_config.bl)

        vdg = DGModel(linmodel, driver_config.grid, driver_config.numfluxnondiff,
                      driver_config.numfluxdiff, driver_config.gradnumflux,
                      auxstate=dg.auxstate, direction=VerticalDirection())

        solver = solver_type.solver_method(dg, vdg, SingleColumnLU(), Q; dt=dt, t0=t0)
    end

    @toc setup_solver

    return SolverConfiguration(driver_config.name, driver_config.mpicomm, dg, Q,
                               t0, timeend, dt, forcecpu, init_args, solver)
end

"""
    CLIMA.invoke!(solver_config)

Run the simulation.
"""
function invoke!(solver_config::SolverConfiguration;
                 user_callbacks=(),
                 check_euclidean_distance=false,
                 adjustfinalstep=false
                )
    mpicomm = solver_config.mpicomm
    dg = solver_config.dg
    bl = dg.balancelaw
    Q = solver_config.Q
    FT = eltype(Q)
    timeend = solver_config.timeend
    forcecpu = solver_config.forcecpu
    init_args = solver_config.init_args
    solver = solver_config.solver

    # set up callbacks
    callbacks = ()
    if Settings.show_updates
        # set up the information callback
        starttime = Ref(now())
        cbinfo = GenericCallbacks.EveryXWallTimeSeconds(Settings.update_interval, mpicomm) do (init=false)
            if init
                starttime[] = now()
            else
                runtime = Dates.format(convert(Dates.DateTime,
                                               Dates.now()-starttime[]),
                                       Dates.dateformat"HH:MM:SS")
                energy = norm(solver_config.Q)
                @info @sprintf("""Update
                               simtime = %.16e
                               runtime = %s
                               norm(Q) = %.16e""",
                               ODESolvers.gettime(solver),
                               runtime,
                               energy)
            end
            nothing
        end
        callbacks = (callbacks..., cbinfo)
    end
    if Settings.enable_diagnostics
        # set up diagnostics callback
        diagnostics_time_str = replace(string(now()), ":" => ".")
        cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(Settings.diagnostics_interval) do (init=false)
            sim_time_str = string(ODESolvers.gettime(solver))
            gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                               Settings.output_dir, ODESolvers.gettime(solver))
            nothing
        end
        callbacks = (callbacks..., cbdiagnostics)
    end
    if Settings.enable_vtk
        # set up VTK output callback
        step = [0]
        cbvtk = GenericCallbacks.EveryXSimulationSteps(Settings.vtk_interval) do (init=false)
            vprefix = @sprintf("%s_mpirank%04d_step%04d", solver_config.name,
                               MPI.Comm_rank(mpicomm), step[1])
            outprefix = joinpath(Settings.output_dir, vprefix)
            statenames = Atmos.flattenednames(Atmos.vars_state(bl, FT))
            auxnames = Atmos.flattenednames(Atmos.vars_aux(bl, FT))
            writevtk(outprefix, Q, dg, statenames, dg.auxstate, auxnames)
            # Generate the pvtu file for these vtk files
            if MPI.Comm_rank(mpicomm) == 0
                # name of the pvtu file
                pprefix = @sprintf("%s_step%04d", solver_config.name, step[1])
                pvtuprefix = joinpath(Settings.output_dir, pprefix)
                # name of each of the ranks vtk files
                prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                    @sprintf("%s_mpirank%04d_step%04d", solver_config.name, i-1, step[1])
                end
                writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))
            end
            step[1] += 1
            nothing
        end
        callbacks = (callbacks..., cbvtk)
    end
    callbacks = (callbacks..., user_callbacks...)

    # initial condition norm
    eng0 = norm(Q)
    @info @sprintf("""Starting %s
                   dt                      = %.5e
                   timeend                 = %.5e
                   norm(Q)                 = %.16e""",
                   solver_config.name,
                   solver_config.dt,
                   solver_config.timeend,
                   eng0)

    # run the simulation
    @tic solve!
    solve!(Q, solver; timeend=timeend, callbacks=callbacks, adjustfinalstep=adjustfinalstep)
    @toc solve!

    engf = norm(solver_config.Q)

    @info @sprintf("""Finished
                   norm(Q)                 = %.16e
                   norm(Q) / norm(Q₀)      = %.16e
                   norm(Q) - norm(Q₀)      = %.16e""",
                   engf,
                   engf/eng0,
                   engf-eng0)

    if check_euclidean_distance
        Qe = init_ode_state(dg, timeend, init_args...; forcecpu=forcecpu)
        engfe = norm(Qe)
        errf = euclidean_distance(solver_config.Q, Qe)
        @info @sprintf("""Euclidean distance
                       norm(Q - Qe)            = %.16e
                       norm(Q - Qe) / norm(Qe) = %.16e""",
                       errf,
                       errf/engfe)
    end

    return engf / eng0
end

