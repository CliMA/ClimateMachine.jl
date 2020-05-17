module Callbacks

using CUDAapi
using Dates
using KernelAbstractions
using LinearAlgebra
using MPI
using Printf
using Statistics

using CLIMAParameters
using CLIMAParameters.Planet: day

using ..Courant
using ..Checkpoint
using ..DGmethods: courant, vars_state_conservative, vars_state_auxiliary
using ..Diagnostics
using ..GenericCallbacks
using ..MPIStateArrays
using ..ODESolvers
using ..TicToc
using ..VariableTemplates
using ..VTK
using ..Mesh.Grids: HorizontalDirection, VerticalDirection

using CuArrays, CuArrays.CUDAdrv, CuArrays.CUDAnative

_sync_device(::Type{CuArray}) = synchronize()
_sync_device(::Type{Array}) = nothing

"""
    show_updates(show_updates_opt, solver_config, user_info_callback)

Return a callback function that shows simulation updates at the specified
interval and also invokes the user-specified info callback.
"""
function show_updates(show_updates_opt, solver_config, user_info_callback)
    timeend = solver_config.timeend

    cb_constr = CB_constructor(show_updates_opt, solver_config)
    if cb_constr !== nothing
        # set up the information callback
        upd_starttime = Ref(now())
        cb_updates = cb_constr() do (init = false)
            if init
                upd_starttime[] = now()
            else
                runtime = Dates.format(
                    convert(Dates.DateTime, Dates.now() - upd_starttime[]),
                    Dates.dateformat"HH:MM:SS",
                )
                energy = norm(solver_config.Q)
                @info @sprintf(
                    """
Update
    simtime = %8.2f / %8.2f
    runtime = %s
    norm(Q) = %.16e""",
                    ODESolvers.gettime(solver_config.solver),
                    solver_config.timeend,
                    runtime,
                    energy,
                )
                isnan(energy) && error("norm(Q) is NaN")
            end
            user_info_callback(init)
        end
        return cb_updates
    else
        return nothing
    end
end

"""
    diagnostics(diagnostics_opt, solver_config, dgn_starttime, diagnostics_config)

Return callback functions that shows simulation updates at the specified
interval and also invokes the user-specified info callback.
"""
function diagnostics(
    diagnostics_opt,
    solver_config,
    dgn_starttime,
    diagnostics_config,
)
    if diagnostics_opt != "never" && diagnostics_config !== nothing
        # set up a callback for each diagnostics group
        dgncbs = ()
        for dgngrp in diagnostics_config.groups
            cb_constr =
                CB_constructor(diagnostics_opt, solver_config, dgngrp.interval)
            cb_constr === nothing && continue
            fn = cb_constr() do (init = false)
                @tic diagnostics
                currtime = ODESolvers.gettime(solver_config.solver)
                @info @sprintf(
                    """
Diagnostics: %s
    %s at %s""",
                    dgngrp.name,
                    init ? "initializing" : "collecting",
                    string(currtime),
                )
                dgngrp(currtime, init = init)
                @toc diagnostics
                nothing
            end
            dgncbs = (dgncbs..., fn)
        end
        return dgncbs
    else
        return nothing
    end
end

"""
    vtk(vtk_opt, solver_config)

Return a callback that saves state and auxiliary variables to a VTK
file.
"""
function vtk(vtk_opt, solver_config, output_dir)
    cb_constr = CB_constructor(vtk_opt, solver_config)
    if cb_constr !== nothing
        vtknum = Ref(1)

        mpicomm = solver_config.mpicomm
        dg = solver_config.dg
        bl = dg.balance_law
        Q = solver_config.Q
        FT = eltype(Q)

        cb_vtk = cb_constr() do (init = false)
            @tic vtk
            vprefix = @sprintf(
                "%s_mpirank%04d_num%04d",
                solver_config.name,
                MPI.Comm_rank(mpicomm),
                vtknum[],
            )
            outprefix = joinpath(output_dir, vprefix)

            statenames = flattenednames(vars_state_conservative(bl, FT))
            auxnames = flattenednames(vars_state_auxiliary(bl, FT))

            writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)

            # Generate the pvtu file for these vtk files
            if MPI.Comm_rank(mpicomm) == 0
                # name of the pvtu file
                pprefix = @sprintf("%s_num%04d", solver_config.name, vtknum[])
                pvtuprefix = joinpath(output_dir, pprefix)

                # name of each of the ranks vtk files
                prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                    @sprintf(
                        "%s_mpirank%04d_num%04d",
                        solver_config.name,
                        i - 1,
                        vtknum[],
                    )
                end
                writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))
            end

            vtknum[] += 1
            @toc vtk
            nothing
        end
        return cb_vtk
    else
        return nothing
    end
end

"""
    monitor_timestep_duration(mtd_opt, array_type, comm)

Returns a callback function that displays wall-clock time per time-step
statistics across MPI ranks in the communicator `comm`.  The times are
averaged over the `mtd_opt` time steps requested for output.  The
`array_type` is used to synchronize the compute device.
"""
function monitor_timestep_duration(mtd_opt, array_type, comm)
    if mtd_opt == "never"
        return nothing
    elseif !endswith(mtd_opt, "steps")
        @warn @sprintf(
            """
monitor-timestep-duration must be in 'steps'; %s unrecognized; disabling""",
            mtd_opt,
        )
        return nothing
    end
    steps = parse(Int, mtd_opt[1:(end - 5)])

    _sync_device(array_type)
    before = time_ns()

    cb_mtd = GenericCallbacks.EveryXSimulationSteps(steps) do
        _sync_device(array_type)
        after = time_ns()

        time_per_timesteps = after - before

        times = MPI.Gather(time_per_timesteps, 0, comm)
        if MPI.Comm_rank(comm) == 0
            ns_per_s = 1e9
            times = times ./ ns_per_s ./ steps

            @info @sprintf(
                """Wall-clock time per time-step (statistics across MPI ranks)
                   maximum (s) = %25.16e
                   minimum (s) = %25.16e
                   median  (s) = %25.16e
                   std     (s) = %25.16e
                """,
                maximum(times),
                minimum(times),
                median(times),
                std(times),
            )
        end

        _sync_device(array_type)
        before = time_ns()

        nothing
    end

    return cb_mtd
end

"""
    monitor_courant_numbers(mcn_opt, solver_config)

Return a callback function that displays Courant numbers for the simulation
at `mcn_opt` intervals.
"""
function monitor_courant_numbers(mcn_opt, solver_config)
    cb_constr = CB_constructor(mcn_opt, solver_config)
    if cb_constr !== nothing
        cb_cfl = cb_constr() do (init = false)
            Δt = solver_config.dt
            c_v = courant(
                nondiffusive_courant,
                solver_config,
                direction = VerticalDirection(),
            )
            c_h = courant(
                nondiffusive_courant,
                solver_config,
                direction = HorizontalDirection(),
            )
            ca_v = courant(
                advective_courant,
                solver_config,
                direction = VerticalDirection(),
            )
            ca_h = courant(
                advective_courant,
                solver_config,
                direction = HorizontalDirection(),
            )
            cd_v = courant(
                diffusive_courant,
                solver_config,
                direction = VerticalDirection(),
            )
            cd_h = courant(
                diffusive_courant,
                solver_config,
                direction = HorizontalDirection(),
            )
            simtime = ODESolvers.gettime(solver_config.solver)
            @info @sprintf(
                """
Courant numbers at simtime: %8.2f, Δt = %8.2f s
    Acoustic (vertical) Courant number    = %.2g
    Acoustic (horizontal) Courant number  = %.2g
    Advection (vertical) Courant number   = %.2g
    Advection (horizontal) Courant number = %.2g
    Diffusion (vertical) Courant number   = %.2g
    Diffusion (horizontal) Courant number = %.2g""",
                simtime,
                Δt,
                c_v,
                c_h,
                ca_v,
                ca_h,
                cd_v,
                cd_h,
            )
            nothing
        end
        return cb_cfl
    else
        return nothing
    end
end

"""
    checkpoint(
        checkpoint_opt,
        checkpoint_keep_one,
        solver_config,
        checkpoint_dir,
    )

Return a callback function that runs at `checkpoint_opt` intervals
and stores a simulation checkpoint into `checkpoint_dir` identified
by a running number.
"""
function checkpoint(
    checkpoint_opt,
    checkpoint_keep_one,
    solver_config,
    checkpoint_dir,
)
    cb_constr = CB_constructor(checkpoint_opt, solver_config)
    if cb_constr !== nothing
        cpnum = Ref(1)
        cb_checkpoint = cb_constr() do (init = false)
            write_checkpoint(
                solver_config,
                checkpoint_dir,
                solver_config.name,
                solver_config.mpicomm,
                cpnum[],
            )
            if checkpoint_keep_one
                rm_checkpoint(
                    checkpoint_dir,
                    solver_config.name,
                    solver_config.mpicomm,
                    cpnum[] - 1,
                )
            end
            cpnum[] += 1
            nothing
        end
        return cb_checkpoint
    else
        return nothing
    end
end

"""
    CB_constructor(interval, solver_config, default)

Parse the specified `interval` and return the appropriate `GenericCallbacks`
constructor. If `interval` is "default", then use `default` as the interval.
"""
function CB_constructor(interval::String, solver_config, default = nothing)
    mpicomm = solver_config.mpicomm
    solver = solver_config.solver
    dg = solver_config.dg
    bl = dg.balance_law
    secs_per_day = day(bl.param_set)

    if endswith(interval, "smonths")
        ticks = 30.0 * secs_per_day * parse(Float64, interval[1:(end - 7)])
        return (func) ->
            GenericCallbacks.EveryXSimulationTime(func, ticks, solver)
    elseif endswith(interval, "sdays")
        ticks = secs_per_day * parse(Float64, interval[1:(end - 5)])
        return (func) ->
            GenericCallbacks.EveryXSimulationTime(func, ticks, solver)
    elseif endswith(interval, "shours")
        ticks = 60.0 * 60.0 * parse(Float64, interval[1:(end - 6)])
        return (func) ->
            GenericCallbacks.EveryXSimulationTime(func, ticks, solver)
    elseif endswith(interval, "smins")
        ticks = 60.0 * parse(Float64, interval[1:(end - 5)])
        return (func) ->
            GenericCallbacks.EveryXSimulationTime(func, ticks, solver)
    elseif endswith(interval, "ssecs")
        ticks = parse(Float64, interval[1:(end - 5)])
        return (func) ->
            GenericCallbacks.EveryXSimulationTime(func, ticks, solver)
    elseif endswith(interval, "steps")
        steps = parse(Int, interval[1:(end - 5)])
        return (func) -> GenericCallbacks.EveryXSimulationSteps(func, steps)
    elseif endswith(interval, "hours")
        secs = 60 * 60 * parse(Int, interval[1:(end - 5)])
        return (func) ->
            GenericCallbacks.EveryXWallTimeSeconds(func, secs, mpicomm)
    elseif endswith(interval, "mins")
        secs = 60 * parse(Int, interval[1:(end - 4)])
        return (func) ->
            GenericCallbacks.EveryXWallTimeSeconds(func, secs, mpicomm)
    elseif endswith(interval, "secs")
        secs = parse(Int, interval[1:(end - 4)])
        return (func) ->
            GenericCallbacks.EveryXWallTimeSeconds(func, secs, mpicomm)
    elseif interval == "default"
        if default === nothing
            @warn "no default available; ignoring"
            return nothing
        elseif default === ""
            return nothing
        else
            return CB_constructor(default, solver_config, "")
        end
    elseif interval == "never"
        return nothing
    else
        @warn @sprintf(
            """
%s: unrecognized interval; ignoring""",
            interval,
        )
        return nothing
    end
end

function __init__()
    tictoc()
end

end # module
