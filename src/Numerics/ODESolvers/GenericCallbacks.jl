"""
    GenericCallbacks

This module defines interfaces and wrappers for callbacks to be used with an
`AbstractODESolver`.

A callback `cb` defines two methods:

- `GenericCallbacks.init!(cb, solver, Q, param, t0)`, to be called at solver
  initialization.

- `GenericCallbacks.call!(cb, solver, Q, param, t)`, to be called after each time step:
  the return value dictates what action should be taken:

   * `0` or `nothing`: continue time stepping as usual
   * `1`: stop time stepping after all callbacks have been executed
   * `2`: stop time stepping immediately

Additionally, "wrapper" callbacks can be used to execute the callbacks under certain
conditions:

 - [`AtStart`](@ref)
 - [`EveryXWallTimeSeconds`](@ref)
 - [`EveryXSimulationTime`](@ref)
 - [`EveryXSimulationSteps`](@ref)

For convenience, the following objects can also be used as callbacks:

 - a `Function` object `f`, `init!` is a no-op, and `call!` will call `f()`, and ignore the return value.
 - a `Tuple` object will call `init!` and `call!` on each element of the tuple.

"""
module GenericCallbacks

export AtStart,
    EveryXWallTimeSeconds, EveryXSimulationTime, EveryXSimulationSteps

using MPI

init!(f::Function, solver, Q, param, t0) = nothing
function call!(f::Function, solver, Q, param, t0)
    f()
    return nothing
end

function init!(callbacks::Tuple, solver, Q, param, t0)
    for cb in callbacks
        GenericCallbacks.init!(cb, solver, Q, param, t0)
    end
end
function call!(callbacks::Tuple, solver, Q, param, t)
    val = 0
    for cb in callbacks
        val_i = GenericCallbacks.call!(cb, solver, Q, param, t)
        val_i = (val_i === nothing) ? 0 : val_i
        val = max(val, val_i)
        if val == 2
            return val
        end
    end
    return val
end

abstract type AbstractCallback end

"""
    AtStart(callback) <: AbstractCallback

A wrapper callback to execute `callback` at initialization, in addition to after each
timestep.
"""
struct AtStart <: AbstractCallback
    callback
end
function init!(cb::AtStart, solver, Q, param, t0)
    init!(cb.callback, solver, Q, param, t0)
    call!(cb.callback, solver, Q, param, t0)
end
function call!(cb::AtStart, solver, Q, param, t)
    call!(cb.callback, solver, Q, param, t)
end

"""
    EveryXWallTimeSeconds(callback, Δtime, mpicomm)

A wrapper callback to execute `callback` every `Δtime` wallclock time seconds.
`mpicomm` is used to syncronize runtime across MPI ranks.
"""
mutable struct EveryXWallTimeSeconds <: AbstractCallback
    "callback to wrap"
    callback
    "wall time seconds between callbacks"
    Δtime::Real
    "MPI communicator"
    mpicomm::MPI.Comm
    "time of the last callback"
    lastcbtime_ns::UInt64
    function EveryXWallTimeSeconds(callback, Δtime, mpicomm)
        lastcbtime_ns = zero(UInt64)
        new(callback, Δtime, mpicomm, lastcbtime_ns)
    end
end

function init!(cb::EveryXWallTimeSeconds, solver, Q, param, t0)
    cb.lastcbtime_ns = time_ns()
    init!(cb.callback, solver, Q, param, t0)
end
function call!(cb::EveryXWallTimeSeconds, solver, Q, param, t)
    # Check whether we should do a callback
    currtime_ns = time_ns()
    runtime = (currtime_ns - cb.lastcbtime_ns) * 1e-9
    runtime = MPI.Allreduce(runtime, max, cb.mpicomm)
    if runtime < cb.Δtime
        return 0
    else
        # Compute the next time to do a callback
        cb.lastcbtime_ns = currtime_ns
        return call!(cb.callback, solver, Q, param, t)
    end
end


"""
    EveryXSimulationTime(f, Δtime)

A wrapper callback to execute `callback` every `time` simulation time seconds.
"""
mutable struct EveryXSimulationTime <: AbstractCallback
    "callback to wrap"
    callback
    "simulation time seconds between callbacks"
    Δtime::Real
    "time of the last callback"
    lastcbtime::Real
    function EveryXSimulationTime(callback, Δtime)
        new(callback, Δtime, 0)
    end
end

function init!(cb::EveryXSimulationTime, solver, Q, param, t0)
    cb.lastcbtime = t0
    init!(cb.callback, solver, Q, param, t0)
end
function call!(cb::EveryXSimulationTime, solver, Q, param, t)
    # Check whether we should do a callback
    if (t - cb.lastcbtime) < cb.Δtime
        return 0
    else
        # Compute the next time to do a callback
        cb.lastcbtime = t
        return call!(cb.callback, solver, Q, param, t)
    end
end


"""
    EveryXSimulationSteps(callback, Δsteps)

A wrapper callback to execute `callback` every `nsteps` of the time stepper.
"""
mutable struct EveryXSimulationSteps <: AbstractCallback
    "callback to wrap"
    callback
    "number of steps between callbacks"
    Δsteps::Int
    "number of steps since last callback"
    steps::Int
    function EveryXSimulationSteps(callback, Δsteps)
        new(callback, Δsteps, 0)
    end
end

function init!(cb::EveryXSimulationSteps, solver, Q, param, t0)
    cb.steps = 0
    init!(cb.callback, solver, Q, param, t0)
end
function call!(cb::EveryXSimulationSteps, solver, Q, param, t)
    cb.steps += 1
    if cb.steps < cb.Δsteps
        return 0
    else
        cb.steps = 0
        return call!(cb.callback, solver, Q, param, t)
    end
end

end
