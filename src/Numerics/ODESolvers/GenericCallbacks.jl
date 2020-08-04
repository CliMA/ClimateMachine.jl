"""
    GenericCallbacks

This module defines interfaces and wrappers for callbacks to be used with an
`AbstractODESolver`.

A callback `cb` defines three methods:

- `GenericCallbacks.init!(cb, solver, Q, param, t)`, to be called at solver
  initialization.

- `GenericCallbacks.call!(cb, solver, Q, param, t)`, to be called after each
  time step: the return value dictates what action should be taken:

   * `0` or `nothing`: continue time stepping as usual
   * `1`: stop time stepping after all callbacks have been executed
   * `2`: stop time stepping immediately

- `GenericCallbacks.fini!(cb, solver, Q, param, t)`, to be called at solver
  finish.

Additionally, _wrapper_ callbacks can be used to execute the callbacks under
certain conditions:

 - [`AtInit`](@ref)
 - [`AtInitAndFini`](@ref)
 - [`EveryXWallTimeSeconds`](@ref)
 - [`EveryXSimulationTime`](@ref)
 - [`EveryXSimulationSteps`](@ref)

For convenience, the following objects can also be used as callbacks:

- A `Function` object `f`, `init!` and `fini!` are no-ops, and `call!` will
  call `f()`, and ignore the return value.
- A `Tuple` object will call `init!`, `call!` and `fini!` on each element
  of the tuple.
"""
module GenericCallbacks

export AtInit,
    AtInitAndFini,
    EveryXWallTimeSeconds,
    EveryXSimulationTime,
    EveryXSimulationSteps

using MPI

init!(f::Function, solver, Q, param, t) = nothing
function call!(f::Function, solver, Q, param, t)
    f()
    return nothing
end
fini!(f::Function, solver, Q, param, t) = nothing

function init!(callbacks::Tuple, solver, Q, param, t)
    for cb in callbacks
        GenericCallbacks.init!(cb, solver, Q, param, t)
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
function fini!(callbacks::Tuple, solver, Q, param, t)
    for cb in callbacks
        GenericCallbacks.fini!(cb, solver, Q, param, t)
    end
end

abstract type AbstractCallback end

"""
    AtInit(callback) <: AbstractCallback

A wrapper callback to execute `callback` at initialization as well as
after each interval.
"""
struct AtInit <: AbstractCallback
    callback
end
function init!(cb::AtInit, solver, Q, param, t)
    init!(cb.callback, solver, Q, param, t)
    call!(cb.callback, solver, Q, param, t)
end
function call!(cb::AtInit, solver, Q, param, t)
    call!(cb.callback, solver, Q, param, t)
end
function fini!(cb::AtInit, solver, Q, param, t)
    fini!(cb.callback, solver, Q, param, t)
end

"""
    AtInitAndFini(callback) <: AbstractCallback

A wrapper callback to execute `callback` at initialization and at
finish as well as after each interval.
"""
struct AtInitAndFini <: AbstractCallback
    callback
end
function init!(cb::AtInitAndFini, solver, Q, param, t)
    init!(cb.callback, solver, Q, param, t)
    call!(cb.callback, solver, Q, param, t)
end
function call!(cb::AtInitAndFini, solver, Q, param, t)
    call!(cb.callback, solver, Q, param, t)
end
function fini!(cb::AtInitAndFini, solver, Q, param, t)
    call!(cb.callback, solver, Q, param, t)
    fini!(cb.callback, solver, Q, param, t)
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

function init!(cb::EveryXWallTimeSeconds, solver, Q, param, t)
    cb.lastcbtime_ns = time_ns()
    init!(cb.callback, solver, Q, param, t)
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
function fini!(cb::EveryXWallTimeSeconds, solver, Q, param, t)
    fini!(cb.callback, solver, Q, param, t)
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

function init!(cb::EveryXSimulationTime, solver, Q, param, t)
    cb.lastcbtime = t
    init!(cb.callback, solver, Q, param, t)
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
function fini!(cb::EveryXSimulationTime, solver, Q, param, t)
    fini!(cb.callback, solver, Q, param, t)
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

function init!(cb::EveryXSimulationSteps, solver, Q, param, t)
    cb.steps = 0
    init!(cb.callback, solver, Q, param, t)
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
function fini!(cb::EveryXSimulationSteps, solver, Q, param, t)
    fini!(cb.callback, solver, Q, param, t)
end

end
