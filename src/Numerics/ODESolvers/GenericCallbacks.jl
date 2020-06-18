"""
    GenericCallbacks

A set of callback functions to be used with an `AbstractODESolver`
"""
module GenericCallbacks
using MPI

using ..ODESolvers

"""
    EveryXWallTimeSeconds(f, time, mpicomm)

This callback will run the function 'f()' every `time` wallclock time seconds.
The `mpicomm` is used to syncronize runtime across MPI ranks.
"""
struct EveryXWallTimeSeconds
    "time of the last callback"
    lastcbtime_ns::Array{UInt64}
    "time between callbacks"
    Δtime::Real
    "MPI communicator"
    mpicomm
    "function to execute for callback"
    func::Function
    function EveryXWallTimeSeconds(func, Δtime, mpicomm)
        lastcbtime_ns = [time_ns()]
        new(lastcbtime_ns, Δtime, mpicomm, func)
    end
end
function (cb::EveryXWallTimeSeconds)(initialize::Bool = false)
    # Is this an initialization call? If so, start the timers
    if initialize
        cb.lastcbtime_ns[1] = time_ns()
        # If this is initialization init the callback too
        try
            cb.func(true)
        catch
        end
        return 0
    end

    # Check whether we should do a callback
    currtime_ns = time_ns()
    runtime = (currtime_ns - cb.lastcbtime_ns[1]) * 1e-9
    runtime = MPI.Allreduce(runtime, MPI.MAX, cb.mpicomm)
    if runtime < cb.Δtime
        return 0
    else
        # Compute the next time to do a callback
        cb.lastcbtime_ns[1] = currtime_ns
        return cb.func()
    end
end

"""
   EveryXSimulationTime(f, time, state)

This callback will run the function 'f()' every `time` wallclock time seconds.
The `state` is used to query for the simulation time.
"""
struct EveryXSimulationTime
    "time of the last callback"
    lastcbtime::Array{Real}
    "time between callbacks"
    Δtime::Real
    "function to execute for callback"
    func::Function
    "timestepper, used to query for time"
    solver
    function EveryXSimulationTime(func, Δtime, solver)
        lastcbtime = [ODESolvers.gettime(solver)]
        new(lastcbtime, Δtime, func, solver)
    end
end
function (cb::EveryXSimulationTime)(initialize::Bool = false)
    # Is this an initialization call? If so, start the timers
    if initialize
        cb.lastcbtime[1] = ODESolvers.gettime(cb.solver)
        # If this is initialization init the callback too
        try
            cb.func(true)
        catch
        end
        return 0
    end

    # Check whether we should do a callback
    currtime = ODESolvers.gettime(cb.solver)
    if (currtime - cb.lastcbtime[1]) < cb.Δtime
        return 0
    else
        # Compute the next time to do a callback
        cb.lastcbtime[1] = currtime
        return cb.func()
    end
end

"""
   EveryXSimulationSteps(f, steps)

This callback will run the function 'f()' every `steps` of the time stepper
"""
struct EveryXSimulationSteps
    "number of steps since last callback"
    steps::Array{Int}
    "number of steps between callbacks"
    Δsteps::Integer
    "function to execute for callback"
    func::Function
    function EveryXSimulationSteps(func, Δsteps)
        steps = [Int(0)]
        new(steps, Δsteps, func)
    end
end
function (cb::EveryXSimulationSteps)(initialize::Bool = false)
    # Is this an initialization call? If so, start the timers
    if initialize
        cb.steps[1] = 0
        # If this is initialization init the callback too
        try
            cb.func(true)
        catch
        end
        return 0
    end

    # Check whether we should do a callback
    cb.steps[1] += 1
    if cb.steps[1] < cb.Δsteps
        return 0
    else
        cb.steps[1] = 0
        return cb.func()
    end
end

end
