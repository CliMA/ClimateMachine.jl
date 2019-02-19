module GenericCallbacks
using MPI

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
function (cb::EveryXWallTimeSeconds)(initialize::Bool=false)
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
    retval = cb.func()
  end
end

#=
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
  "pointer to the space state to query for time"
  runner::AD.Runner
  function EveryXSimulationTime(func, Δtime, runner)
    lastcbtime = [AD.gettime(runner)]
    new(lastcbtime, Δtime, func, runner)
  end
end
function (cb::EveryXSimulationTime)(initialize::Bool=false)
  # Is this an initialization call? If so, start the timers
  if initialize
    cb.lastcbtime[1] = AD.gettime(cb.runner)
    # If this is initialization init the callback too
    try
      cb.func(true)
    catch
    end
    return 0
  end

  # Check whether we should do a callback
  currtime = AD.gettime(cb.runner)
  if (currtime - cb.lastcbtime[1]) < cb.Δtime
    return 0
  else
    # Compute the next time to do a callback
    cb.lastcbtime[1] = currtime
    retval = cb.func()
  end
end
=#

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
function (cb::EveryXSimulationSteps)(initialize::Bool=false)
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
    retval = cb.func()
  end
end

end
