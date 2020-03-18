"""
    ODESolvers

Ordinary differential equation solvers
"""
module ODESolvers

using GPUifyLoops
using StaticArrays
using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

using ..SpaceMethods
using ..LinearSolvers
using ..MPIStateArrays: device, realview

export solve!, updatedt!, gettime

abstract type AbstractODESolver end
"""
    gettime(solver::AbstractODESolver)

Returns the current simulation time of the ODE solver `solver`
"""
gettime(solver::AbstractODESolver) = solver.t

"""
    getdt(solver::AbstractODESolver)

Returns the current simulation time step of the ODE solver `solver`
"""
getdt(solver::AbstractODESolver) = solver.dt

function dostep! end

"""
    updatedt!(solver::AbstractODESolver, dt)

Change the time step size to `dt` for the ODE solver `solver`.
"""
updatedt!(solver::AbstractODESolver, dt) =
  error("Variable time stepping not implemented for $(typeof(solver))")

"""
    updatetime!(solver::AbstractODESolver, time)

Change the current time to `time` for the ODE solver `solver`.
"""
updatetime!(solver::AbstractODESolver, time) =
  error("Variable time stepping not implemented for $(typeof(solver))")

isadjustable(solver::AbstractODESolver) = true

# {{{ run!
"""
    solve!(Q, solver::AbstractODESolver; timeend,
           stopaftertimeend=true, numberofsteps, callbacks)

Solves an ODE using the `solver` starting from a state `Q`. The state `Q` is
updated inplace. The final time `timeend` or `numberofsteps` must be specified.

A series of optional callback functions can be specified using the tuple
`callbacks`; see [`GenericCallbacks`](@ref).
"""
function solve!(Q, solver::AbstractODESolver, p=nothing; timeend::Real=Inf,
                adjustfinalstep=true, numberofsteps::Integer=0, callbacks=())

  @assert isfinite(timeend) || numberofsteps > 0
  if adjustfinalstep && !isadjustable(solver)
    error("$solver does not support time step adjustments. Can only be used with `adjustfinalstep=false`.")
  end
  t0 = gettime(solver)

  # Loop through an initialize callbacks (if they need it)
  foreach(callbacks) do cb
    try
      cb(true)
    catch
    end
  end

  step = 0
  time = t0
  while time < timeend
    step += 1

    time = dostep!(Q, solver, p, timeend, adjustfinalstep)

    # FIXME: Determine better way to handle postcallback behavior
    # Current behavior:
    #   retval = 1 exit after all callbacks
    #   retval = 2 exit immediately
    retval = 0
    for (i, cb) in enumerate(callbacks)
      # FIXME: Consider whether callbacks need anything, or if function closure
      #        can be used for everything
      thisretval = cb()
      thisretval = (thisretval == nothing) ? 0 : thisretval
      !(thisretval in (0, 1, 2)) &&
      error("callback #$(i) returned invalid value. It should return either:
            `nothing` (continue time stepping)
            `0`       (continue time stepping)
            `1`       (stop time stepping after all callbacks)
            `2`       (stop time stepping immediately)")
      retval = max(thisretval, retval)
      retval == 2 && return gettime(solver)
    end
    retval == 1 && return gettime(solver)

    # Figure out if we should stop
    if numberofsteps == step
      return gettime(solver)
    end
  end
  gettime(solver)
end
# }}}

include("LowStorageRungeKuttaMethod.jl")
include("StrongStabilityPreservingRungeKuttaMethod.jl")
include("AdditiveRungeKuttaMethod.jl")
include("MultirateInfinitesimalStepMethod.jl")
include("MultirateRungeKuttaMethod.jl")

end # module
