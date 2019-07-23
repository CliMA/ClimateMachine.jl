module ODESolvers

using ..MPIStateArrays
using GPUifyLoops

export solve!, updatedt!

abstract type AbstractODESolver end
"""
    gettime(solver::AbstractODESolver)

Returns the current simulation time of the ODE solver `solver`
"""
gettime(solver::AbstractODESolver) = solver.t[1]
function dostep! end

"""
    updatedt!(solver::AbstractODESolver, dt)

Change the time step size to `dt` for the ODE solver `solver`.
"""
updatedt!(solver::AbstractODESolver, dt) =
  error("Variable time stepping not implemented for $(typeof(solver))")

# `realview` and `device` are used for testing ODE solvers independently of spatial discretisations,
# i.e. using plain arrays as state vectors
realview(Q::MPIStateArray) = view(Q.Q, axes(Q.Q)[1:end-1]..., Q.realelems)
realview(Q::Array) = Q
device(::Array) = CPU()
device(Q::MPIStateArray) = device(Q.Q)

using Requires
@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  realview(Q::CuArray) = Q
  device(::CuArray) = CUDA()
end

# {{{ run!
"""
    solve!(Q, solver::AbstractODESolver; timeend,
           stopaftertimeend=true, numberofsteps, callbacks)

Solves an ODE using the `solver` starting from a state `Q`. The state `Q` is
updated inplace. The final time `timeend` or `numberofsteps` must be specified.

A series of optional callback functions can be specified using the tuple
`callbacks`; see [`GenericCallbacks`](@ref).
"""
function solve!(Q, solver::AbstractODESolver, param=nothing; timeend::Real=Inf,
                adjustfinalstep=true, numberofsteps::Integer=0, callbacks=())

  @assert isfinite(timeend) || numberofsteps > 0

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

    time = dostep!(Q, solver, param, timeend, adjustfinalstep)

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

end # module

