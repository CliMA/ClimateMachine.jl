module CLIMAAtmosDycore

export solve!, getrhsfunction

using Requires

abstract type AbstractAtmosDiscretization end

"""
    getrhsfunction(disc::AbstractAtmosDiscretization)

The spatial discretizations are of the form ``Q̇ = f(Q)``, and this function
returns the handle to right-hand side function ``f`` of the `disc`
"""
getrhsfunction(disc::AbstractAtmosDiscretization) =
throw(MethodError(getrhsfunction, typeof(disc)))

abstract type AbstractAtmosODESolver end
gettime(solver::AbstractAtmosODESolver) = solver.t[1]
dostep!(Q, solver::AbstractAtmosODESolver) = error()
# {{{ run!
"""
    solve!(Q, solver::AbstractAtmosODESolver; timeend,
           stopaftertimeend=true, numberofsteps, callbacks)

Solves an ODE using the `solver` starting from a state `Q`. The state `Q` is
updated inplace. The final time `timeend` or `numberofsteps` must be specified.

A series of optional callback functions can be specified using the tuple
`callbacks`; see [`GenericCallbacks`](@ref).

!!! todo

    Currently `stopaftertimeend` is not used. The idea behind it was that a user
    might want to stop either one step before or after the final time `timeend`.
    This should either be removed or used.
"""
function solve!(Q, solver::AbstractAtmosODESolver; timeend::Real=Inf,
                stopaftertimeend=true, numberofsteps::Integer=0, callbacks=())

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

    time = dostep!(Q, solver)

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
      retval == 2 && return
    end
    retval == 1 && return

    # Figure out if we should stop
    if numberofsteps == step
      return
    end
    if !stopaftertimeend && (t0 + (step+1) * dt) > timeend
      return
    end
  end
end
# }}}

include("AtmosStateArrays.jl")
include("VanillaAtmosDiscretizations.jl")
include("GenericCallbacks.jl")

end # module
