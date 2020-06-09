"""
    ODESolvers

Ordinary differential equation solvers
"""
module ODESolvers

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays
using ..SystemSolvers
using ..MPIStateArrays: array_device, realview

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

"""
    ODESolvers.general_dostep!(Q, solver::AbstractODESolver, p,
                               timeend::Real, adjustfinalstep::Bool)

Use the solver to step `Q` forward in time from the current time, to the time
`timeend`. If `adjustfinalstep == true` then `dt` is adjusted so that the step
does not take the solution beyond the `timeend`.
"""
function general_dostep!(
    Q,
    solver::AbstractODESolver,
    p,
    timeend::Real;
    adjustfinalstep::Bool,
)
    time, dt = solver.t, solver.dt
    final_step = false
    if adjustfinalstep && time + dt > timeend
        orig_dt = dt
        dt = timeend - time
        updatedt!(solver, dt)
        final_step = true
    end
    @assert dt > 0

    dostep!(Q, solver, p, time)

    if !final_step
        solver.t += dt
    else
        updatedt!(solver, orig_dt)
        solver.t = timeend
    end
end

"""
    updatedt!(solver::AbstractODESolver, dt)

Change the time step size to `dt` for the ODE solver `solver`.
"""
updatedt!(solver::AbstractODESolver, dt) = (solver.dt = dt)

"""
    updatetime!(solver::AbstractODESolver, time)

Change the current time to `time` for the ODE solver `solver`.
"""
updatetime!(solver::AbstractODESolver, time) = (solver.t = time)

isadjustable(solver::AbstractODESolver) = true

# {{{ run!
"""
    solve!(Q, solver::AbstractODESolver; timeend,
           stopaftertimeend=true, numberofsteps, callbacks)

Solves an ODE using the `solver` starting from a state `Q`. The state `Q` is
updated inplace. The final time `timeend` or `numberofsteps` must be specified.

A series of optional callback functions can be specified using the tuple
`callbacks`; see the `GenericCallbacks` module.
"""
function solve!(
    Q,
    solver::AbstractODESolver,
    p = nothing;
    timeend::Real = Inf,
    adjustfinalstep = true,
    numberofsteps::Integer = 0,
    callbacks = (),
)

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

        time = general_dostep!(
            Q,
            solver,
            p,
            timeend;
            adjustfinalstep = adjustfinalstep,
        )

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

include("BackwardEulerSolvers.jl")
include("MultirateInfinitesimalGARKExplicit.jl")
include("MultirateInfinitesimalGARKDecoupledImplicit.jl")
include("LowStorageRungeKuttaMethod.jl")
include("StrongStabilityPreservingRungeKuttaMethod.jl")
include("AdditiveRungeKuttaMethod.jl")
include("MultirateInfinitesimalStepMethod.jl")
include("MultirateRungeKuttaMethod.jl")

end # module
