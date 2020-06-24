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
using ..GenericCallbacks


# 3 types:
#  problem: contains rhs!, Q, tspan
#  algo: just the tableau "DistributedODESolvers"
#  cache: intermediate storage

# 1. LSRK: define our own IncrementODEProblem (https://github.com/SciML/DifferentialEquations.jl/issues/615)
# 2. IMEX: use either SplitODEProblem (for linear + remainder) or ODEProblem with `jac` argument (https://docs.sciml.ai/latest/features/performance_overloads/#ode_explicit_jac-1)
# 3. Provide our own linear solvers via `linsolve` argument to `Solver`
# 4. Multirate?
#   - either a tuple or nested problems?


import DiffEqBase

abstract type DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm
end

abstract type AbstractCache end

struct IncrementODEProblem{uType,tType,P,F,K} <: DiffEqBase.AbstractODEProblem{uType,tType,true}
    """The ODE is `du/dt = f(u,p,t)`: this must have a method `ODESolvers.afpby` """
    f::F
    """The initial condition is `u(tspan[1]) = u0`."""
    u0::uType
    """The solution `u(t)` will be computed for `tspan[1] ≤ t ≤ tspan[2]`."""
    tspan::tType
    """Constant parameters to be supplied as the second argument of `f`."""
    p::P
    """A callback to be applied to every solver which uses the problem."""
    kwargs::K
end

"""
    ODESolvers.afpby(f, y, u, param, t, α, β)

A function stub that can be extended for computing

    y .= α .* f(u, param, t) .+ β .* y

This must be defined on `f` any `IncrementODEProblem(f, ...)`.
"""
function afpby end




# we would define
function DiffEqBase.solve(prob::IncrementODEProblem, solver::DistributedODEAlgorithm, cache=cache(prob, solver))
end
function DiffEqBase.solve(prob::ODEProblem, solver::DistributedODEAlgorithm, cache=cache(prob, solver))
end


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

abstract type AbstractODEProblem end


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
    param = nothing;
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
    GenericCallbacks.init!(callbacks, solver, Q, param, t0)

    step = 0
    time = t0
    while time < timeend
        step += 1

        time = general_dostep!(
            Q,
            solver,
            param,
            timeend;
            adjustfinalstep = adjustfinalstep,
        )

        val = GenericCallbacks.call!(callbacks, solver, Q, param, time)
        if val !== nothing && val > 0
            return gettime(solver)
        end

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
include("SplitExplicitMethod.jl")

end # module
