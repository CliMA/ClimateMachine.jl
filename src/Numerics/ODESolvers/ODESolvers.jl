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


# Types
#  `DiffEqBase.AbstractODEProblem`: contains rhs!, Q0, tspan
#    - reuse `OrdinaryDiffEq.ODEProblem` for explicit
#    - reuse `OrdinaryDiffEq.ODEProblem` with `jac` arg for linear + full IMEX (https://docs.sciml.ai/latest/features/performance_overloads/#ode_explicit_jac-1)
#    - reuse `OrdinaryDiffEq.SplitODEProblem` for linear + remainder IMEX
#    - define new `IncrementODEProblem` for LSRK (https://github.com/SciML/DifferentialEquations.jl/issues/615)
#    - Multirate: new problem type, or nested problem type?
#
#  `DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm` for indicating which algorithm to use
#    - singleton type if no options, e.g. LSRK
#    - fields for options such as "linear solver" for IMEX
#
#  `DistributedODEIntegrator <: DiffEqBase.AbstractODEIntegrator{algType,true,uType,tType}` is a stripped down `ODEIntegrator` 
#  (https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123)
#    - capture full state (current time, etc)
# 
#  Do we want a `DiffEqBase.AbstractODESolution` object to capture the result?
# 
# Linear solvers:
#   https://docs.sciml.ai/latest/features/linear_nonlinear/#Linear-Solvers:-linsolve-Specification-1
#   https://docs.sciml.ai/latest/features/linear_nonlinear/#Implementing-Your-Own-LinSolve:-How-LinSolveFactorize-Was-Created-1
#
# plan 1: 
#  - similar to https://github.com/SciML/OrdinaryDiffEq.jl/blob/a9f9a0d07bf34ba567e2ca9dfc826d5c359d5a41/src/solve.jl#L1-L7?
function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
        alg::DistributedODEAlgorithm, args...;
        kwargs...)
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    solve!(integrator)
    return nothing
end



#  define `DiffEqBase.__init(prob::AbstractODEProblem, alg::DistributedODEAlgorithm, args...; kwargs...)` which shoudl return `::DistributedODEIntegrator`
#  define `DiffEqBase.solve!(::DistributedODEIntegrator)`
#  would we want a 




# usage:
#
#   prob = IncrementODEProblem(dg, Q0, (0.0,timeend))
#   solve(prob, LSRK54CarpenterKennedy())
#
#   prob = ODEProblem(dg, Q0, (0.0,timeend); jac=dg1dlinear)
#   solve(prob, ARK2GiraldoKellyConstantinescu(linsolve=BandedGMRES()))
#

# 

#  algo: just the tableau "DistributedODESolvers"
#  cache: intermediate storage

# 1. LSRK: define our own IncrementODEProblem 
# 2. IMEX: use either SplitODEProblem (for linear + remainder) or ODEProblem with `jac` argument 
# 3. Provide our own linear solvers via `linsolve` argument to `Solver`
# 4. Multirate?
#   - either a tuple or nested problems?


import DiffEqBase

abstract type DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm
end



abstract type AbstractCache end

struct IncrementODEProblem{uType,tType,P,F,K} <: DiffEqBase.AbstractODEProblem{uType,tType,true}
    """
    The ODE is `du/dt = f(u,p,t)`: this should define a method `f(y, u, p, t, α, β)` which computes
    ```
    y .= α .* f(u, p, t) .+ β .* y
    ```
    for scalar `α` and `β`.
    """
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
