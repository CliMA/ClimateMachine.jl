import DiffEqBase
export DiffEqJLSolver, DiffEqJLIMEXSolver

abstract type AbstractDiffEqJLSolver <: AbstractODESolver end

"""
    DiffEqJLSolver(f, RKA, RKB, RKC, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

via a DifferentialEquations.jl DEAlgorithm, which includes support
for OrdinaryDiffEq.jl, Sundials.jl, and more.
"""
mutable struct DiffEqJLSolver{I} <: AbstractDiffEqJLSolver
    integ::I

    function DiffEqJLSolver(
        rhs!,
        alg,
        Q,
        args...;
        t0 = 0,
        p = nothing,
        kwargs...,
    )
        prob = DiffEqBase.ODEProblem(
            (du, u, p, t) -> rhs!(du, u, p, t; increment = false),
            Q,
            (float(t0), typemax(typeof(float(t0)))),
            p,
        )
        integ = DiffEqBase.init(
            prob,
            alg,
            args...;
            adaptive = false,
            save_everystep = false,
            save_start = false,
            save_end = false,
            kwargs...,
        )
        new{typeof(integ)}(integ)
    end
end

"""
    DiffEqJLSolver(f, RKA, RKB, RKC, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f_I(Q, t) + f_E(Q, t)
```

via a DifferentialEquations.jl DEAlgorithm, which includes support
for OrdinaryDiffEq.jl, Sundials.jl, and more.
"""
mutable struct DiffEqJLIMEXSolver{I} <: AbstractDiffEqJLSolver
    integ::I

    function DiffEqJLIMEXSolver(
        rhs!,
        rhs_implicit!,
        alg,
        Q,
        args...;
        t0 = 0,
        p = nothing,
        kwargs...,
    )
        prob = DiffEqBase.SplitODEProblem(
            (du, u, p, t) -> rhs_implicit!(du, u, p, t; increment = false),
            (du, u, p, t) -> rhs!(du, u, p, t; increment = false),
            Q,
            (float(t0), typemax(typeof(float(t0)))),
            p,
        )
        integ = DiffEqBase.init(
            prob,
            alg,
            args...;
            adaptive = false,
            save_everystep = false,
            save_start = false,
            save_end = false,
            kwargs...,
        )

        new{typeof(integ)}(integ)
    end
end

gettime(solver::AbstractDiffEqJLSolver) = solver.integ.t
getdt(solver::AbstractDiffEqJLSolver) = solver.integ.dt
updatedt!(solver::AbstractDiffEqJLSolver, dt) =
    DiffEqBase.set_proposed_dt!(solver.integ, dt)
updatetime!(solver::AbstractDiffEqJLSolver, t) =
    DiffEqBase.set_t!(solver.integ, t)
isadjustable(solver::AbstractDiffEqJLSolver) = true # Is this isadaptive? Or something different?

"""
    ODESolvers.general_dostep!(Q, solver::AbstractODESolver, p,
                               timeend::Real, adjustfinalstep::Bool)

Use the solver to step `Q` forward in time from the current time, to the time
`timeend`. If `adjustfinalstep == true` then `dt` is adjusted so that the step
does not take the solution beyond the `timeend`.
"""
function general_dostep!(
    Q,
    solver::AbstractDiffEqJLSolver,
    p,
    timeend::Real;
    adjustfinalstep::Bool,
)
    integ = solver.integ

    if DiffEqBase.DataStructures.top(integ.opts.tstops) !== timeend
        DiffEqBase.add_tstop!(integ, timeend)
    end
    dostep!(Q, solver, p, time)
    solver.integ.t
end

function dostep!(
    Q,
    solver::AbstractDiffEqJLSolver,
    p,
    time,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
)

    integ = solver.integ
    integ.p = p # Can this change?

    rv_Q = realview(Q)
    if integ.u != Q
        integ.u .= Q
        DiffEqBase.u_modified!(integ, true)
        # Will time always be correct?
    end

    DiffEqBase.step!(integ)
    rv_Q .= solver.integ.u
end

function DiffEqJLConstructor(alg)
    constructor =
        (F, Q; dt = 0, t0 = 0) -> DiffEqJLSolver(F, alg, Q; t0 = t0, dt = dt)
    return constructor
end
