using CLIMA.DGmethods:
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

export MultistateRungeKutta

LSRK2N = LowStorageRungeKutta2N

"""
    MultistateRungeKutta(slow_solver, fast_solver; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q_fast} = f_fast(Q_fast, Q_slow, t)
  \\dot{Q_slow} = f_slow(Q_slow, Q_fast, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multistate multirate Runge-Kutta scheme using two
different RK solvers and two different MPIStateArrays. This is based on

Currently only the low storage RK methods can be used as slow solvers

  - [`LowStorageRungeKuttaMethod`](@ref)

### References
"""
mutable struct MultistateRungeKutta{SS, SA, FS, RT} <: AbstractODESolver
    "slow solver"
    slow_solver::SS
    "sAlt solver"
    sAlt_solver::SA
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT

    function MultistateRungeKutta(
        slow_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        sAlt_solver = nothing,
        dt = getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        SA = typeof(sAlt_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))
        return new{SS, SA, FS, RT}(
            slow_solver,
            sAlt_solver,
            fast_solver,
            RT(dt),
            RT(t0),
        )
    end
end
MSRK = MultistateRungeKutta

function dostep!(Qvec, msmrrk::MSRK{SS}, param, time) where {SS <: LSRK2N}
    slow_dt = msmrrk.dt
    slow = msmrrk.slow_solver
    sAlt = msmrrk.sAlt_solver
    fast = msmrrk.fast_solver

    Qslow = Qvec.slow
    Qfast = Qvec.fast


    dQfast = fast.dQ
    if sAlt == nothing
        dQslow = slow.dQ
        dQ2fast = similar(dQslow)
    else
        dQslow = sAlt.dQ
        dQ2fast = slow.dQ
    end

    slow_bl = slow.rhs!.balancelaw
    fast_bl = fast.rhs!.balancelaw

    groupsize = 256

    for slow_s in 1:length(slow.RKA)
        # Currnent slow state time
        slow_stage_time = time + slow.RKC[slow_s] * slow_dt

        # set state to match slow model
        # zero out the cummulative arrays
        initialize_fast_state!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
        )

        # Evaluate the slow mode
        # TODO: replace slow.rhs! call with use of dQ2fast]
        if sAlt != nothing
            sAlt.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)
        end
        slow.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)

        # --> save tendency for the fast
        slow.rhs!(dQ2fast, Qslow, param, slow_stage_time, increment = false)

        event = Event(device(Qslow))
        event = update!(device(Qslow), groupsize)(
            realview(dQslow),
            realview(Qslow),
            slow.RKA[slow_s % length(slow.RKA) + 1],
            slow.RKB[slow_s],
            slow_dt,
            nothing,
            nothing,
            nothing;
            ndrange = length(realview(Qslow)),
            dependencies = (event,),
        )
        wait(device(Qslow), event)

        # get slow tendency contribution to advance fast equation
        #  ---> work with dQ2fast as input
        pass_tendency_from_slow_to_fast!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qfast,
            dQ2fast,
        )

        # --> calculate tendency for the fast
        fast.rhs!(dQfast, Qfast, param, slow_stage_time, increment = true)

        event = Event(device(Qfast))
        event = update!(device(Qfast), groupsize)(
            realview(dQfast),
            realview(Qfast),
            slow.RKA[slow_s % length(slow.RKA) + 1],
            slow.RKB[slow_s],
            slow_dt,
            nothing,
            nothing,
            nothing;
            ndrange = length(realview(Qfast)),
            dependencies = (event,),
        )
        wait(device(Qfast), event)

        #  ---> need to cumulate U at this time (and not at each RKB sub-time-step)
        cummulate_fast_solution!(
            fast_bl,
            fast.rhs!,
            Qfast,
            slow_stage_time,
            slow_dt,
            0,
        )

        ### later testing ignore this
        # reconcile slow equation using fast equation
        reconcile_from_fast_to_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            1,
        )
    end
    return nothing
end
