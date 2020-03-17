using CLIMA.DGmethods:
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

include("MultirateRungeKuttaMethod_kernels.jl")

export MultistateMultirateRungeKutta

ODEs = ODESolvers
LSRK2N = LowStorageRungeKutta2N

"""
    MultistateMultirateRungeKutta(slow_solver, fast_solver; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q_fast} = f_fast(Q_fast, Q_slow, t)
  \\dot{Q_slow} = f_slow(Q_slow, Q_fast, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multistate multirate Runge-Kutta scheme using two different RK
solvers and two different MPIStateArrays. This is based on

Currently only the low storage RK methods can be used as slow solvers

  - [`LowStorageRungeKuttaMethod`](@ref)

### References
"""
mutable struct MultistateMultirateRungeKutta{SS, SA, FS, RT} <:
               ODEs.AbstractODESolver
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

    function MultistateMultirateRungeKutta(
        slow_solver::LSRK2N,
        sAlt_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        dt = ODEs.getdt(slow_solver),
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
MSMRRK = MultistateMultirateRungeKutta

function ODEs.dostep!(
    Qvec,
    msmrrk::MSMRRK,
    param,
    timeend::Real,
    adjustfinalstep::Bool,
)
    time, dt = msmrrk.t, msmrrk.dt
    @assert dt > 0
    if adjustfinalstep && time + dt > timeend
        dt = timeend - time
        @assert dt > 0
    end

    ODEs.dostep!(Qvec, msmrrk, param, time, dt)

    if dt == mrrk.dt
        msmrrk.t += dt
    else
        msmrrk.t = timeend
    end
    return msmrrk.t
end

function ODEs.dostep!(
    Qvec,
    msmrrk::MSMRRK{SS},
    param,
    time::Real,
    slow_dt::AbstractFloat,
) where {SS <: LSRK2N}
    slow = msmrrk.slow_solver
    sAlt = msmrrk.sAlt_solver
    fast = msmrrk.fast_solver

    Qslow = Qvec.slow
    Qfast = Qvec.fast

    dQ2fast = slow.dQ
    dQslow = sAlt.dQ

    slow_bl = slow.rhs!.balancelaw
    fast_bl = fast.rhs!.balancelaw

    # set state to match slow model
    # zero out the cummulative arrays
    initialize_fast_state!(slow_bl, fast_bl, Qslow, Qfast, slow.rhs!, fast.rhs!)
    total_fast_step = 0

    threads = 256
    blocks = div(length(realview(Qslow)) + threads - 1, threads)

    for slow_s in 1:length(slow.RKA)
        # Currnent slow state time
        slow_stage_time = time + slow.RKC[slow_s] * slow_dt

        # Evaluate the slow mode
        # --> save tendency for the fast
        slow.rhs!(dQ2fast, Qslow, param, slow_stage_time, increment = false)

        # TODO: replace slow.rhs! call with use of dQ2fast
        slow.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)
        sAlt.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)

        event = Event(device(Q))
        event = update!(device(Q), groupsize)(
            realview(dQslow),
            realview(Qslow),
            RKB[s],
            RKA[s % length(RKA) + 1];
            ndrange = length(realview(Q)),
            dependencies = (event,),
        )
        wait(device(Q), event)

        ### for testing comment out everything below this

        # Fractional time for slow stage
        if slow_s == length(slow.RKA)
            γ = 1 - slow.RKC[slow_s]
        else
            γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
        end

        # Determine number of substeps we need
        fast_dt = ODEs.getdt(fast)
        nsubsteps = fast_dt > 0 ? ceil(Int, γ * slow_dt / ODEs.getdt(fast)) : 1
        fast_dt = γ * slow_dt / nsubsteps

        # get slow tendency contribution to advance fast equation
        #  ---> work with dQ2fast as input
        pass_tendency_from_slow_to_fast!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qfast,
            realview(dQ2fast),
        )

        for substep in 1:nsubsteps
            fast_time = slow_stage_time + (substep - 1) * fast_dt
            ODEs.dostep!(Qfast, fast, param, fast_time, fast_dt)
            #  ---> need to cumulate U at this time (and not at each RKB sub-time-step)
            cummulate_fast_solution!(
                fast_bl,
                Qfast,
                fast_time,
                fast_dt,
                total_fast_step,
            )
        end

        ### later testing ignore this
        # reconcile slow equation using fast equation
        reconcile_from_fast_to_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            slow.fast!,
            realview(dQslow),
            realview(Qslow),
            realview(Qfast),
            total_fast_step,
        )
    end
    return nothing
end
