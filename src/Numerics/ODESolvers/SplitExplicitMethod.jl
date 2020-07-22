export SplitExplicitSolver

using ..BalanceLaws:
    initialize_states!,
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

LSRK2N = LowStorageRungeKutta2N

@doc """
    SplitExplicitSolver(slow_solver, fast_solver; dt, t0 = 0, coupled = true)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q_fast} = f_fast(Q_fast, Q_slow, t)
  \\dot{Q_slow} = f_slow(Q_slow, Q_fast, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This method performs an operator splitting to timestep the vertical average
of the model at a faster rate than the full model. This results in a first-
order time stepper.

""" SplitExplicitSolver
mutable struct SplitExplicitSolver{SS, FS, RT, MSA} <: AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT
    "storage for transfer tendency"
    dQ2fast::MSA

    function SplitExplicitSolver(
        slow_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        dt = getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))

        dQ2fast = similar(slow_solver.dQ)
        dQ2fast .= -0
        MSA = typeof(dQ2fast)

        return new{SS, FS, RT, MSA}(
            slow_solver,
            fast_solver,
            RT(dt),
            RT(t0),
            dQ2fast,
        )
    end
end

function dostep!(
    Qvec,
    split::SplitExplicitSolver{SS},
    param,
    time::Real,
) where {SS <: LSRK2N}
    slow = split.slow_solver
    fast = split.fast_solver

    Qslow = Qvec.slow
    Qfast = Qvec.fast

    dQslow = slow.dQ
    dQ2fast = split.dQ2fast

    slow_bl = slow.rhs!.balance_law
    fast_bl = fast.rhs!.balance_law

    groupsize = 256

    slow_dt = getdt(slow)
    fast_dt_in = getdt(fast)

    for slow_s in 1:length(slow.RKA)
        # Current slow state time
        slow_stage_time = time + slow.RKC[slow_s] * slow_dt

        # Initialize fast model and tendency adjustment
        # before evalution of slow mode
        initialize_states!(slow_bl, fast_bl, slow.rhs!, fast.rhs!, Qslow, Qfast)

        # Evaluate the slow mode
        # --> save tendency for the fast
        slow.rhs!(dQ2fast, Qslow, param, slow_stage_time, increment = false)

        # vertically integrate slow tendency to advance fast equation
        # and use vertical mean for slow model (negative source)
        # ---> work with dQ2fast as input
        tendency_from_slow_to_fast!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            dQ2fast,
        )

        # Compute (and RK update) slow tendency
        slow.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)

        # Fractional time for slow stage
        if slow_s == length(slow.RKA)
            γ = 1 - slow.RKC[slow_s]
        else
            γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
        end

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        nsubsteps = fast_dt_in > 0 ? ceil(Int, γ * slow_dt / fast_dt_in) : 1
        fast_dt = γ * slow_dt / nsubsteps

        updatedt!(fast, fast_dt)

        for substep in 1:nsubsteps
            fast_time = slow_stage_time + (substep - 1) * fast_dt
            dostep!(Qfast, fast, param, fast_time)
            cummulate_fast_solution!(
                slow_bl,
                fast_bl,
                fast.rhs!,
                Qfast,
                fast_time,
                fast_dt,
                substep,
            )
        end

        # Update (RK-stage) slow state
        event = Event(array_device(Qslow))
        event = update!(array_device(Qslow), groupsize)(
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
        wait(array_device(Qslow), event)

        # reconcile slow equation using fast equation
        reconcile_from_fast_to_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
        )
    end
    updatedt!(fast, fast_dt_in)

    return nothing
end
