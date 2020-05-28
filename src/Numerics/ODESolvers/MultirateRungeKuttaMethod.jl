
export MultirateRungeKutta

LSRK2N = LowStorageRungeKutta2N

"""
    MultirateRungeKutta(slow_solver, fast_solver; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multirate Runge-Kutta scheme using two different RK
solvers. This is based on

Currently only the low storage RK methods can be used as slow solvers

### References

    @article{SchlegelKnothArnoldWolke2012,
      title={Implementation of multirate time integration methods for air
             pollution modelling},
      author={Schlegel, M and Knoth, O and Arnold, M and Wolke, R},
      journal={Geoscientific Model Development},
      volume={5},
      number={6},
      pages={1395--1405},
      year={2012},
      publisher={Copernicus GmbH}
    }
"""
mutable struct MultirateRungeKutta{SS, FS, RT} <: AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT

    function MultirateRungeKutta(
        slow_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        dt = getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))
        new{SS, FS, RT}(slow_solver, fast_solver, RT(dt), RT(t0))
    end
end

function MultirateRungeKutta(
    solvers::Tuple,
    Q = nothing;
    dt = getdt(solvers[1]),
    t0 = solvers[1].t,
) where {AT <: AbstractArray}
    if length(solvers) < 2
        error("Must specify atleast two solvers")
    elseif length(solvers) == 2
        fast_solver = solvers[2]
    else
        fast_solver = MultirateRungeKutta(solvers[2:end], Q; dt = dt, t0 = t0)
    end

    slow_solver = solvers[1]

    MultirateRungeKutta(slow_solver, fast_solver, Q; dt = dt, t0 = t0)
end

function dostep!(
    Q,
    mrrk::MultirateRungeKutta{SS},
    param,
    time,
    in_slow_δ = nothing,
    in_slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
) where {SS <: LSRK2N}
    dt = mrrk.dt

    slow = mrrk.slow_solver
    fast = mrrk.fast_solver

    slow_rv_dQ = realview(slow.dQ)

    groupsize = 256

    fast_dt_in = getdt(fast)

    for slow_s in 1:length(slow.RKA)
        # Currnent slow state time
        slow_stage_time = time + slow.RKC[slow_s] * dt

        # Evaluate the slow mode
        slow.rhs!(slow.dQ, Q, param, slow_stage_time, increment = true)

        if in_slow_δ !== nothing
            slow_scaling = nothing
            if slow_s == length(slow.RKA)
                slow_scaling = in_slow_scaling
            end
            # update solution and scale RHS
            event = Event(array_device(Q))
            event = update!(array_device(Q), groupsize)(
                slow_rv_dQ,
                in_slow_rv_dQ,
                in_slow_δ,
                slow_scaling;
                ndrange = length(realview(Q)),
                dependencies = (event,),
            )
            wait(array_device(Q), event)
        end

        # Fractional time for slow stage
        if slow_s == length(slow.RKA)
            γ = 1 - slow.RKC[slow_s]
        else
            γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
        end

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        slow_δ = slow.RKB[slow_s] / (γ)

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        nsubsteps = fast_dt_in > 0 ? ceil(Int, γ * dt / fast_dt_in) : 1
        fast_dt = γ * dt / nsubsteps

        updatedt!(fast, fast_dt)

        for substep in 1:nsubsteps
            slow_rka = nothing
            if substep == nsubsteps
                slow_rka = slow.RKA[slow_s % length(slow.RKA) + 1]
            end
            fast_time = slow_stage_time + (substep - 1) * fast_dt
            dostep!(Q, fast, param, fast_time, slow_δ, slow_rv_dQ, slow_rka)
        end
    end
    updatedt!(fast, fast_dt_in)
end

@kernel function update!(fast_dQ, slow_dQ, δ, slow_rka = nothing)
    i = @index(Global, Linear)
    @inbounds begin
        fast_dQ[i] += δ * slow_dQ[i]
        if slow_rka !== nothing
            slow_dQ[i] *= slow_rka
        end
    end
end
