
include("MultirateRungeKuttaMethod_kernels.jl")

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

  function MultirateRungeKutta(slow_solver::LSRK2N,
                               fast_solver,
                               Q=nothing;
                               dt=getdt(slow_solver), t0=slow_solver.t
                              ) where {AT<:AbstractArray}
    SS = typeof(slow_solver)
    FS = typeof(fast_solver)
    RT = real(eltype(slow_solver.dQ))
    new{SS, FS, RT}(slow_solver, fast_solver, RT(dt), RT(t0))
  end
end

function MultirateRungeKutta(solvers::Tuple, Q=nothing;
                             dt=getdt(solvers[1]), t0=solvers[1].t
                            ) where {AT<:AbstractArray}
  if length(solvers) < 2
    error("Must specify atleast two solvers")
  elseif length(solvers) == 2
    fast_solver = solvers[2]
  else
    fast_solver = MultirateRungeKutta(solvers[2:end], Q; dt = dt, t0=t0)
  end

  slow_solver = solvers[1]

  MultirateRungeKutta(slow_solver, fast_solver, Q; dt = dt, t0=t0)
end

function dostep!(Q, mrrk::MultirateRungeKutta, param,
                 timeend::Real, adjustfinalstep::Bool)
  time, dt = mrrk.t, mrrk.dt
  @assert dt > 0
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  dostep!(Q, mrrk, param, time, dt)

  if dt == mrrk.dt
    mrrk.t += dt
  else
    mrrk.t = timeend
  end
  return mrrk.t
end

function dostep!(Q, mrrk::MultirateRungeKutta{SS}, param, time::Real,
                 dt::AbstractFloat, in_slow_δ = nothing,
                 in_slow_rv_dQ = nothing,
                 in_slow_scaling = nothing) where {SS <: LSRK2N}
  slow = mrrk.slow_solver
  fast = mrrk.fast_solver

  slow_rv_dQ = realview(slow.dQ)

  threads = 256
  blocks = div(length(realview(Q)) + threads - 1, threads)

  for slow_s = 1:length(slow.RKA)
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
      @launch(device(Q), threads=threads, blocks=blocks,
              update!(slow_rv_dQ, in_slow_rv_dQ, in_slow_δ, slow_scaling))
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
    nsubsteps = getdt(fast) > 0 ? ceil(Int, γ * dt / getdt(fast)) : 1
    fast_dt = γ * dt / nsubsteps

    for substep = 1:nsubsteps
      slow_rka = nothing
      if substep == nsubsteps
        slow_rka = slow.RKA[slow_s%length(slow.RKA) + 1]
      end
      fast_time = slow_stage_time + (substep - 1) * fast_dt
      dostep!(Q, fast, param, fast_time, fast_dt, slow_δ, slow_rv_dQ,
              slow_rka)
    end
  end
end

