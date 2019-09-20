module MultirateRungeKuttaMethod
using ..ODESolvers
ODEs = ODESolvers
using ..LowStorageRungeKuttaMethod
using ..MPIStateArrays: device, realview

using GPUifyLoops
include("MultirateRungeKuttaMethod_kernels.jl")

export MultirateRungeKutta

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

Currently only the following low storage RK methods can be used as fast and slow
solvers

  - [`LSRK54CarpenterKennedy`](@ref)
  - [`LSRK144NiegemannDiehlBusch`](@ref)

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
struct MultirateRungeKutta{SS, FS, RT} <: ODEs.AbstractODESolver
  "slow solver"
  slow_solver::SS
  "fast solver"
  fast_solver::FS
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}

  function MultirateRungeKutta(slow_solver::LowStorageRungeKutta2N,
                               fast_solver::LowStorageRungeKutta2N,
                               Q=nothing;
                               dt=0, t0=slow_solver.t[1]
                              ) where {AT<:AbstractArray}
    SS = typeof(slow_solver)
    FS = typeof(fast_solver)
    RT = real(eltype(slow_solver.dQ))
    new{SS, FS, RT}(slow_solver, fast_solver, [dt], [t0])
  end
end

ODEs.updatedt!(mrrk::MultirateRungeKutta, dt) = mrrk.dt[1] = dt

function ODEs.dostep!(Q, mrrk::MultirateRungeKutta{SS, FS}, param, timeend,
                      adjustfinalstep) where {SS <: LowStorageRungeKutta2N,
                                              FS <: LowStorageRungeKutta2N}
  slow_param = param[1]
  fast_param = param[2]
  time, dt = mrrk.t[1], mrrk.dt[1]
  @assert dt > 0
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  slow = mrrk.slow_solver
  fast = mrrk.fast_solver

  rv_Q = realview(Q)
  slow_rv_dQ= realview(slow.dQ)
  fast_rv_dQ = realview(fast.dQ)

  threads = 256
  blocks = div(length(rv_Q) + threads - 1, threads)

  for slow_s = 1:length(slow.RKA)
    # Currnent slow state time
    slow_stage_time = time + slow.RKC[slow_s] * dt

    # Evaluate the slow mode
    slow.rhs!(slow.dQ, Q, slow_param, slow_stage_time, increment = true)

    # Fractional time for slow stage
    if slow_s == length(slow.RKA)
      γ = 1 - slow.RKC[slow_s]
    else
      γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
    end

    # RKB for the slow with fractional time factor remove (since full
    # integration of fast will result in scaling by γ)
    δ = slow.RKB[slow_s] / γ

    # Fast update
    for fast_s = 1:length(fast.RKA)
      fast_stage_time = slow_stage_time + γ * fast.RKC[fast_s] * dt

      # Evaluate the fast mode
      fast.rhs!(fast.dQ, Q, fast_param, fast_stage_time, increment = true)

      # On laste update we also need to scale slow_dQ and this catches that
      slow_rka = nothing
      if fast_s == length(fast.RKA)
        slow_rka = slow.RKA[slow_s%length(slow.RKA) + 1]
      end

      # update solution and scale RHS
      @launch(device(Q), threads=threads, blocks=blocks,
              update!(fast_rv_dQ, slow_rv_dQ, rv_Q, δ,
                      fast.RKA[fast_s%length(fast.RKA) + 1],
                      γ * fast.RKB[fast_s], dt, slow_rka))
    end
  end

  if dt == mrrk.dt[1]
    mrrk.t[1] += dt
  else
    mrrk.t[1] = timeend
  end
  return mrrk.t[1]
end

end
