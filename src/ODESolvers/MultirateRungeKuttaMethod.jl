module MultirateRungeKuttaMethod
using ..ODESolvers
ODEs = ODESolvers
using ..LowStorageRungeKuttaMethod
LSRK2N = LowStorageRungeKutta2N
using ..StrongStabilityPreservingRungeKuttaMethod
SSPRK = StrongStabilityPreservingRungeKutta
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

Currently only the low storage RK methods can be used as slow solvers

  - [`LowStorageRungeKuttaMethod`](@ref)

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

  function MultirateRungeKutta(slow_solver::LSRK2N,
                               fast_solver::Union{LSRK2N, SSPRK},
                               Q=nothing;
                               dt=ODEs.getdt(slow_solver), t0=slow_solver.t[1]
                              ) where {AT<:AbstractArray}
    SS = typeof(slow_solver)
    FS = typeof(fast_solver)
    RT = real(eltype(slow_solver.dQ))
    new{SS, FS, RT}(slow_solver, fast_solver, [dt], [t0])
  end
end

function ODEs.dostep!(Q, mrrk::MultirateRungeKutta{SS}, param, timeend,
                      adjustfinalstep) where {SS <: LSRK2N}
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

  slow_rv_dQ = realview(slow.dQ)

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
    slow_δ = slow.RKB[slow_s] / (γ)

    # RKB for the slow with fractional time factor remove (since full
    # integration of fast will result in scaling by γ)
    nsubsteps = ODEs.getdt(fast) > 0 ? ceil(Int, γ * dt / ODEs.getdt(fast)) : 1
    fast_dt = γ * dt / nsubsteps

    for substep = 1:nsubsteps
      slow_rka = nothing
      if substep == nsubsteps
        slow_rka = slow.RKA[slow_s%length(slow.RKA) + 1]
      end
      fast_time = slow_stage_time + (substep - 1) * fast_dt
      ODEs.dostep!(Q, fast, fast_param, fast_time, fast_dt, slow_δ, slow_rv_dQ,
                   slow_rka)
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
