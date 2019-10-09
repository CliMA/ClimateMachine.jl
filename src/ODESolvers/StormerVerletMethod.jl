module StormerVerletMethod
export StormerVerlet

using ..ODESolvers
const ODEs = ODESolvers
using ..SpaceMethods
using ..MPIStateArrays: device, realview

"""
    LowStorageRungeKutta2N(f, RKA, RKB, RKC, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a low-storage Runge-Kutta scheme using 2N
storage based on the provided `RKA`, `RKB` and `RKC` coefficient arrays.

The available concrete implementations are:

  - [`LSRK54CarpenterKennedy`](@ref)
  - [`LSRK144NiegemannDiehlBusch`](@ref)
"""
struct StormerVerlet{T, RT, AT} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!

  mask_a
  mask_b

  dQ::AT
  function StormerVerlet(rhs!, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    dQ = similar(Q)
    fill!(dQ, 0)
    
    new{T, RT, AT}(dt, t0, rhs!, mask_a, mask_b, dQ)
  end
end

function StormerVerlet(spacedisca::AbstractSpaceMethod, 
                       spacediscb::AbstractSpaceMethod,
                       Q::AT; dt=0, t0=0) where {AT<:AbstractArray}
  rhsa! = (x...; increment) -> SpaceMethods.odefun!(spacedisca, x..., increment = increment)
  rhsb! = (x...; increment) -> SpaceMethods.odefun!(spacediscb, x..., increment = increment)
  StormerVerlet(rhsa!, rhsb!, Q; dt=dt, t0=t0)
end

ODEs.updatedt!(sv::StormerVerlet, dt) = sv.dt[1] = dt

"""
    ODESolvers.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                       adjustfinalstep::Bool)

Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function ODEs.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                      adjustfinalstep::Bool, slow_δ, slow_rv_dQ, slow_rka)
  time, dt = sv.t[1], sv.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
  end
  @assert dt > 0

  ODEs.dostep!(Q, sv, p, time, dt, slow_δ, slow_rv_dQ, slow_rka)

  if dt == sv.dt[1]
    sv.t[1] += dt
  else
    sv.t[1] = timeend
  end

end

"""
    ODESolvers.dostep!(Q, lsrk::LowStorageRungeKutta2N, p, time::Real,
                       dt::Real, [slow_δ, slow_rv_dQ, slow_scaling])

Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time `time` to final time `time + dt`.

If the optional parameter `slow_δ !== nothing` then `slow_rv_dQ * slow_δ` is
added as an additional ODE right-hand side source. If the optional parameter
`slow_scaling !== nothing` then after the final stage update the scaling
`slow_rv_dQ *= slow_scaling` is performed.
"""
function ODEs.dostep!(Q, sv::StormerVerlet, p, time::Real,
                      dt::Real, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])

  # do a half step
  rhs!(dQ, Q, p, time, increment = false)
  if slow_δ === nothing
    Qa .+= dQa .* dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
  end

  rhs!(dQ, Q, p, time, increment = false)
  if slow_δ === nothing
    Qb .+= dQb .* dt
  else
    Qb .+= (dQb .+ slow_rv_dQb .* slow_δ) .* dt
  end

  rhs!(dQ, Q, p, time, increment = false)
  if slow_δ === nothing
    Qa .+= dQa .* dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
  end

  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end

end
