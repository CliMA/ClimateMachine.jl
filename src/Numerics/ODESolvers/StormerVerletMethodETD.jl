export StormerVerletETD

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
struct StormerVerletETD{T, RT, AT} <: AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!

  mask_a
  mask_b

  offset1
  offset0

  dQ::AT
  function StormerVerletETD(rhs!, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    dQ = similar(Q)
    offset1 = similar(Q)
    offset0 = similar(Q)

    fill!(dQ, 0)
    fill!(offset1, 0)
    fill!(offset0, 0)

    new{T, RT, AT}(dt, t0, rhs!, mask_a, mask_b, offset1, offset0, dQ)
  end
end

updatedt!(sv::StormerVerletETD, dt) = sv.dt[1] = dt

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
function dostep!(Q, sv::StormerVerletETD, p, time::Real,
  dτ::Real, nsLoc::Int, iStage, βS, β, nPhi, fYnj, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ

  Qa = realview(Q[:,sv.mask_a,:])
  Qb = realview(Q[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])


  offset1=realview(sv.offset1);
  offset0=realview(sv.offset0);
  offset1a = @view(offset1[:,sv.mask_a,:]);
  offset1b = @view(offset1[:,sv.mask_b,:]);
  offset0a = @view(offset0[:,sv.mask_a,:]);
  offset0b = @view(offset0[:,sv.mask_b,:]);

  τ=0.0;
  dTime=nsLoc*dτ;


  groupsize = 256
  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
         offset1,
         Val(iStage),
         map(realview, fYnj[1:iStage]),
         βS,
         τ,
         nPhi;
         ndrange = length(offset1),
         dependencies = (event,),
  )
  wait(device(Q), event)

  rhs!(dQ, Q, p, time, increment = false)
  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
         dQa,
         Qa,
         dτ/2,
         slow_δ,
         offset1a;
         ndrange = length(Qa),
         dependencies = (event,),
  )
  wait(device(Q), event)
  Q.realdata[:,sv.mask_a,:].=Qa

  τ += dτ/dTime;

  for i = 1:nsLoc-1

    offset0=deepcopy(offset1);
    offset0a = realview(offset0[:,sv.mask_a,:]);
    offset0b = realview(offset0[:,sv.mask_b,:]);

    event = Event(device(Q))
    event = update!(device(Q), groupsize)(
           offset1,
           Val(iStage),
           map(realview, fYnj[1:iStage]),
           βS,
           τ,
           nPhi;
           ndrange = length(offset1),
           dependencies = (event,),
    )
    wait(device(Q), event)

    rhs!(dQ, Q, p, time, increment = false)
    event = Event(device(Q))
    event = update!(device(Q), groupsize)(
           dQb,
           Qb,
           dτ,
           slow_δ,
           0.5.*(offset0b.+offset1b);
           ndrange = length(Qb),
           dependencies = (event,),
    )
    wait(device(Q), event)
    Q.realdata[:,sv.mask_b,:].=Qb

    rhs!(dQ, Q, p, time, increment = false) #increment = true? damit auf dQ draufaddiert
    event = Event(device(Q))
    event = update!(device(Q), groupsize)(
           dQa,
           Qa,
           dτ,
           slow_δ,
           offset1a;
           ndrange = length(Qa),
           dependencies = (event,),
    )
    wait(device(Q), event)
    Q.realdata[:,sv.mask_a,:].=Qa

    τ += dτ/dTime
  end

  if τ-1.0>0.01
    error("tau ist nicht 1, sondern: $τ !!")
  end

  offset0=deepcopy(offset1);
  offset0a = realview(offset0[:,sv.mask_a,:]);
  offset0b = realview(offset0[:,sv.mask_b,:]);

  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
         offset1,
         Val(iStage),
         map(realview, fYnj[1:iStage]),
         βS,
         τ,
         nPhi;
         ndrange = length(offset1),
         dependencies = (event,),
  )
  wait(device(Q), event)

  rhs!(dQ, Q, p, time, increment = false)
  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
         dQb,
         Qb,
         dτ,
         slow_δ,
         0.5*(offset0b.+offset1b);
         ndrange = length(Qb),
         dependencies = (event,),
  )
  wait(device(Q), event)
  Q.realdata[:,sv.mask_b,:].=Qb

  rhs!(dQ, Q, p, time, increment = false)
  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
         dQa,
         Qa,
         dτ/2,
         slow_δ,
         offset1a;
         ndrange = length(Qa),
         dependencies = (event,),
  )
  wait(device(Q), event)
  Q.realdata[:,sv.mask_a,:].=Qa

  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end

@kernel function update!(
    offset,
    ::Val{iStage},
    fYnj,
    βS,
    τ,
    nPhi
) where {iStage}
    e = @index(Global, Linear)
    @inbounds begin
      fac = βS[nPhi][iStage+1,1];
      @unroll for k in (nPhi-1):-1:1
        fac=fac.*τ.+βS[k][iStage+1,1];
      end
      offset[e]=fac.*fYnj[1][e];

      @unroll for jStage in 2:iStage
        fac = βS[nPhi][iStage+1,jStage];

        @unroll for k in (nPhi-1):-1:1
          fac=fac.*τ.+βS[k][iStage+1,jStage];
        end
         offset[e]+=fac.*fYnj[jStage][e];
      end
    end
end
