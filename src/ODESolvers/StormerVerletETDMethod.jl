export StormerVerletETD

include("StormerVerletETD_kernel.jl")

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
struct StormerVerletETD{T, RT, AT} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!

  max_inner_dt::RT

  mask_a
  mask_b

  offset1
  offset0

  dQ::AT
  function StormerVerletETD(rhs!, max_inner_dt, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

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

    new{T, RT, AT}(dt, t0, rhs!, max_inner_dt, mask_a, mask_b, offset1, offset0, dQ)
  end
end

ODEs.updatedt!(sv::StormerVerletETD, dt) = sv.dt[1] = dt

"""
    ODESolvers.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                       adjustfinalstep::Bool)
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function ODEs.dostep!(Q, sv::StormerVerletETD, p, timeend::Real,
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
function ODEs.dostep!(Q, sv::StormerVerletETD, p, nsLoc::Int, time::Real,
                      dτ::Real, iStage, βS, β, nPhi, fYnj, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])


  offset1=sv.offset1;
  offset0=sv.offset0;
  offset1a = @view(offset1.realdata[:,sv.mask_a,:]);
  offset1b = @view(offset1.realdata[:,sv.mask_b,:]);
  offset0a = @view(offset0.realdata[:,sv.mask_a,:]);
  offset0b = @view(offset0.realdata[:,sv.mask_b,:]);

  τ=0.0;
  dTime=nsLoc*dτ;

    threads = 256
    blocks = div(length(realview(Q)) + threads - 1, threads)
    @launch(device(Q), threads=threads, blocks=blocks,
            update!(realview(offset1), Val(iStage), map(realview, fYnj[1:iStage]),
            βS[iStage+1,:], τ, nPhi))

  rhs!(dQ, Q, p, time, increment = false)
  Qa .+= (offset1a .+ dQa) .* dτ/2

  τ += dτ/dTime;

  for i = 1:nsLoc-1

    offset0=deepcopy(offset1);
    offset0a = @view(offset0.realdata[:,sv.mask_a,:]);
    offset0b = @view(offset0.realdata[:,sv.mask_b,:]);

    @launch(device(Q), threads=threads, blocks=blocks,
            update!(realview(offset1), Val(iStage), map(realview, fYnj[1:iStage]),
            βS[iStage+1,:], τ, nPhi))

    rhs!(dQ, Q, p, time, increment = false)
    Qb .+= (0.5*(offset0b.+offset1b) .+ dQb) .* dτ

    rhs!(dQ, Q, p, time, increment = false) #increment = true? damit auf dQ draufaddiert
    Qa .+= (offset1a .+ dQa) .* dτ

    τ += dτ/dTime
  end

  if τ-1.0>0.01
    error("tau ist nicht 1, sondern: $τ !!")
  end

  offset0=deepcopy(offset1);
  offset0a = @view(offset0.realdata[:,sv.mask_a,:]);
  offset0b = @view(offset0.realdata[:,sv.mask_b,:]);

  @launch(device(Q), threads=threads, blocks=blocks,
          update!(realview(offset1), Val(iStage), map(realview, fYnj[1:iStage]),
          βS[iStage+1,:], τ, nPhi))

  rhs!(dQ, Q, p, time, increment = false)
  Qb .+= (0.5*(offset0b.+offset1b) .+ dQb) .* dτ

  rhs!(dQ, Q, p, time, increment = false)
  Qa .+= (offset1a .+ dQa) .* dτ/2

  if slow_rka !== nothing
    println("last case happened")
    slow_rv_dQ .*= slow_rka
  end

end









#=
function y=VerletETDLin(y,FY,ns,dTau,Time,J,ETD,iStage)
global Param
nFast=Param.nFast;
n=size(y,1);
TimeLoc=Time;
dTime=ns*dTau;
tau=0.0e0;
gamma=Param.gamma;
Sl1=Slow(FY,tau,ETD,iStage);
yy=y(nFast+1:n);
y(nFast+1:n)=y(nFast+1:n)+0.5*dTau*(J(nFast+1:n,1:nFast)*y(1:nFast)+...
  Sl1(nFast+1:n));
tau=tau+dTau/dTime;
for is=1:ns-1
  Sl0=Sl1;
  Sl1=Slow(FY,tau,ETD,iStage);
  y(1:nFast)=y(1:nFast)+dTau*(J(1:nFast,nFast+1:n)*((1+gamma)*y(nFast+1:n)-gamma*yy)+...
    0.5*(Sl0(1:nFast)+Sl1(1:nFast)));
  yy=y(nFast+1:n);
  y(nFast+1:n)=y(nFast+1:n)+dTau*(J(nFast+1:n,1:nFast)*y(1:nFast)+...
    Sl1(nFast+1:n));
  tau=tau+dTau/dTime;
end
Sl0=Sl1;
Sl1=Slow(FY,tau,ETD,iStage);
y(1:nFast)=y(1:nFast)+dTau*(J(1:nFast,nFast+1:n)*((1+gamma)*y(nFast+1:n)-gamma*yy)+...
  0.5*(Sl0(1:nFast)+Sl1(1:nFast)));
y(nFast+1:n)=y(nFast+1:n)+0.5*dTau*(J(nFast+1:n,1:nFast)*y(1:nFast)+...
  Sl1(nFast+1:n));
end
=#
