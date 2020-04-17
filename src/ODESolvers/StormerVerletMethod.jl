export StormerVerlet, StromerVerletHEVI

abstract type AbstractStormerVerlet <: AbstractODESolver end

updatedt!(sv::AbstractStormerVerlet, dt) = sv.dt[1] = dt
#=
"""
    ODESolvers.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                       adjustfinalstep::Bool)
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function dostep!(Q, sv::AbstractStormerVerlet, p, timeend::Real,
                      adjustfinalstep::Bool, slow_δ, slow_rv_dQ, slow_rka)
  time, dt = sv.t[1], sv.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
  end
  @assert dt > 0
  dostep!(Q, sv, p, time, dt, slow_δ, slow_rv_dQ, slow_rka)
  if dt == sv.dt[1]
    sv.t[1] += dt
  else
    sv.t[1] = timeend
  end
end
=#

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
struct StormerVerlet{N, T, RT, AT} <: AbstractStormerVerlet
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs!

  mask_a
  mask_b

  gamma::RT

  dQ::AT
  function StormerVerlet(rhs!::TimeScaledRHS{N,RT} where {RT}, mask_a, mask_b, Q::AT; dt=0, t0=0, gamma=0.0) where {N,AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    dQ = similar(Q)
    fill!(dQ, 0)

    new{N, T, RT, AT}(dt, t0, rhs!, mask_a, mask_b, gamma, dQ)
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
function dostep!(Q, sv::StormerVerlet{1,T,RT,AT} where {T,RT,AT}, p, time::Real,
                      dt::Real, nsteps::Int, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ
  gamma = sv.gamma

  Qa = realview(Q.realdata[:,sv.mask_a,:])
  Qb = realview(Q.realdata[:,sv.mask_b,:])
  dQa = realview(dQ.realdata[:,sv.mask_a,:])
  dQb = realview(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = realview(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = realview(slow_rv_dQ[:,sv.mask_b,:])

  groupsize = 256

  # do a half step
  rhs!(dQ, Q, p, time, increment = false)
  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
      dQa,
      Qa,
      dt/2,
      slow_δ,
      slow_rv_dQa;
      ndrange = length(Qa),
      dependencies = (event,),
  )
  wait(device(Q), event)
  time += dt/2

  for i = 1:nsteps
    rhs!(dQ, Q, p, time, increment = false)
    event = Event(device(Q))
    event = update!(device(Q), groupsize)(
        dQb,
        Qb,
        dt,
        slow_δ,
        slow_rv_dQb;
        ndrange = length(Qb),
        dependencies = (event,),
    )
    wait(device(Q), event)
    time += dt

    rhs!(dQ, Q, p, time, increment = false)
    if i < nsteps
      event = Event(device(Q))
      event = update!(device(Q), groupsize)(
          dQa,
          Qa,
          dt,
          slow_δ,
          slow_rv_dQa;
          ndrange = length(Qa),
          dependencies = (event,),
      )
      wait(device(Q), event)
      time += dt
    else
      event = Event(device(Q))
      event = update!(device(Q), groupsize)(
          dQa,
          Qa,
          dt/2,
          slow_δ,
          slow_rv_dQa;
          ndrange = length(Qa),
          dependencies = (event,),
      )
      wait(device(Q), event)
      time += dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
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
function dostep!(Q, sv::StormerVerlet{2,T,RT,AT} where {T,RT,AT}, p, time::Real,
                      dt::Real, nsteps::Int, slow_δ, slow_rv_dQ, slow_rka)

  rhs!, dQ = sv.rhs!, sv.dQ
  #gamma = sv.gamma

  Qa = realview(Q.realdata[:,sv.mask_a,:])
  Qb = realview(Q.realdata[:,sv.mask_b,:])
  dQa = realview(dQ.realdata[:,sv.mask_a,:])
  dQb = realview(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = realview(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = realview(slow_rv_dQ[:,sv.mask_b,:])

  #QOld=copy(Q);

  groupsize = 256

  # do a half step
  rhs!(dQ, Q, p, time, 2, increment = false) #Thermo
  event = Event(device(Q))
  event = update!(device(Q), groupsize)(
      dQa,
      Qa,
      dt/2,
      slow_δ,
      slow_rv_dQa;
      ndrange = length(Qa),
      dependencies = (event,),
  )
  wait(device(Q), event)
  time += dt/2

  for i = 1:nsteps
    rhs!(dQ, Q, p, time, 1, increment = false) #Momentum
    #rhs!(dQ, (1+gamma)*Q-gamma*QOld, p, time, 1, increment = false) #Momentum
    event = Event(device(Q))
    event = update!(device(Q), groupsize)(
        dQb,
        Qb,
        dt,
        slow_δ,
        slow_rv_dQb;
        ndrange = length(Qb),
        dependencies = (event,),
    )
    wait(device(Q), event)
    time += dt
    #copy!(QOld,Q)
    rhs!(dQ, Q, p, time, 2, increment = false) #Thermo
    if i < nsteps
      event = Event(device(Q))
      event = update!(device(Q), groupsize)(
          dQa,
          Qa,
          dt,
          slow_δ,
          slow_rv_dQa;
          ndrange = length(Qa),
          dependencies = (event,),
      )
      wait(device(Q), event)
      time += dt
    else
      event = Event(device(Q))
      event = update!(device(Q), groupsize)(
          dQa,
          Qa,
          dt/2,
          slow_δ,
          slow_rv_dQa;
          ndrange = length(Qa),
          dependencies = (event,),
      )
      wait(device(Q), event)
      time += dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end


@kernel function update!(
    dQ,
    Q,
    dt,
    slow_δ,
    slow_rv_dQ,
)
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ === nothing
            Q[i] += dQ[i] * dt
        else
            Q[i] += (dQ[i] + slow_rv_dQ[i] * slow_δ) * dt
        end
    end
end




struct StormerVerletHEVI{T, RT, AT} <: AbstractStormerVerlet
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs_h!
  rhs_v!

  A_v

  mask_a
  mask_b

  dQ::AT
  function StormerVerletHEVI(rhs_h!, rhs_v!, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    A_v = banded_matrix(rhs_v!, similar(Q), similar(Q))

    dQ = similar(Q)
    fill!(dQ, 0)

    new{T, RT, AT}(dt, t0, rhs_h!, rhs_v!, A_v, mask_a, mask_b, dQ)
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
function dostep!(Q, sv::StormerVerletHEVI, p, time::Real,
                      dt::Real, nsteps::Int, slow_δ, slow_rv_dQ, slow_rka)

  rhs_h!, rhs_v!, dQ, A_v = sv.rhs_h!, sv.rhs_v!, sv.dQ, sv.A_v

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])


  # do a half step
  banded_matrix_vector_product!(rhs_v!, A_v, dQ,Q)
  rhs_h!(dQ, Q, p, time, increment = true)
  if slow_δ === nothing
    Qa .+= dQa .* dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
  end
  time += dt/2

  for i = 1:nsteps
    banded_matrix_vector_product!(rhs_v!, A_v, dQ,Q)
    rhs_h!(dQ, Q, p, time, increment = true)
    if slow_δ === nothing
      Qb .+= dQb .* dt
    else
      Qb .+= (dQb .+ slow_rv_dQb .* slow_δ) .* dt
    end
    time += dt

    banded_matrix_vector_product!(rhs_v!, A_v, dQ,Q)
    rhs_h!(dQ, Q, p, time, increment = true)
    if i < nsteps
      if slow_δ === nothing
        Qa .+= dQa .* dt
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt
      end
      time += dt
    else
      if slow_δ === nothing
        Qa .+= dQa .* dt/2
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* dt/2
      end
      time += dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end
