module StormerVerletMethod
export StormerVerlet, StormerVerletMoTh, StromerVerletHEVI

using ..ODESolvers
const ODEs = ODESolvers
using ..SpaceMethods
using ..MPIStateArrays: device, realview
using CLIMA.ColumnwiseLUSolver: banded_matrix, banded_matrix_vector_product!


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

  max_inner_dt::RT

  mask_a
  mask_b

  dQ::AT
  function StormerVerlet(rhs!, max_inner_dt, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    dQ = similar(Q)
    fill!(dQ, 0)

    new{T, RT, AT}(dt, t0, rhs!, max_inner_dt, mask_a, mask_b, dQ)
  end
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

  nsteps = cld(dt, sv.max_inner_dt)
  inner_dt = dt / nsteps

  # do a half step
  rhs!(dQ, Q, p, time, increment = false)
  if slow_δ === nothing
    Qa .+= dQa .* inner_dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* inner_dt/2
  end
  time += inner_dt/2

  for i = 1:nsteps
    rhs!(dQ, Q, p, time, increment = false)
    if slow_δ === nothing
      Qb .+= dQb .* inner_dt
    else
      Qb .+= (dQb .+ slow_rv_dQb .* slow_δ) .* inner_dt
    end
    time += inner_dt

    rhs!(dQ, Q, p, time, increment = false)
    if i < nsteps
      if slow_δ === nothing
        Qa .+= dQa .* inner_dt
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* inner_dt
      end
      time += inner_dt
    else
      if slow_δ === nothing
        Qa .+= dQa .* inner_dt/2
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* inner_dt/2
      end
      time += inner_dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
  println(sqrt(sum(Q.^2)))
end




struct StormerVerletMoTh{T, RT, AT} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs_momentum!
  rhs_thermo!

  max_inner_dt::RT

  mask_a
  mask_b

  dQ::AT
  function StormerVerletMoTh(rhs_momentum!, rhs_thermo!, max_inner_dt, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    dQ = similar(Q)
    fill!(dQ, 0)

    new{T, RT, AT}(dt, t0, rhs_momentum!, rhs_thermo!, max_inner_dt, mask_a, mask_b, dQ)
  end
end

ODEs.updatedt!(sv::StormerVerletMoTh, dt) = sv.dt[1] = dt

"""
    ODESolvers.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                       adjustfinalstep::Bool)
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function ODEs.dostep!(Q, sv::StormerVerletMoTh, p, timeend::Real,
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
function ODEs.dostep!(Q, sv::StormerVerletMoTh, p, time::Real,
                      dt::Real, slow_δ, slow_rv_dQ, slow_rka)

  rhs_momentum!, rhs_thermo!, dQ = sv.rhs_momentum!, sv.rhs_thermo!, sv.dQ

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])

  nsteps = cld(dt, sv.max_inner_dt)
  inner_dt = dt / nsteps

  # do a half step
  rhs_thermo!(dQ, Q, p, time, increment = false)
  if slow_δ === nothing
    @. Qa += dQa * inner_dt/2
  else
    @. Qa += (dQa + slow_rv_dQa * slow_δ) * inner_dt/2
  end
  time += inner_dt/2

  for i = 1:nsteps
    rhs_momentum!(dQ, Q, p, time, increment = false)
    if slow_δ === nothing
      @. Qb += dQb * inner_dt
    else
      @. Qb += (dQb + slow_rv_dQb * slow_δ) * inner_dt
    end
    time += inner_dt

    rhs_thermo!(dQ, Q, p, time, increment = false)
    if i < nsteps
      if slow_δ === nothing
        @. Qa += dQa * inner_dt
      else
        @. Qa += (dQa + slow_rv_dQa * slow_δ) * inner_dt
      end
      time += inner_dt
    else
      if slow_δ === nothing
        @. Qa += dQa * inner_dt/2
      else
        @. Qa += (dQa + slow_rv_dQa * slow_δ) * inner_dt/2
      end
      time += inner_dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
end


struct StormerVerletHEVI{T, RT, AT} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs_h!
  rhs_v!

  A_v

  max_inner_dt::RT

  mask_a
  mask_b

  dQ::AT
  dQ_h::AT
  dQ_v::AT
  function StormerVerletHEVI(rhs_h!, rhs_v!, max_inner_dt, mask_a, mask_b, Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    A_v = banded_matrix(rhs_v!, similar(Q), similar(Q))

    dQ = similar(Q)
    dQ_h = similar(Q)
    dQ_v = similar(Q)
    fill!(dQ, 0)
    fill!(dQ_h, 0)
    fill!(dQ_v, 0)

    new{T, RT, AT}(dt, t0, rhs_h!, rhs_v!, A_v, max_inner_dt, mask_a, mask_b, dQ, dQ_h, dQ_v)
  end
end

ODEs.updatedt!(sv::StormerVerletHEVI, dt) = sv.dt[1] = dt

"""
    ODESolvers.dostep!(Q, sv::StormerVerlet, p, timeend::Real,
                       adjustfinalstep::Bool)
Use the 2N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function ODEs.dostep!(Q, sv::StormerVerletHEVI, p, timeend::Real,
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
function ODEs.dostep!(Q, sv::StormerVerletHEVI, p, time::Real,
                      dt::Real, slow_δ, slow_rv_dQ, slow_rka)

  rhs_h!, rhs_v!, dQ, A_v = sv.rhs_h!, sv.rhs_v!, sv.dQ, sv.A_v
  dQ_h=sv.dQ_h
  dQ_v=sv.dQ_v
  #rhs!, dQ = sv.rhs!, sv.dQ

  Qa = @view(Q.realdata[:,sv.mask_a,:])
  Qb = @view(Q.realdata[:,sv.mask_b,:])
  dQa = @view(dQ.realdata[:,sv.mask_a,:])
  dQb = @view(dQ.realdata[:,sv.mask_b,:])
  slow_rv_dQa = @view(slow_rv_dQ[:,sv.mask_a,:])
  slow_rv_dQb = @view(slow_rv_dQ[:,sv.mask_b,:])

  nsteps = cld(dt, sv.max_inner_dt)
  inner_dt = dt / nsteps

  # do a half step
  rhs_h!(dQ, Q, p, time, increment = false)
  banded_matrix_vector_product!(rhs_v!, A_v, dQ_v,Q)
  dQ+=dQ_v
  if slow_δ === nothing
    Qa .+= dQa .* inner_dt/2
  else
    Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* inner_dt/2
  end
  time += inner_dt/2

  for i = 1:nsteps
    rhs_h!(dQ, Q, p, time, increment = false)
    banded_matrix_vector_product!(rhs_v!, A_v, dQ_v,Q)
    dQ+=dQ_v
    if slow_δ === nothing
      Qb .+= dQb .* inner_dt
    else
      Qb .+= (dQb .+ slow_rv_dQb .* slow_δ) .* inner_dt
    end
    time += inner_dt

    rhs_h!(dQ, Q, p, time, increment = false)
    banded_matrix_vector_product!(rhs_v!, A_v, dQ_v,Q)
    dQ+=dQ_v
    if i < nsteps
      if slow_δ === nothing
        Qa .+= dQa .* inner_dt
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* inner_dt
      end
      time += inner_dt
    else
      if slow_δ === nothing
        Qa .+= dQa .* inner_dt/2
      else
        Qa .+= (dQa .+ slow_rv_dQa .* slow_δ) .* inner_dt/2
      end
      time += inner_dt/2
    end
  end
  if slow_rka !== nothing
    slow_rv_dQ .*= slow_rka
  end
  println(sqrt(sum(Q.^2)))
end
end
