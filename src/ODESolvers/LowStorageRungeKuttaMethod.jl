module LowStorageRungeKuttaMethod
export LowStorageRungeKutta, updatedt!

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative
  using .CuArrays.CUDAnative.CUDAdrv

  include("LowStorageRungeKuttaMethod_cuda.jl")
end

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

"""
    LowStorageRungeKutta(f, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

    Q̇ = f(Q)

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses the fourth-order, low-storage, Runge--Kutta scheme of Carpenter
and Kennedy (1994) (in their notation (5,4) 2N-Storage RK scheme).

### References

    @TECHREPORT{CarpenterKennedy1994,
      author = {M.~H. Carpenter and C.~A. Kennedy},
      title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},
      institution = {National Aeronautics and Space Administration},
      year = {1994},
      number = {NASA TM-109112},
      address = {Langley Research Center, Hampton, VA},
    }
"""
struct LowStorageRungeKutta{T, AT, Nstages} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{T,1}
  "time"
  t::Array{T,1}
  "rhs function"
  rhs!::Function
  "Storage for RHS during the LowStorageRungeKutta update"
  dQ::AT
  "low storage RK coefficient vector A (rhs scaling)"
  RKA::NTuple{Nstages, T}
  "low storage RK coefficient vector B (rhs add in scaling)"
  RKB::NTuple{Nstages, T}
  "low storage RK coefficient vector C (time scaling)"
  RKC::NTuple{Nstages, T}
  function LowStorageRungeKutta(rhs!::Function, Q::AT; dt=nothing,
                                t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    dt = [T(dt)]
    t0 = [T(t0)]
    # FIXME: Add reference
    RKA = (T(0),
           T(-567301805773)  / T(1357537059087),
           T(-2404267990393) / T(2016746695238),
           T(-3550918686646) / T(2091501179385),
           T(-1275806237668) / T(842570457699 ))

    RKB = (T(1432997174477) / T(9575080441755 ),
           T(5161836677717) / T(13612068292357),
           T(1720146321549) / T(2090206949498 ),
           T(3134564353537) / T(4481467310338 ),
           T(2277821191437) / T(14882151754819))

    RKC = (T(0),
           T(1432997174477) / T(9575080441755),
           T(2526269341429) / T(6820363962896),
           T(2006345519317) / T(3224310063776),
           T(2802321613138) / T(2924317926251))

    new{T, AT, length(RKA)}(dt, t0, rhs!, similar(Q), RKA, RKB, RKC)
  end
end

function LowStorageRungeKutta(spacedisc::AbstractSpaceMethod, Q; dt=nothing,
                              t0=0)
  LowStorageRungeKutta((x...) -> SpaceMethods.odefun!(spacedisc, x...),
                       Q; dt=dt, t0=t0)
end


"""
    updatedt!(lsrk::LowStorageRungeKutta, dt)

Change the time step size to `dt` for `lsrk.
"""
updatedt!(lsrk::LowStorageRungeKutta, dt) = lsrk.dt[1] = dt

function ODEs.dostep!(Q, lsrk::LowStorageRungeKutta, timeend,
                      adjustfinalstep)
  time, dt = lsrk.t[1], lsrk.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end
  RKA, RKB, RKC = lsrk.RKA, lsrk.RKB, lsrk.RKC
  rhs!, dQ = lsrk.rhs!, lsrk.dQ
  for s = 1:length(RKA)
    rhs!(dQ, Q, time)

    # update solution and scale RHS
    # FIXME: GPUify
    update!(Val(size(Q,2)), Val(size(Q,1)), dQ.Q, Q.Q, Q.realelems,
            RKA[s%length(RKA)+1], RKB[s], dt)

    time += RKC[s] * dt
  end
  if dt == lsrk.dt[1]
    lsrk.t[1] += dt
  else
    lsrk.t[1] = timeend
  end

end

# {{{ Update solution (for all dimensions)
function update!(::Val{nstates}, ::Val{Np}, rhs::Array{T, 3}, Q, elems, rka,
                 rkb, dt) where {nstates, Np, T}
  @inbounds for e = elems, s = 1:nstates, i = 1:Np
    Q[i, s, e] += rkb * dt * rhs[i, s, e]
    rhs[i, s, e] *= rka
    
  end
end
# }}}

end
