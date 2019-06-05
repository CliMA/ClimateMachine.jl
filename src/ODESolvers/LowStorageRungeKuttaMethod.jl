module LowStorageRungeKuttaMethod
export LowStorageRungeKutta, updatedt!

using GPUifyLoops
include("LowStorageRungeKuttaMethod_kernels.jl")

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
struct LowStorageRungeKutta{T, RT, AT, Nstages} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!::Function
  "Storage for RHS during the LowStorageRungeKutta update"
  dQ::AT
  "low storage RK coefficient vector A (rhs scaling)"
  RKA::NTuple{Nstages, RT}
  "low storage RK coefficient vector B (rhs add in scaling)"
  RKB::NTuple{Nstages, RT}
  "low storage RK coefficient vector C (time scaling)"
  RKC::NTuple{Nstages, RT}
  function LowStorageRungeKutta(rhs!::Function, Q::AT; dt=nothing,
                                t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    RT = real(T)
    dt = [T(dt)]
    t0 = [T(t0)]
    # FIXME: Add reference
    RKA = (RT(0),
           RT(-567301805773  // 1357537059087),
           RT(-2404267990393 // 2016746695238),
           RT(-3550918686646 // 2091501179385),
           RT(-1275806237668 // 842570457699 ))

    RKB = (RT(1432997174477 // 9575080441755 ),
           RT(5161836677717 // 13612068292357),
           RT(1720146321549 // 2090206949498 ),
           RT(3134564353537 // 4481467310338 ),
           RT(2277821191437 // 14882151754819))

    RKC = (RT(0),
           RT(1432997174477 // 9575080441755),
           RT(2526269341429 // 6820363962896),
           RT(2006345519317 // 3224310063776),
           RT(2802321613138 // 2924317926251))

    dQ = similar(Q)
    fill!(dQ, 0)
    new{T, RT, AT, length(RKA)}(dt, t0, rhs!, dQ, RKA, RKB, RKC)
  end
end

function LowStorageRungeKutta(spacedisc::AbstractSpaceMethod, Q; dt=nothing,
                              t0=0)
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
  LowStorageRungeKutta(rhs!, Q; dt=dt, t0=t0)
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

  rv_Q = ODEs.realview(Q)
  rv_dQ = ODEs.realview(dQ)

  threads = 1024
  blocks = div(length(rv_Q) + threads - 1, threads)

  for s = 1:length(RKA)
    rhs!(dQ, Q, time + RKC[s] * dt, increment = true)
    # update solution and scale RHS
    @launch(device(Q), threads=threads, blocks=blocks,
            update!(rv_dQ, rv_Q, RKA[s%length(RKA)+1], RKB[s], dt))
  end
  if dt == lsrk.dt[1]
    lsrk.t[1] += dt
  else
    lsrk.t[1] = timeend
  end

end

end
