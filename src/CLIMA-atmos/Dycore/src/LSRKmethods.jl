module LSRKmethods
export LSRK, updatedt!

using Requires

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("LSRK_cuda.jl")
end

using ..CLIMAAtmosDycore
AD = CLIMAAtmosDycore

struct LSRK{T, AT, Nstages, F<:Function} <: AD.AbstractAtmosODESolver
  "time step"
  dt::Array{T,1}
  "time"
  t::Array{T,1}
  "rhs function"
  rhs!::F
  "Storage for RHS during the LSRK update"
  dQ::AT
  "low storage RK coefficient vector A (rhs scaling)"
  RKA::NTuple{Nstages, T}
  "low storage RK coefficient vector B (rhs add in scaling)"
  RKB::NTuple{Nstages, T}
  "low storage RK coefficient vector C (time scaling)"
  RKC::NTuple{Nstages, T}
  function LSRK(dQ, Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

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

    new{T, AT, length(RKA), typeof(dQ)}(dt, t0, dQ, similar(Q), RKA, RKB, RKC)
  end
end

updatedt!(lsrk::LSRK, dt) = lsrk.dt[1] = dt
function AD.dostep!(Q, lsrk::LSRK)
  time, dt = lsrk.t[1], lsrk.dt[1]
  RKA, RKB, RKC = lsrk.RKA, lsrk.RKB, lsrk.RKC
  rhs!, dQ = lsrk.rhs!, lsrk.dQ
  for s = 1:length(RKA)
    rhs!(dQ, Q, time)

    # update solution and scale RHS
    # FIXME: GPUify
    # FIXME: Figure out how to properly use our new AtmosStateArrays
    update!(Val(size(Q,2)), Val(size(Q,1)), dQ.Q, Q.Q, Q.realelems,
            RKA[s%length(RKA)+1], RKB[s], dt)
    time += RKC[s] * dt
  end
  lsrk.t[1] += dt
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
