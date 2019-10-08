module MultirateInfinitesimalStepMethod
using ..ODESolvers
using ..AdditiveRungeKuttaMethod
using ..LowStorageRungeKuttaMethod
using ..StrongStabilityPreservingRungeKuttaMethod
using ..MPIStateArrays: device, realview

using StaticArrays

struct OffsetRHS{R,O}
  rhs!::R
  "storage for offset term"
  offset::O
end

function (o::OffsetRHS)(dQ, Q, params, t)
  copyto!(dQ, o.offset)
  o.rhs!(dQ, Q, params, r; increment=true)
end


using GPUifyLoops

struct MultirateInfinitesimalStep{T, RT, AT, OR, N1, N2} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!
  "storage for y_n"
  yn::A
  "Storage for ``Y_nj - y_n``"
  deltaYnj::NTuple{Nstages,A}
  "Storage for ``f(Y_nj)``"
  fYnj::NTuple{Nstages,A}
  "RHS for fast solver"
  fastrhs!::OffsetRHS{F,A}
  fastsolver

  beta::NTuple{N1, RT}
  alpha::NTuple{N2, RT}
  gamma::NTuple{N2, RT}

  function MultirateInfinitesimalStep(rhs!, RKA, RKB, RKC,
                                  Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]

    dQ = similar(Q)
    fill!(dQ, 0)
    
    new{T, RT, AT, length(RKA)}(dt, t0, rhs!, dQ, RKA, RKB, RKC)
  end
end

function MultirateInfinitesimalStep(spacedisc::AbstractSpaceMethod, RKA, RKB, RKC,
                                Q::AT; dt=0, t0=0) where {AT<:AbstractArray}
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
  MultirateInfinitesimalStep(rhs!, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end


function MIS2(F, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}
  T = eltype(Q)
  RT = real(T)

  0.126848494553
  -0.784838278826  1.37442675268
  
  ...
end

function ODEs.dostep!(Q, mis::MultirateInfinitesimalStep{SS}, param,
  time::AbstractFloat, dt::AbstractFloat,
  in_slow_δ = nothing, in_slow_rv_dQ = nothing,
  in_slow_scaling = nothing) where {SS <: LSRK2N}

  copyto!(mis.yn, Q)
  for i = 1:??
    t += ??

    slow.rhs!(fYnj[i], Q, params, t)
    # if not last loop
    deltaYnj[i] .= Q .- mis.yn

    # TODO write a kernel to do this
    Q .= mis.yn .+ sum(j -> alpha[i,j].*deltaYnj[j], 1:i)  # (1a) 
    fastrhs.offset = sum(j -> gamma[i,j].*deltaYnj[j].+beta[i,j].*fYnj[j], 1:i)  # (1b) 

    tau = t
    # solve!(Q, fast, param; timeend = t+ ??) (1c)
    while tau < t + ??
      dostep!(Q, fast, param, tau, dt)
    end
  end



end
