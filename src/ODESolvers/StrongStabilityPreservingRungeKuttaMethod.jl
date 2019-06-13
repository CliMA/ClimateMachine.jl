module StrongStabilityPreservingRungeKuttaMethod
export StrongStabilityPreservingRungeKutta
export SSPRK33ShuOsher, SSPRK34SpiteriRuuth

using GPUifyLoops
include("StrongStabilityPreservingRungeKuttaMethod_kernels.jl")

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

"""
    StrongStabilityPreservingRungeKutta(f, RKA, RKB, RKC, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a strong-stability-preserving Runge--Kutta scheme
based on the provided `RKA`, `RKB` and `RKC` coefficient arrays.

The available concrete implementations are:

  - [`SSPRK33ShuOsher`](@ref)  
  - [`SSPRK34SpiteriRuuth`](@ref)  
"""
struct StrongStabilityPreservingRungeKutta{T, RT, AT, Nstages} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!::Function
  "Storage for RHS during the StrongStabilityPreservingRungeKutta update"
  Rstage::AT
  "Storage for the stage state during the StrongStabilityPreservingRungeKutta update"
  Qstage::AT
  "RK coefficient vector A (rhs scaling)"
  RKA::Array{RT,2}
  "RK coefficient vector B (rhs add in scaling)"
  RKB::Array{RT,1}
  "RK coefficient vector C (time scaling)"
  RKC::Array{RT,1}

  function StrongStabilityPreservingRungeKutta(rhs!::Function, RKA, RKB, RKC,
                                               Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}
    @assert dt != nothing
    
    T = eltype(Q)
    RT = real(T)
    dt = [dt]
    t0 = [t0]
    new{T, RT, AT, length(RKB)}(dt, t0, rhs!, similar(Q), similar(Q), RKA, RKB, RKC)
  end
end

function StrongStabilityPreservingRungeKutta(spacedisc::AbstractSpaceMethod, RKA, RKB, RKC,
                                             Q; dt=nothing, t0=0)
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
  StrongStabilityPreservingRungeKutta(rhs!, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

ODEs.updatedt!(ssp::StrongStabilityPreservingRungeKutta, dt) = ssp.dt[1] = dt

function ODEs.dostep!(Q, ssp::StrongStabilityPreservingRungeKutta, timeend, adjustfinalstep)
  time, dt = ssp.t[1], ssp.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end
  RKA, RKB, RKC = ssp.RKA, ssp.RKB, ssp.RKC
  rhs! = ssp.rhs!
  Rstage, Qstage = ssp.Rstage, ssp.Qstage
  
  rv_Q = ODEs.realview(Q)
  rv_Rstage = ODEs.realview(Rstage)
  rv_Qstage = ODEs.realview(Qstage)
  
  threads = 1024
  blocks = div(length(rv_Q) + threads - 1, threads)
  
  rv_Qstage .= rv_Q
  for s = 1:length(RKB)
    rhs!(Rstage, Qstage, time + RKC[s] * dt, increment = false)
  
    @launch(ODEs.device(Q), threads = threads, blocks = blocks,
            update!(rv_Rstage, rv_Q, rv_Qstage, RKA[s,1], RKA[s,2], RKB[s], dt))
  end
  rv_Q .= rv_Qstage
  
  if dt == ssp.dt[1]
    ssp.t[1] += dt
  else
    ssp.t[1] = timeend
  end
end

"""
    SSPRK33ShuOsher(f, Q; dt, t0 = 0)

This function returns a [`StrongStabilityPreservingRungeKutta`](@ref) time stepping object
for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses the third-order, 3-stage, strong-stability-preserving, Runge--Kutta scheme
of Shu and Osher (1988)

### References
    @article{shu1988efficient,
      title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},
      author={Shu, Chi-Wang and Osher, Stanley},
      journal={Journal of computational physics},
      volume={77},
      number={2},
      pages={439--471},
      year={1988},
      publisher={Elsevier}
    }
"""
function SSPRK33ShuOsher(F::Union{Function, AbstractSpaceMethod},
                         Q::AT; dt=nothing, t0=0) where {AT <: AbstractArray}
  T = eltype(Q)
  RT = real(T)
  RKA = [ RT(1) RT(0); RT(3//4) RT(1//4); RT(1//3) RT(2//3) ]
  RKB = [ RT(1), RT(1//4), RT(2//3) ]
  RKC = [ RT(0), RT(1), RT(1//2) ]
  StrongStabilityPreservingRungeKutta(F, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

"""
    SSPRK34SpiteriRuuth(f, Q; dt, t0 = 0)

This function returns a [`StrongStabilityPreservingRungeKutta`](@ref) time stepping object
for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses the third-order, 4-stage, strong-stability-preserving, Runge--Kutta scheme
of Spiteri and Ruuth (1988)

### References
    @article{spiteri2002new,
      title={A new class of optimal high-order strong-stability-preserving time discretization methods},
      author={Spiteri, Raymond J and Ruuth, Steven J},
      journal={SIAM Journal on Numerical Analysis},
      volume={40},
      number={2},
      pages={469--491},
      year={2002},
      publisher={SIAM}
    }
"""
function SSPRK34SpiteriRuuth(F::Union{Function, AbstractSpaceMethod},
                             Q::AT; dt=nothing, t0=0) where {AT <: AbstractArray}
  T = eltype(Q)
  RT = real(T)
  RKA = [ RT(1) RT(0); RT(0) RT(1); RT(2//3) RT(1//3); RT(0) RT(1) ]
  RKB = [ RT(1//2); RT(1//2); RT(1//6); RT(1//2) ]
  RKC = [ RT(0); RT(1//2); RT(1); RT(1//2) ]
  StrongStabilityPreservingRungeKutta(F, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

end
