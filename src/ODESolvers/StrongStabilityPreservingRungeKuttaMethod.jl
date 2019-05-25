module StrongStabilityPreservingRungeKuttaMethod
export StrongStabilityPreservingRungeKutta, updatedt!
export StrongStabilityPreservingRungeKutta33
export StrongStabilityPreservingRungeKutta34

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays
    using .CuArrays.CUDAnative
    using .CuArrays.CUDAnative.CUDAdrv

    include("StrongStabilityPreservingRungeKuttaMethod_cuda.jl")
end

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

"""
    StrongStabilityPreservingRungeKutta(f, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

        Q̇ = f(Q)

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses either 2nd or 3rd order Strong-Stability-Preserving (SS) time-integrators.

### References
S. Gottlieb and C.-W. Shu,Total variation diminishing Runge-Kutta schemes, Math .Comp .,67 (1998), pp .73–85
"""

struct StrongStabilityPreservingRungeKutta{T, AT, Nstages} <: ODEs.AbstractODESolver
"time step"
dt::Array{T,1}
"time"
t::Array{T,1}
"rhs function"
rhs!::Function
"Storage for RHS during the StrongStabilityPreservingRungeKutta update"
Rstages::NTuple{Nstages,AT}
"Storage for RHS during the StrongStabilityPreservingRungeKutta update"
Qstages::NTuple{Nstages,AT}
"RK coefficient vector A (rhs scaling)"
RKA::Array{T,2}
"RK coefficient vector B (rhs add in scaling)"
RKB::Array{T,1}
"RK coefficient vector C (time scaling)"
RKC::Array{T,1}

function StrongStabilityPreservingRungeKutta(rhs!::Function, RKA, RKB, RKC, Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    dt = [T(dt)]
    t0 = [T(t0)]
    new{T, AT, length(RKB)}(dt, t0, rhs!, ntuple(i->similar(Q), length(RKB)), ntuple(i->similar(Q), length(RKB)), RKA, RKB, RKC)
end
end

function StrongStabilityPreservingRungeKutta(spacedisc::AbstractSpaceMethod, RKA, RKB, RKC,Q; dt=nothing, t0=0)
    rhs! = (x...) -> SpaceMethods.odefun!(spacedisc, x...)
    StrongStabilityPreservingRungeKutta(rhs!, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

function StrongStabilityPreservingRungeKutta33(spacedisc, Q; dt=nothing,t0=0)
    T=eltype(Q)
    RKA = [ T(1) T(0); T(3//4) T(1//4); T(1//3) T(2//3) ]
    RKB = [T(1), T(1//4), T(2//3)]
    RKC = [ T(0), T(1//4), T(2//3) ]
    StrongStabilityPreservingRungeKutta(spacedisc, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

function StrongStabilityPreservingRungeKutta34(spacedisc, Q; dt=nothing,t0=0)
    T=eltype(Q)
    RKA = [ T(1) T(0); T(0) T(1); T(2//3) T(1//3); T(0) T(1) ]
    RKB = [ T(1//2); T(1//2); T(1//6); T(1//2) ]
    RKC = [ T(0); T(1//4); T(2//3); T(3//3) ]
    StrongStabilityPreservingRungeKutta(spacedisc, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end


"""
    updatedt!(ssp::StrongStabilityPreservingRungeKutta, dt)

Change the time step size to `dt` for `ssp.
"""
updatedt!(ssp::StrongStabilityPreservingRungeKutta, dt) = ssp.dt[1] = dt

function ODEs.dostep!(Q, ssp::StrongStabilityPreservingRungeKutta, timeend, adjustfinalstep)
    time, dt = ssp.t[1], ssp.dt[1]
    if adjustfinalstep && time + dt > timeend
        dt = timeend - time
        @assert dt > 0
    end
    RKA, RKB, RKC = ssp.RKA, ssp.RKB, ssp.RKC
    rhs! = ssp.rhs!
    Rstages, Qstages = ssp.Rstages, ssp.Qstages
    for s = 1:length(RKB)
        Qstages[s] .= Q
        Rstages[s].Q .= 0
        rhs!(Rstages[s], Qstages[s], time)
        update!(Val(size(Q,2)), Val(size(Q,1)), Rstages[s].Q, Qstages[1].Q, Q.Q, Q.realelems, RKA[s,1], RKA[s,2], RKB[s], dt)
        time += RKC[s] * dt
    end
    if dt == ssp.dt[1]
        ssp.t[1] += dt
    else
        ssp.t[1] = timeend
    end

end

# {{{ Update solution (for all dimensions)
function update!(::Val{nstates}, ::Val{Np}, rhs::Array{T,3}, Q0, Q, elems, rka1, rka2, rkb, dt) where {nstates, Np, T}
    @inbounds for e = elems, s = 1:nstates, i = 1:Np
        Q[i, s, e] = rka1*Q0[i, s, e] + rka2*Q[i, s, e] + dt*rkb*rhs[i, s, e]
    end
end
# }}}

end
