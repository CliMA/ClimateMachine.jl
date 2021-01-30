export RLSRK54CarpenterKennedy, RLSRK144NiegemannDiehlBusch
using NLsolve: nlsolve

mutable struct RelaxationLowStorageRungeKutta2N{T, RT, AT, Nstages} <: AbstractODESolver
    "time step"
    dt::RT
    "relaxation time step scaling"
    γ::Vector{RT}
    "time"
    t::RT
    "elapsed time steps"
    steps::Int
    "rhs function"
    rhs!::Any
    "entropy integral function"
    entropy_integral::Any
    "entropy product function"
    entropy_product::Any
    "Storage for cumulative RHS during the LowStorageRungeKutta update"
    dQ::AT
    "Storage for RHS during the LowStorageRungeKutta update"
    k::AT
    "Storage for initial state"
    Q0::AT
    "low storage RK coefficient vector A (rhs scaling)"
    RKA::NTuple{Nstages, RT}
    "low storage RK coefficient vector B (rhs add in scaling)"
    RKB::NTuple{Nstages, RT}
    "full RK coefficient vector b"
    RKb::NTuple{Nstages, RT}
    "low storage RK coefficient vector C (time scaling)"
    RKC::NTuple{Nstages, RT}

    function RelaxationLowStorageRungeKutta2N(
        rhs!,
        entropy_integral,
        entropy_product,
        RKA,
        RKB,
        RKC,
        Q::AT;
        dt = 0,
        t0 = 0,
    ) where {AT <: AbstractArray}

        T = eltype(Q)
        RT = real(T)

        dQ = similar(Q)
        fill!(dQ, 0)
        k = similar(Q)
        
        Q0 = similar(Q)

        # construct standard RK b coefficients
        RKb = zeros(RT, length(RKB))
        RKb[end] = RKB[end]
        for i in length(RKb)-1:-1:1
          RKb[i] = RKA[i+1] * RKb[i+1] + RKB[i]
        end

        γ = [RT(1)]
        new{T, RT, AT, length(RKA)}(
          RT(dt),
          γ,
          RT(t0),
          0,
          rhs!,
          entropy_integral,
          entropy_product,
          dQ,
          k,
          Q0,
          RKA,
          RKB,
          Tuple(RKb),
          RKC
        )
    end
end

function dostep!(
    Q,
    lsrk::RelaxationLowStorageRungeKutta2N,
    p,
    time,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
)
    FT = eltype(Q)

    @assert slow_δ == nothing
    @assert slow_rv_dQ == nothing
    @assert in_slow_scaling == nothing

    dt = lsrk.dt
    γ = lsrk.γ

    RKA, RKB, RKb, RKC = lsrk.RKA, lsrk.RKB, lsrk.RKb, lsrk.RKC
    rhs!, dQ, Q0, k = lsrk.rhs!, lsrk.dQ, lsrk.Q0, lsrk.k
    entropy_integral, entropy_product  = lsrk.entropy_integral, lsrk.entropy_product

    rv_Q = realview(Q)
    rv_dQ = realview(dQ)

    groupsize = 256
    
    Q0 .= Q
    η0 = entropy_integral(rhs!, Q0)
    dη = -zero(FT)

    for s in 1:length(RKA)
        rhs!(k, Q, p, time + RKC[s] * dt, increment = false)
        dQ .+= k

        dη += RKb[s] * dt * entropy_product(rhs!, Q, k)

        slow_scaling = nothing
        if s == length(RKA)
            slow_scaling = in_slow_scaling
        end
        # update solution and scale RHS
        event = Event(array_device(Q))
        event = update!(array_device(Q), groupsize)(
            rv_dQ,
            rv_Q,
            RKA[s % length(RKA) + 1],
            RKB[s],
            dt,
            slow_δ,
            slow_rv_dQ,
            slow_scaling;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Q), event)
    end

    # nonlinear solve
    @. k = Q - Q0
    function r(F, γ)
      @. Q = Q0 + γ[1] * k
      F[1] = entropy_integral(rhs!, Q) - η0 - γ[1] * dη
    end
    function dr(J, γ)
      @. Q = Q0 + γ[1] * k
      J[1] = entropy_product(rhs!, Q, k) - dη
    end
    sol = nlsolve(r, dr, γ; ftol=10eps(FT))
    γ = sol.zero
    Q .= Q0 .+ γ[1] * k
end

"""
    RLSRK54CarpenterKennedy(f, Q; dt, t0 = 0)

This function returns a [`LowStorageRungeKutta2N`](@ref) time stepping object
for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

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
function RLSRK54CarpenterKennedy(
    F,
    entropy_integral,
    entropy_product,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    T = eltype(Q)
    RT = real(T)

    RKA = (
        RT(0),
        RT(-567301805773 // 1357537059087),
        RT(-2404267990393 // 2016746695238),
        RT(-3550918686646 // 2091501179385),
        RT(-1275806237668 // 842570457699),
    )

    RKB = (
        RT(1432997174477 // 9575080441755),
        RT(5161836677717 // 13612068292357),
        RT(1720146321549 // 2090206949498),
        RT(3134564353537 // 4481467310338),
        RT(2277821191437 // 14882151754819),
    )

    RKC = (
        RT(0),
        RT(1432997174477 // 9575080441755),
        RT(2526269341429 // 6820363962896),
        RT(2006345519317 // 3224310063776),
        RT(2802321613138 // 2924317926251),
    )

    RelaxationLowStorageRungeKutta2N(F,
                                     entropy_integral,
                                     entropy_product,
                                     RKA,
                                     RKB,
                                     RKC,
                                     Q; dt = dt, t0 = t0)
end

"""
    LSRK144NiegemannDiehlBusch((f, Q; dt, t0 = 0)

This function returns a [`LowStorageRungeKutta2N`](@ref) time stepping object
for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses the fourth-order, 14-stage, low-storage, Runge--Kutta scheme of
Niegemann, Diehl, and Busch (2012) with optimized stability region

### References

    @article{niegemann2012efficient,
      title={Efficient low-storage Runge--Kutta schemes with optimized stability regions},
      author={Niegemann, Jens and Diehl, Richard and Busch, Kurt},
      journal={Journal of Computational Physics},
      volume={231},
      number={2},
      pages={364--372},
      year={2012},
      publisher={Elsevier}
    }
"""
function RLSRK144NiegemannDiehlBusch(
    F,
    entropy_integral,
    entropy_product,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    T = eltype(Q)
    RT = real(T)

    RKA = (
        RT(0),
        RT(-0.7188012108672410),
        RT(-0.7785331173421570),
        RT(-0.0053282796654044),
        RT(-0.8552979934029281),
        RT(-3.9564138245774565),
        RT(-1.5780575380587385),
        RT(-2.0837094552574054),
        RT(-0.7483334182761610),
        RT(-0.7032861106563359),
        RT(0.0013917096117681),
        RT(-0.0932075369637460),
        RT(-0.9514200470875948),
        RT(-7.1151571693922548),
    )

    RKB = (
        RT(0.0367762454319673),
        RT(0.3136296607553959),
        RT(0.1531848691869027),
        RT(0.0030097086818182),
        RT(0.3326293790646110),
        RT(0.2440251405350864),
        RT(0.3718879239592277),
        RT(0.6204126221582444),
        RT(0.1524043173028741),
        RT(0.0760894927419266),
        RT(0.0077604214040978),
        RT(0.0024647284755382),
        RT(0.0780348340049386),
        RT(5.5059777270269628),
    )

    RKC = (
        RT(0),
        RT(0.0367762454319673),
        RT(0.1249685262725025),
        RT(0.2446177702277698),
        RT(0.2476149531070420),
        RT(0.2969311120382472),
        RT(0.3978149645802642),
        RT(0.5270854589440328),
        RT(0.6981269994175695),
        RT(0.8190890835352128),
        RT(0.8527059887098624),
        RT(0.8604711817462826),
        RT(0.8627060376969976),
        RT(0.8734213127600976),
    )

    RelaxationLowStorageRungeKutta2N(F,
                                     entropy_integral,
                                     entropy_product,
                                     RKA,
                                     RKB,
                                     RKC,
                                     Q;
                                     dt = dt, t0 = t0)
end
