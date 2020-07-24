export LowStorageRungeKutta3N
export LS3NRK44Classic, LS3NRK33Heuns

"""
    LowStorageRungeKutta3N(f, RKA, RKB, RKC, RKW, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a low-storage Runge--Kutta scheme using 3N storage
based on the provided `RKA`, `RKB` and `RKC` coefficient arrays.
 `RKC` (vector of length the number of stages `ns`) set nodal points position;
 `RKA` and `RKB` (size: ns x 2) set weight for tendency and stage-state;
 `RKW` (unused) provides RK weight (last row in Butcher's tableau).

The 3-N storage formulation from Fyfe (1966) is applicable to any 4-stage,
fourth-order RK scheme. It is implemented here as:

```math
\\hspace{-20mm} for ~~ j ~~ in ~ [1:ns]: \\hspace{10mm}
  t_j = t^n + \\Delta t ~ rkC_j
```
```math
  dQ_j = dQ^*_j + f(Q_j,t_j)
```
```math
  Q_{j+1} = Q_{j} + \\Delta t \\{ rkB_{j,1} ~ dQ_j + rkB_{j,2} ~ dR_j \\}
```
```math
  dR_{j+1} = dR_j + rkA_{j+1,2} ~ dQ_j
```
```math
  dQ^*_{j+1} = rkA_{j+1,1} ~ dQ_j
```

The available concrete implementations are:

  - [`LS3NRK44Classic`](@ref)
  - [`LS3NRK33Heuns`](@ref)

### References

    @article{Fyfe1966,
       title = {Economical Evaluation of Runge-Kutta Formulae},
       author = {Fyfe, David J.},
       journal = {Mathematics of Computation},
       volume = {20},
       pages = {392--398},
       year = {1966}
    }
"""
mutable struct LowStorageRungeKutta3N{T, RT, AT, Nstages} <: AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    rhs!
    "Storage for RHS during the `LowStorageRungeKutta3N` update"
    dQ::AT
    "Secondary Storage for RHS during the `LowStorageRungeKutta3N` update"
    dR::AT
    "low storage RK coefficient array A (rhs scaling)"
    RKA::Array{RT, 2}
    "low storage RK coefficient array B (rhs add in scaling)"
    RKB::Array{RT, 2}
    "low storage RK coefficient vector C (time scaling)"
    RKC::Array{RT, 1}
    "RK weight coefficient vector W (last row in Butcher's tableau)"
    RKW::Array{RT, 1}

    function LowStorageRungeKutta3N(
        rhs!,
        RKA,
        RKB,
        RKC,
        RKW,
        Q::AT;
        dt = 0,
        t0 = 0,
    ) where {AT <: AbstractArray}

        T = eltype(Q)
        RT = real(T)

        dQ = similar(Q)
        dR = similar(Q)
        fill!(dQ, 0)
        fill!(dR, 0)

        new{T, RT, AT, length(RKC)}(
            RT(dt),
            RT(t0),
            rhs!,
            dQ,
            dR,
            RKA,
            RKB,
            RKC,
            RKW,
        )
    end
end

"""
    dostep!(Q, lsrk3n::LowStorageRungeKutta3N, p, time::Real,
            [slow_δ, slow_rv_dQ, slow_scaling])

Use the 3N low storage Runge--Kutta method `lsrk3n` to step `Q` forward in time
from the current time `time` to final time `time + getdt(lsrk3n)`.

If the optional parameter `slow_δ !== nothing` then `slow_rv_dQ * slow_δ` is
added as an additional ODE right-hand side source. If the optional parameter
`slow_scaling !== nothing` then after the final stage update the scaling
`slow_rv_dQ *= slow_scaling` is performed.
"""
function dostep!(
    Q,
    lsrk3n::LowStorageRungeKutta3N,
    p,
    time,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
)
    dt = lsrk3n.dt

    RKA, RKB, RKC = lsrk3n.RKA, lsrk3n.RKB, lsrk3n.RKC
    rhs!, dQ, dR = lsrk3n.rhs!, lsrk3n.dQ, lsrk3n.dR

    rv_Q = realview(Q)
    rv_dQ = realview(dQ)
    rv_dR = realview(dR)
    groupsize = 256

    rv_dR .= -0
    for s in 1:length(RKC)
        rhs!(dQ, Q, p, time + RKC[s] * dt, increment = true)

        slow_scaling = nothing
        if s == length(RKC)
            slow_scaling = in_slow_scaling
        end
        # update solution and scale RHS
        event = Event(array_device(Q))
        event = update!(array_device(Q), groupsize)(
            rv_dQ,
            rv_dR,
            rv_Q,
            RKA[s % length(RKC) + 1, 1],
            RKA[s % length(RKC) + 1, 2],
            RKB[s, 1],
            RKB[s, 2],
            dt,
            slow_δ,
            slow_rv_dQ,
            slow_scaling;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Q), event)
    end
end

@kernel function update!(
    dQ,
    dR,
    Q,
    rka1,
    rka2,
    rkb1,
    rkb2,
    dt,
    slow_δ,
    slow_dQ,
    slow_scaling,
)
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            dQ[i] += slow_δ * slow_dQ[i]
        end
        Q[i] += rkb1 * dt * dQ[i] + rkb2 * dt * dR[i]
        dR[i] += rka2 * dQ[i]
        dQ[i] *= rka1
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end
    end
end

"""
    LS3NRK44Classic(f, Q; dt, t0 = 0)

This function returns a [`LowStorageRungeKutta3N`](@ref) time stepping object
for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses the classic 4-stage, fourth-order Runge--Kutta scheme
in the low-storage implementation of Blum (1962)

### References
    @article {Blum1962,
       title = {A Modification of the Runge-Kutta Fourth-Order Method}
       author = {Blum, E. K.},
       journal = {Mathematics of Computation},
       volume = {16},
       pages = {176-187},
       year = {1962}
    }
"""
function LS3NRK44Classic(F, Q::AT; dt = 0, t0 = 0) where {AT <: AbstractArray}
    T = eltype(Q)
    RT = real(T)

    RKA = [
        RT(0) RT(0)
        RT(0) RT(1)
        RT(-1 // 2) RT(0)
        RT(2) RT(-6)
    ]

    RKB = [
        RT(1 // 2) RT(0)
        RT(1 // 2) RT(-1 // 2)
        RT(1) RT(0)
        RT(1 // 6) RT(1 // 6)
    ]

    RKC = [RT(0), RT(1 // 2), RT(1 // 2), RT(1)]

    RKW = [RT(1 // 6), RT(1 // 3), RT(1 // 3), RT(1 // 6)]

    LowStorageRungeKutta3N(F, RKA, RKB, RKC, RKW, Q; dt = dt, t0 = t0)
end

"""
    LS3NRK33Heuns(f, Q; dt, t0 = 0)

This function returns a [`LowStorageRungeKutta3N`](@ref) time stepping object
for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This method uses the 3-stage, third-order Heun's Runge--Kutta scheme.

### References
    @article {Heun1900,
       title = {Neue Methoden zur approximativen Integration der
       Differentialgleichungen einer unabh\"{a}ngigen Ver\"{a}nderlichen}
       author = {Heun, Karl},
       journal = {Z. Math. Phys},
       volume = {45},
       pages = {23--38},
       year = {1900}
    }
"""
function LS3NRK33Heuns(
    F,
    Q::AT;
    dt = nothing,
    t0 = 0,
) where {AT <: AbstractArray}
    T = eltype(Q)
    RT = real(T)

    RKA = [
        RT(0) RT(0)
        RT(0) RT(1)
        RT(-1) RT(1 // 3)
    ]

    RKB = [
        RT(1 // 3) RT(0)
        RT(2 // 3) RT(-1 // 3)
        RT(3 // 4) RT(1 // 4)
    ]

    RKC = [RT(0), RT(1 // 3), RT(2 // 3)]

    RKW = [RT(1 // 4), RT(0), RT(3 // 4)]

    LowStorageRungeKutta3N(F, RKA, RKB, RKC, RKW, Q; dt = dt, t0 = t0)
end
