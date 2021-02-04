import ClimateMachine.SystemSolvers: enable_duals, preconditioner_update!,
    defaultbatches

export AbstractOperator, GenericImplicitOperator, FixedPointImplicitOperator

abstract type AbstractOperator end # All instances must have field `f!`.

"""
    op! = GenericImplicitOperator(f!, ϵ)

Construct a linear operator which performs an explicit Euler step ``Q + α
f(Q)``, where `f!` and `op!` both operate inplace, with extra arguments passed
through, i.e.
```
op!(LQ, Q, args...)
```
is equivalent to
```
f!(dQ, Q, args...)
LQ .= Q .+ ϵ .* dQ
```
"""
mutable struct GenericImplicitOperator{F, FT} <: AbstractOperator
    f!::F
    ϵ::FT
end

function (op::GenericImplicitOperator)(LQ, Q, args...)
    op.f!(LQ, Q, args..., increment = false)
    @. LQ = Q - op.ϵ * LQ
end

enable_duals(op::GenericImplicitOperator, n::Int = 1, tag = nothing) =
    GenericImplicitOperator(enable_duals(op.f!, n, tag), op.ϵ)

function preconditioner_update!(
    op,
    eo::GenericImplicitOperator,
    preconditioner::ColumnwiseLUPreconditioner,
    args...,
)
    preconditioner_update!(op, eo.f!, preconditioner, args...)
end

function defaultbatches(Q, op::GenericImplicitOperator, coupledstates)
    return defaultbatches(Q, op.f!, coupledstates)
end

mutable struct FixedPointImplicitOperator{F, FT, AT} <: AbstractOperator
    f!::F
    ϵ::FT
    C::AT
end

function (op::FixedPointImplicitOperator)(LQ, Q, args...)
    op.f!(LQ, Q, args..., increment = false)
    @. LQ = op.C + op.ϵ * LQ
end

    #=
Forward Euler Scheme: dQ/dt(Q^n) = (Q^{n+1} - Q^n)/Δt
Adaptive Timestep Forward Euler Scheme: dQ/dt(Q^n) = (Q^{n+m} - Q^n)/α where m = α/Δt

Backward Euler Scheme: dQ/dt(Q^{n+1}) = (Q^{n+1} - Q^n)/Δt
Adaptive Timestep Backward Euler Scheme: dQ/dt(Q^{n+m}) = (Q^{n+m} - Q^n)/α where m = α/Δt
Newton Rearrange: Q^{n+m} - α * dQ/dt(Q^{n+m}) = Q^n
Picard Rearrange: Q^{n+m} = Q^n + α * dQ/dt(Q^{n+m})

Cranky-Nicolson Scheme: (dQ/dt(Q^n) + dQ/dt(Q^{n+1}))/2 = (Q^{n+1} - Q^n)/Δt
Adaptive Timestep Cranky-Nicolson Scheme: (dQ/dt(Q^n) + dQ/dt(Q^{n+m}))/2 = (Q^{n+m} - Q^n)/α where m = α/Δt
Newton Rearrange: Q^{n+m} - α/2 * dQ/dt(Q^{n+m}) = Q^n - α/2 * dQ/dt(Q^n)
Picard Rearrange: Q^{n+m} = Q^n - α/2 * dQ/dt(Q^n) + α/2 * dQ/dt(Q^{n+m})

For Cranky, we define internal array C which stores Q^n - α/2 * dQ/dt(Q^n). We then have that
Newton Rearrange: Q^{n+m} - α/2 * dQ/dt(Q^{n+m}) = C
Picard Rearrange: Q^{n+m} = C + α/2 * dQ/dt(Q^{n+m})

These structures are consistent between Crank-Nicolson and BackwardEuler (different internals)
=#