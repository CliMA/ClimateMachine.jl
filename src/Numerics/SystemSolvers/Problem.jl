export Problem, StandardProblem, FixedPointProblem

"""
    Problem

Abstract type representing an equation to be solved.
"""
abstract type Problem end

"""
    StandardProblem(f!, rhs, Q)

Represents the equation `f(Q) = rhs`, where `f!(fQ, Q, args...)` sets
`fQ = f(Q)` and `rhs` is some array that contains the right-hand side.
"""
struct StandardProblem{F!T, AT1, AT2} <: Problem
    f!::F!T
    Q::AT1
    rhs::AT2
end

"""
    FixedPointProblem(f!, Q)

Represents the equation `f(Q) = Q`, where `f!(fQ, Q, args...)` sets
`fQ = f(Q)`.
"""
struct FixedPointProblem{F!T, AT} <: Problem
    f!::F!T
    Q::AT
end

# TODO: Move this somewhere else.
abstract type AbstractImplicitOperator end

"""
    op! = EulerOperator(f!, ϵ)

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
mutable struct EulerOperator{F, FT} <: AbstractImplicitOperator
    f!::F
    ϵ::FT
end

function (op::EulerOperator)(LQ, Q, args...)
    op.f!(LQ, Q, args..., increment = false)
    @. LQ = Q + op.ϵ * LQ
end