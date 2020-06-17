"""
    Equations

Module defining critical types for formulating, manupulating,
and labeling/annotating balance laws.
"""
module Equations

"""
Base type for all Clima PDE expressions
"""
abstract type AbstractExpression end

"""
An expression that does not depend on any other expression.

Why? Expressions (PDEs) can be represented as a syntax tree
and it will be beneficial for us to explicitly define Terminal
expressions so tree visitors (functions traversing the AST)
know when they reach the end of a branch.
"""
abstract type Terminal <: AbstractExpression end

# Different types of `Terminal` quantities
# PrognosticQuantity like the state is a terminal quantity.
# What other things could be terminal quantities?
"""
Momentum, density, total energy, etc.
"""
abstract type PrognosticQuantity <: Terminal end

"""
Q = (Momentum, density, total energy, etc.)
"""
abstract type MixedPrognosticQuantity <: Terminal end

# What do we do about arbitrary tracers?
# People want to be able to look at individual equations
# in addition to terms. How can we best do this?

"""
An expression obtained after applying an operator to
an existing expression. For example, differentiation.

We can create a class of operators. We might want to distinguish
between different types of operators.
"""
abstract type Operator <: AbstractExpression end

"""
∇⋅(F_1(q))

When we go into DG, we will need to deal with
face AND volume integrals for the DifferentialOperator:

ϕ ∇⋅(F_1(q)) * dx = -∇ϕ F_1 * dx + ϕ H_1(q) * ds
"""
abstract type DifferentialOperator <: Operator end

struct Divergence{T <: AbstractExpression} <: Operator
    operand::T
end

struct Curl{T <: AbstractExpression} <: Operator
    operand::T
end

struct Gradient{T <: AbstractExpression} <: Operator
    operand::T
end

# Define operators
struct Grad end
const ∇ = Grad()
(::Grad)(operand) = Gradient(operand)
(⋅)(::Grad, operand) = Divergence(operand)
(×)(::Grad, operand) = Curl(operand)

# Sum of terms
struct Sum <: AbstractExpression
    operands
end
Base.(:+)(t::AbstractExpression...) = Sum(t)


"""
Sample equation:

∂ₜ q = S(q) - ∇⋅(F_1(q)) - ∇⋅(F_2(q, σ)) + ...,
   σ = Σ(∇q, ...)

q - state (ρ, ρu, ρe)
F_1 - First order (advective) flux of q
F_2 - Second order (diffusive) flux of q
S - source
"""
# Field Signature
abstract type AbstractSignature end
struct Signature{𝒮, 𝒯, 𝒰, 𝒱} <: AbstractSignature
    time_scale::𝒮
    domain_space::𝒯
    range_space::𝒰
    model::𝒱
end

# What we want:
"""
∂t(Q)
"""
function ∂t(Q, ...)
    return Tendency(Q, ...)
end

"""
∂ₜ Q
"""
struct Tendency{L} <: AbstractTerm
    label::L
    ...
    ...
    function Tendency(Q, ...)
        ...
    end
end

function S(Q,...)
    return SourceTerm(...)
end

"""
S(q)

In DG, we only need volume integrals:

ϕS(q)*dx

"""
struct SourceTerm <: AbstractTerm
    label
    evaluation::Function
    ...
    function SourceTerm(...)
        ...
    end
end

struct GravitySource <: SourceTerm
    foo
    bar
end

function GravitySource(...)
    return GravitySource(foo=..., bar=...)
end

"""
∇⋅(F_1(q))

When we go into DG, we will need to deal with
face AND volume integrals:

ϕ ∇⋅(F_1(q)) * dx
= ∇ϕ F_1 * dx - ϕ H_1(q) * ds
"""
struct DifferentialTerm <: AbstractTerm end


# One can imagine specializing the DifferentialTerms
PressureGradient <: DifferentialTerm
AdvectionTerm <: DifferentialTerm
DiffusionTerm <: DifferentialTerm
HyperDiffusionTerm <: DifferentialTerm

"""
TODO: Need to pin down some concrete specification
of the LaplacianTerm in DG.
"""
struct LaplacianTerm <: DifferentialTerm
    diffusivity::Function
    ...
    function LaplacianTerm(diffusivity)
        return LaplacianTerm(...)
    end
end


"""
Σ = (0, 2 * S, ∇e)
S = 1/2 (∇u + ∇u^t)


ϕ Σ * dx = ...

"""

# Think about solving equations with linear/nonlinear algebraic constraints
# (see split-explicit equations by Andre and Brandon)


abstract type PrognosticQuantity <: AbstractField end

q = PrognosticQuantity()

struct Mass <: PrognosticQuantity end
struct Momentum <: PrognosticQuantity end

struct Divergence <: DifferentialTerm
    operand
end
struct Gradient <: DifferentialTerm
    operand
end

# define operators
struct Grad end
const ∇ = Grad()
(::Grad)(operand) = Gradient(operand)
(⋅)(::Grad, operand) = Divergence(operand)

struct TermSum <: AbstractTerm
    operands
end
Base.(:+)(t::AbstractTerm...) = TermSum(t)

linearization(o::AbstractTerm) = o

islinear(::PrognosticQuantity) = true
islinear(d::Divergence) = islinear(d.operand)
islinear(d::Gradient) = islinear(d.operand)
islinear(d::TermSum) = all(islinear, d.operands)

isvertical(::Momentum) = false
isvertical(::VericalProjection) = true

struct Pressure <: DiagnosticQuantity
end

islinear(::Pressure) = false

const ρ = Mass()
const ρu = Momentum()

u = ρu / ρ
p = Pressure()

∂t(ρ) ~ ∇ ⋅ ρu + s(ρ)
S ~ (∇(u) + ∇(u)')/2
τ = -2*ν .* S

ρu_euler = ∇⋅(u ⊗ ρu + p * I)
ρu_diffusive = ∇⋅(ρ * τ)


abstractmodel = ∂t(ρu) ~  ∇⋅(u ⊗ ρu + p * I) + ∇⋅(ρ * τ)

"""
Idea:

Lowering from Continuum to fully discrete (3 stages):

Continuum -T_1-> semi-discrete (temporally) -T_2-> fully discrete (full DG model)

"""


# challenges
# - how to "name" subexpressions
#   - numerical fluxes
#   - boundary conditions
#   - time rates
#   - Computational performance:
#     - communication/computation (fluxes!)

end