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
Q = (Momentum, density, total energy, etc.)
"""
abstract type PrognosticQuantity <: Terminal end

"""
pressure/ Exner function, potential temp / temp.
vorticity, PV, etc.
"""
abstract type DiagnosticQuantity <: Terminal end

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

abstract type VerticalIntegralOperator <: Operator end

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
struct Sum <: Operator
    operands
end
Base.(:+)(t::AbstractExpression...) = Sum(t)

"""
Helper function
"""
function ∂ₜ(Q)
    ...
    return Tendency(Q, args...)
end

"""
∂ₜ Q
"""
struct Tendency <: AbstractExpression
    ...
    ...
    function Tendency(Q, args...)
        ...
        return new(Q, args...)
    end
end

struct Source{ST} <: AbstractExpression
    source_type::ST
    ...
    function Source(Q, args...)
        ...
        return new(source_type, args...)
    end
end

"""
Helper functions for creating source terms
"""
function S(q)
    ...
    return Source(q, ...)
end

"""
An abstract type describing a system of PDEs of the form:

∂ₜ Q = Σᵢ Tᵢ(Q),

where ∂ₜ Q is the `Tendency` and Σᵢ Tᵢ(Q) denotes a sum of
terms.
"""
abstract type AbstractPDESystem end

struct BalanceLaw{TT <: Tendency, ET <: AbstractExpression} <: AbstractPDESystem
    tendency::TT
    termsum::ET
end
Base.:(==)(a::BalanceLaw, b::BalanceLaw) = isequal((a.tendency, a.tendency), (b.termsum, b.termsum))

"""
Allows us to write:

∂ₜ(Q) === S(q) - ∇⋅(F(q)) - ∇⋅(G(q, ∇q))

in code and immediate construct the `BalanceLaw`.

"""
Base.:(===)(tendency::Tendency, terms::AbstractExpression) = BalanceLaw(tendency, terms)

# Sketch of search functions for extracting specific terms
function get_terms!(bl::BalanceLaw, terms, term_type)
    if term_type == "Tendency"
        return append!(terms, [bl.tendency])
    else
        get_terms!(bl.termsum, terms, term_type)
    return terms
end
function get_terms!(expr::Operator, terms, term_type)
    if term_type == expr.term_label
        append!(terms, [expr])
    end
    for term ∈ expr.operands
        get_terms!(term, terms, term_type)
    end
    return terms
end
# Repeat until reach Terminal nodes
function get_terms!(expr::Terminal, terms, term_type)
    if term_type == expr.term_label
        append!(terms, [expr])
    end
    return terms
end

∂ₜ q === S(q) - ∇⋅(F(q); rate=...) - ∇⋅(G(q, ∇q); rate=...)

function linearization(tendency) end

"""
Sample equation:

∂ₜ q = S(q) - ∇⋅(F(q)) - ∇⋅(G(q, ∇q))                                     (eq:foo)

q - state (ρ, ρu, ρe)
F - flux of q,
G - flux of q which also depends on ∇q
S - source

When we go to DG, (eq:foo) becomes (cell-wise integral):

∫ ϕ ⋅ ∂ₜ q dx = ∫ ϕ ⋅ S(q) dx + ∫ ∇ϕ ⋅ F(q) dx - ∮ ϕ ⋅ H₁(q) ds
                + ∫ ∇ϕ ⋅ G(q) dx - ∮ ϕ ⋅ H₂(q, σ) ds,             ∀ ϕ,    (eq:DG-1)

∫ ϕ ⋅ σ dx    = -∫ ∇ϕ ⋅ g(q) dx + ∮ ϕ ⋅ H₃(g(q)) ds,              ∀ ϕ,    (eq:DG-2)

where g is some simple map (coefficient scaling) and H₃ is the numerical flux
for the auxiliary equation. (eq:DG-2) is introduced as an auxiliary variable
for approximating σ = g(∇q).
"""




# Field Signature
abstract type AbstractSignature end
struct Signature{𝒮, 𝒯, 𝒰, 𝒱} <: AbstractSignature
    time_scale::𝒮
    domain_space::𝒯
    range_space::𝒰
    model::𝒱
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