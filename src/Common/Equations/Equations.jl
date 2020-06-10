"""
    Equations

Module defining critical types for formulating, manupulating,
and labeling/annotating balance laws.
"""
module Equations

abstract type AbstractTerm end


"""
Sample equation:

∂ₜ q = S(q) - ∇⋅(F_1(q)) - ∇⋅(F_2(q, σ)),
   σ = Σ(∇q)

q - state (ρ, ρu, ρe)
F_1 - First order (advective) flux of q
F_2 - Second order (diffusive) flux of q
S - source
"""
# Field Signature
struct Signature{𝒮, 𝒯, 𝒰, 𝒱} <: AbstractFruitSignature
    time_scale::𝒮
    domain_space::𝒯
    range_space::𝒰
    model::𝒱
end


# What we want:
"""
dt(Q)
"""
function dt(Q, ...)
    return Tendency(Q, ...)
end

"""
∂ₜ Q
"""
struct Tendency{L} <: AbstractTerm
    label::L
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

end