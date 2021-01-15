#### State types

export AbstractStateType,
    Prognostic,
    PrognosticIn,
    PrognosticOut,
    Primitive,
    Auxiliary,
    Gradient,
    GradientFlux,
    GradientLaplacian,
    Hyperdiffusive,
    UpwardIntegrals,
    DownwardIntegrals

"""
    AbstractStateType

Subtypes of this describe the variables used by different parts of a [`BalanceLaw`](@ref):
- [`Prognostic`](@ref)
- [`Primitive`](@ref)
- [`Auxiliary`](@ref)
- [`Gradient`](@ref)
- [`GradientFlux`](@ref)
- [`GradientLaplacian`](@ref)
- [`Hyperdiffusive`](@ref)
- [`UpwardIntegrals`](@ref)
- [`DownwardIntegrals`](@ref)

See also [`vars_state`](@ref).
"""
abstract type AbstractStateType end

"""
    Prognostic <: AbstractStateType

Prognostic variables in the PDE system,
which are specified by the [`BalanceLaw`](@ref), and
solved for by the ODE solver.
"""
struct Prognostic <: AbstractStateType end

struct PrognosticIn <: AbstractStateType end
struct PrognosticOut <: AbstractStateType end




"""
    Primitive <: AbstractStateType

Primitive variables, which are specified
by the [`BalanceLaw`](@ref).
"""
struct Primitive <: AbstractStateType end

"""
    Auxiliary <: AbstractStateType

Auxiliary variables help serve several purposes:

 - Pre-compute and store "expensive" variables,
   for example, quantities computed in vertical
   integrals.
 - Diagnostic exports
"""
struct Auxiliary <: AbstractStateType end

"""
    Gradient <: AbstractStateType

Variables whose gradients must be computed.
"""
struct Gradient <: AbstractStateType end

"""
    GradientFlux <: AbstractStateType

Flux variables, which are functions of gradients.
"""
struct GradientFlux <: AbstractStateType end

"""
    GradientLaplacian <: AbstractStateType

Gradient-Laplacian variables.
"""
struct GradientLaplacian <: AbstractStateType end

"""
    Hyperdiffusive <: AbstractStateType

Hyper-diffusive variables
"""
struct Hyperdiffusive <: AbstractStateType end

"""
    UpwardIntegrals <: AbstractStateType

Variables computed in upward integrals
"""
struct UpwardIntegrals <: AbstractStateType end

"""
    DownwardIntegrals <: AbstractStateType

Variables computed in downward integrals
"""
struct DownwardIntegrals <: AbstractStateType end
