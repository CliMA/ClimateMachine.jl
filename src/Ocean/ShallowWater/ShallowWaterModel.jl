module ShallowWater

export SWModel, SWProblem

using StaticArrays
using ..VariableTemplates
using LinearAlgebra: I, dot
using CLIMAParameters.Planet: grav

import ClimateMachine.DGmethods:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    vars_integrals,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    compute_gradient_argument!,
    init_state_auxiliary!,
    init_state_conservative!,
    LocalGeometry,
    compute_gradient_flux!

using ..DGmethods.NumericalFluxes

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type SWProblem end

abstract type TurbulenceClosure end
struct LinearDrag{L} <: TurbulenceClosure
    λ::L
end
struct ConstantViscosity{L} <: TurbulenceClosure
    ν::L
end

abstract type AdvectionTerm end
struct NonLinearAdvection <: AdvectionTerm end

struct SWModel{PS, P, T, A, S} <: BalanceLaw
    param_set::PS
    problem::P
    turbulence::T
    advection::A
    c::S
end

function vars_state_conservative(m::SWModel, T)
    @vars begin
        U::SVector{3, T}
        η::T
    end
end

function vars_state_auxiliary(m::SWModel, T)
    @vars begin
        f::SVector{3, T}
        τ::SVector{3, T}  # value includes τₒ, g, and ρ
    end
end

function vars_state_gradient(m::SWModel, T)
    @vars begin
        U::SVector{3, T}
    end
end

function vars_state_gradient_flux(m::SWModel, T)
    @vars begin
        ν∇U::SMatrix{3, 3, T, 9}
    end
end

@inline function flux_first_order!(
    m::SWModel,
    F::Grad,
    q::Vars,
    α::Vars,
    t::Real,
)
    FT = eltype(q)
    _grav::FT = grav(m.param_set)
    U = q.U
    η = q.η
    H = m.problem.H

    F.η += U
    F.U += _grav * H * η * I

    advective_flux!(m, m.advection, F, q, α, t)

    return nothing
end

advective_flux!(::SWModel, ::Nothing, _...) = nothing

@inline function advective_flux!(
    m::SWModel,
    A::NonLinearAdvection,
    F::Grad,
    q::Vars,
    α::Vars,
    t::Real,
)
    U = q.U
    H = m.problem.H

    F.U += 1 / H * U ⊗ U

    return nothing
end

function compute_gradient_argument!(
    m::SWModel,
    f::Vars,
    q::Vars,
    α::Vars,
    t::Real,
)
    compute_gradient_argument!(m.turbulence, f, q, α, t)
end

compute_gradient_argument!(::LinearDrag, _...) = nothing

@inline function compute_gradient_argument!(
    T::ConstantViscosity,
    f::Vars,
    q::Vars,
    α::Vars,
    t::Real,
)
    f.∇U = q.U

    return nothing
end

function compute_gradient_flux!(
    m::SWModel,
    σ::Vars,
    δ::Grad,
    q::Vars,
    α::Vars,
    t::Real,
)
    compute_gradient_flux!(m.turbulence, σ, δ, q, α, t)
end

compute_gradient_flux!(::LinearDrag, _...) = nothing

@inline function compute_gradient_flux!(
    T::ConstantViscosity,
    σ::Vars,
    δ::Grad,
    q::Vars,
    α::Vars,
    t::Real,
)
    ν = T.ν
    ∇U = δ.∇U

    σ.ν∇U = ν * ∇U

    return nothing
end

function flux_second_order!(
    m::SWModel,
    G::Grad,
    q::Vars,
    σ::Vars,
    ::Vars,
    α::Vars,
    t::Real,
)
    flux_second_order!(m.turbulence, G, q, σ, α, t)
end

flux_second_order!(::LinearDrag, _...) = nothing

@inline function flux_second_order!(
    ::ConstantViscosity,
    G::Grad,
    q::Vars,
    σ::Vars,
    α::Vars,
    t::Real,
)
    G.U -= σ.ν∇U

    return nothing
end

@inline wavespeed(m::SWModel, n⁻, q::Vars, α::Vars, t::Real) = m.c

@inline function source!(
    m::SWModel{P},
    S::Vars,
    q::Vars,
    diffusive::Vars,
    α::Vars,
    t::Real,
    direction,
) where {P}
    τ = α.τ
    f = α.f
    U = q.U
    S.U += τ - f × U

    linear_drag!(m.turbulence, S, q, α, t)

    return nothing
end

linear_drag!(::ConstantViscosity, _...) = nothing

@inline function linear_drag!(T::LinearDrag, S::Vars, q::Vars, α::Vars, t::Real)
    λ = T.λ
    U = q.U

    S.U -= λ * U

    return nothing
end

function shallow_init_aux! end
function init_state_auxiliary!(m::SWModel, aux::Vars, geom::LocalGeometry)
    shallow_init_aux!(m.problem, aux, geom)
end

function shallow_init_state! end
function init_state_conservative!(m::SWModel, state::Vars, aux::Vars, coords, t)
    shallow_init_state!(m.problem, m.turbulence, state, aux, coords, t)
end

function boundary_state!(
    nf,
    m::SWModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    shallow_boundary_state!(
        nf,
        m,
        m.turbulence,
        state⁺,
        aux⁺,
        n⁻,
        state⁻,
        aux⁻,
        t,
    )
end

@inline function shallow_boundary_state!(
    ::RusanovNumericalFlux,
    m::SWModel,
    ::LinearDrag,
    state⁺,
    aux⁺,
    n⁻,
    state⁻,
    aux⁻,
    t,
)
    U⁻ = state⁻.U
    n⁻ = SVector(n⁻)

    state⁺.η = state⁻.η
    state⁺.U = U⁻ - 2 * (n⁻ ⋅ U⁻) * n⁻

    return nothing
end

shallow_boundary_state!(
    ::CentralNumericalFluxGradient,
    m::SWModel,
    ::LinearDrag,
    _...,
) = nothing

shallow_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    m::SWModel,
    ::LinearDrag,
    _...,
) = nothing

function boundary_state!(
    nf,
    m::SWModel,
    q⁺::Vars,
    σ⁺::Vars,
    α⁺::Vars,
    n⁻,
    q⁻::Vars,
    σ⁻::Vars,
    α⁻::Vars,
    bctype,
    t,
    _...,
)
    shallow_boundary_state!(nf, m, m.turbulence, q⁺, σ⁺, α⁺, n⁻, q⁻, σ⁻, α⁻, t)
end

@inline function shallow_boundary_state!(
    ::RusanovNumericalFlux,
    m::SWModel,
    ::ConstantViscosity,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
)
    q⁺.η = q⁻.η
    q⁺.U = -q⁻.U

    return nothing
end

@inline function shallow_boundary_state!(
    ::CentralNumericalFluxGradient,
    m::SWModel,
    ::ConstantViscosity,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
)
    q⁺.U = 0

    return nothing
end

@inline function shallow_boundary_state!(
    ::CentralNumericalFluxSecondOrder,
    m::SWModel,
    ::ConstantViscosity,
    q⁺,
    σ⁺,
    α⁺,
    n⁻,
    q⁻,
    σ⁻,
    α⁻,
    t,
)
    q⁺.U = -q⁻.U
    σ⁺.ν∇U = σ⁻.ν∇U

    return nothing
end

end
