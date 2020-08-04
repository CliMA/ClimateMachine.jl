module ShallowWater

export ShallowWaterModel

using StaticArrays
using LinearAlgebra: dot, Diagonal
using CLIMAParameters.Planet: grav

using ..Ocean
using ...VariableTemplates
using ...Mesh.Geometry
using ...DGMethods
using ...DGMethods.NumericalFluxes
using ...BalanceLaws
using ..Ocean: kinematic_stress, coriolis_parameter

import ...DGMethods.NumericalFluxes: update_penalty!
import ...BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!
import ..Ocean: ocean_init_state!, ocean_init_aux!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type TurbulenceClosure end
struct LinearDrag{L} <: TurbulenceClosure
    λ::L
end
struct ConstantViscosity{L} <: TurbulenceClosure
    ν::L
end

abstract type AdvectionTerm end
struct NonLinearAdvection <: AdvectionTerm end

"""
    ShallowWaterModel <: BalanceLaw

A `BalanceLaw` for shallow water modeling.

write out the equations here

# Usage

    ShallowWaterModel(problem)

"""
struct ShallowWaterModel{C, PS, P, T, A, FT} <: BalanceLaw
    param_set::PS
    problem::P
    coupling::C
    turbulence::T
    advection::A
    c::FT
    fₒ::FT
    β::FT
    function ShallowWaterModel{FT}(
        param_set::PS,
        problem::P,
        turbulence::T,
        advection::A;
        coupling::C = Uncoupled(),
        c = FT(0), # m/s
        fₒ = FT(1e-4), # Hz
        β = FT(1e-11), # Hz / m
    ) where {FT <: AbstractFloat, PS, P, T, A, C}
        return new{C, PS, P, T, A, FT}(
            param_set,
            problem,
            coupling,
            turbulence,
            advection,
            c,
            fₒ,
            β,
        )
    end
end
SWModel = ShallowWaterModel

function vars_state(m::SWModel, ::Prognostic, T)
    @vars begin
        η::T
        U::SVector{2, T}
    end
end


function init_state_prognostic!(m::SWModel, state::Vars, aux::Vars, coords, t)
    ocean_init_state!(m, m.problem, state, aux, coords, t)
end

function vars_state(m::SWModel, ::Auxiliary, T)
    @vars begin
        y::T
        Gᵁ::SVector{2, T} # integral of baroclinic tendency
        Δu::SVector{2, T} # reconciliation Δu = 1/H * (Ū - ∫u)
    end
end


function init_state_auxiliary!(m::SWModel, aux::Vars, geom::LocalGeometry)
    ocean_init_aux!(m, m.problem, aux, geom)
end

function vars_state(m::SWModel, ::Gradient, T)
    @vars begin
        ∇U::SVector{2, T}
    end
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

function vars_state(m::SWModel, ::GradientFlux, T)
    @vars begin
        ν∇U::SMatrix{3, 2, T, 6}
    end
end

function compute_gradient_flux!(
    m::SWModel,
    σ::Vars,
    δ::Grad,
    q::Vars,
    α::Vars,
    t::Real,
)
    compute_gradient_flux!(m, m.turbulence, σ, δ, q, α, t)
end

compute_gradient_flux!(::SWModel, ::LinearDrag, _...) = nothing

@inline function compute_gradient_flux!(
    ::SWModel,
    T::ConstantViscosity,
    σ::Vars,
    δ::Grad,
    q::Vars,
    α::Vars,
    t::Real,
)
    ν = Diagonal(@SVector [T.ν, T.ν, -0])
    ∇U = δ.∇U

    σ.ν∇U = -ν * ∇U

    return nothing
end

@inline function flux_first_order!(
    m::SWModel,
    F::Grad,
    q::Vars,
    α::Vars,
    t::Real,
    direction,
)
    U = @SVector [q.U[1], q.U[2], -0]
    η = q.η
    H = m.problem.H
    Iʰ = @SMatrix [
        1 -0
        -0 1
        -0 -0
    ]

    F.η += U
    F.U += grav(m.param_set) * H * η * Iʰ

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
    V = @SVector [U[1], U[2], -0]

    F.U += 1 / H * V ⊗ U

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
    flux_second_order!(m, m.turbulence, G, q, σ, α, t)
end

flux_second_order!(::SWModel, ::LinearDrag, _...) = nothing

@inline function flux_second_order!(
    ::SWModel,
    ::ConstantViscosity,
    G::Grad,
    q::Vars,
    σ::Vars,
    α::Vars,
    t::Real,
)
    G.U += σ.ν∇U

    return nothing
end

@inline wavespeed(m::SWModel, n⁻, q::Vars, α::Vars, t::Real, direction) = m.c

@inline function source!(
    m::SWModel{P},
    S::Vars,
    q::Vars,
    d::Vars,
    α::Vars,
    t::Real,
    direction,
) where {P}
    # f × u
    U, V = q.U
    f = coriolis_parameter(m, m.problem, α.y)
    S.U -= @SVector [-f * V, f * U]

    forcing_term!(m, m.coupling, S, q, α, t)
    linear_drag!(m.turbulence, S, q, α, t)

    return nothing
end

@inline function forcing_term!(m::SWModel, ::Uncoupled, S, Q, A, t)
    S.U += kinematic_stress(m.problem, A.y)

    return nothing
end

linear_drag!(::ConstantViscosity, _...) = nothing

@inline function linear_drag!(T::LinearDrag, S::Vars, q::Vars, α::Vars, t::Real)
    λ = T.λ
    U = q.U

    S.U -= λ * U

    return nothing
end

function boundary_state!(
    nf,
    m::SWModel,
    q⁺::Vars,
    a⁺::Vars,
    n⁻,
    q⁻::Vars,
    a⁻::Vars,
    bctype,
    t,
    _...,
)
    shallow_boundary_state!(nf, m, m.turbulence, q⁺, a⁺, n⁻, q⁻, a⁻, t)
end

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
    ::NumericalFluxFirstOrder,
    m::SWModel,
    ::LinearDrag,
    q⁺,
    a⁺,
    n⁻,
    q⁻,
    a⁻,
    t,
)
    q⁺.η = q⁻.η

    V⁻ = @SVector [q⁻.U[1], q⁻.U[2], -0]
    V⁺ = V⁻ - 2 * n⁻ ⋅ V⁻ .* SVector(n⁻)
    q⁺.U = @SVector [V⁺[1], V⁺[2]]

    return nothing
end

shallow_boundary_state!(
    ::NumericalFluxGradient,
    m::SWModel,
    ::LinearDrag,
    _...,
) = nothing

shallow_boundary_state!(
    ::NumericalFluxSecondOrder,
    m::SWModel,
    ::LinearDrag,
    _...,
) = nothing

@inline function shallow_boundary_state!(
    ::NumericalFluxFirstOrder,
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
    ::NumericalFluxGradient,
    m::SWModel,
    ::ConstantViscosity,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
)
    FT = eltype(q⁺)
    q⁺.U = @SVector zeros(FT, 3)

    return nothing
end

@inline function shallow_boundary_state!(
    ::NumericalFluxSecondOrder,
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
