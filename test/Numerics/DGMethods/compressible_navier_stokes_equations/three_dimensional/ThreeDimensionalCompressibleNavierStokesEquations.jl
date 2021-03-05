module ThreeDimensionalCompressibleNavierStokes

export ThreeDimensionalCompressibleNavierStokesEquations

using Test
using StaticArrays
using LinearAlgebra

using ClimateMachine.Ocean
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Geometry
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws
using ClimateMachine.Ocean: coriolis_parameter
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays: MPIStateArray

import ClimateMachine.BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_conditions,
    boundary_state!
import ClimateMachine.Ocean:
    ocean_init_state!,
    ocean_init_aux!,
    ocean_boundary_state!,
    _ocean_boundary_state!
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type TurbulenceClosure end
struct ConstantViscosity{T} <: TurbulenceClosure
    μ::T
    ν::T
    κ::T
    function ConstantViscosity{T}(;
        μ = T(1e-6),   # m²/s
        ν = T(1e-6),   # m²/s
        κ = T(1e-6),   # m²/s
    ) where {T <: AbstractFloat}
        return new{T}(μ, ν, κ)
    end
end

abstract type CoriolisForce end
struct fPlaneCoriolis{T} <: CoriolisForce
    fₒ::T
    β::T
    function fPlaneCoriolis{T}(;
        fₒ = T(1e-4), # Hz
        β = T(1e-11), # Hz/m
    ) where {T <: AbstractFloat}
        return new{T}(fₒ, β)
    end
end

abstract type Forcing end
struct Buoyancy{T} <: Forcing
    α::T # 1/K
    g::T # m/s²
    function Buoyancy{T}(; α = T(2e-4), g = T(10)) where {T <: AbstractFloat}
        return new{T}(α, g)
    end
end

"""
    ThreeDimensionalCompressibleNavierStokesEquations <: BalanceLaw
A `BalanceLaw` for shallow water modeling.
write out the equations here
# Usage
    ThreeDimensionalCompressibleNavierStokesEquations()
"""
struct ThreeDimensionalCompressibleNavierStokesEquations{
    D,
    A,
    T,
    C,
    F,
    BC,
    FT,
} <: BalanceLaw
    domain::D
    advection::A
    turbulence::T
    coriolis::C
    forcing::F
    boundary_conditions::BC
    cₛ::FT
    ρₒ::FT
    function ThreeDimensionalCompressibleNavierStokesEquations{FT}(
        domain::D,
        advection::A,
        turbulence::T,
        coriolis::C,
        forcing::F,
        boundary_conditions::BC;
        cₛ = FT(sqrt(10)),  #m/s
        ρₒ = FT(1),  #kg/m³
    ) where {FT <: AbstractFloat, D, A, T, C, F, BC}
        return new{D, A, T, C, F, BC, FT}(
            domain,
            advection,
            turbulence,
            coriolis,
            forcing,
            boundary_conditions,
            cₛ,
            ρₒ,
        )
    end
end

CNSE3D = ThreeDimensionalCompressibleNavierStokesEquations

function vars_state(m::CNSE3D, ::Prognostic, T)
    @vars begin
        ρ::T
        ρu::SVector{3, T}
        ρθ::T
    end
end

function init_state_prognostic!(m::CNSE3D, state::Vars, aux::Vars, localgeo, t)
    ocean_init_state!(m, state, aux, localgeo, t)
end

function vars_state(m::CNSE3D, ::Auxiliary, T)
    @vars begin
        x::T
        y::T
        z::T
    end
end

function init_state_auxiliary!(
    model::CNSE3D,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_state_auxiliary!(
        model,
        (model, aux, tmp, geom) -> ocean_init_aux!(model, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )
end

function vars_state(m::CNSE3D, ::Gradient, T)
    @vars begin
        ∇ρ::T
        ∇u::SVector{3, T}
        ∇θ::T
    end
end

function compute_gradient_argument!(
    model::CNSE3D,
    grad::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_argument!(model.turbulence, grad, state, aux, t)
end

@inline function compute_gradient_argument!(
    ::ConstantViscosity,
    grad::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    u = ρu / ρ
    θ = ρθ / ρ

    grad.∇ρ = ρ
    grad.∇u = u
    grad.∇θ = θ

    return nothing
end

function vars_state(m::CNSE3D, ::GradientFlux, T)
    @vars begin
        μ∇ρ::SVector{3, T}
        ν∇u::SMatrix{3, 3, T, 9}
        κ∇θ::SVector{3, T}
    end
end

function compute_gradient_flux!(
    model::CNSE3D,
    gradflux::Vars,
    grad::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_flux!(
        model,
        model.turbulence,
        gradflux,
        grad,
        state,
        aux,
        t,
    )
end

@inline function compute_gradient_flux!(
    ::CNSE3D,
    turb::ConstantViscosity,
    gradflux::Vars,
    grad::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    μ = turb.μ * I
    ν = turb.ν * I
    κ = turb.κ * I

    gradflux.μ∇ρ = -μ * grad.∇ρ
    gradflux.ν∇u = -ν * grad.∇u
    gradflux.κ∇θ = -κ * grad.∇θ

    return nothing
end

@inline function flux_first_order!(
    model::CNSE3D,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    cₛ = model.cₛ
    ρₒ = model.ρₒ

    flux.ρ += ρu
    flux.ρu += (cₛ * ρ)^2 / (2 * ρₒ) * I

    advective_flux!(model, model.advection, flux, state, aux, t)

    return nothing
end

advective_flux!(::CNSE3D, ::Nothing, _...) = nothing

@inline function advective_flux!(
    ::CNSE3D,
    ::NonLinearAdvectionTerm,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    flux.ρu += ρu ⊗ ρu / ρ
    flux.ρθ += ρu * ρθ / ρ

    return nothing
end

function flux_second_order!(
    model::CNSE3D,
    flux::Grad,
    state::Vars,
    gradflux::Vars,
    ::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(model, model.turbulence, flux, state, gradflux, aux, t)
end

@inline function flux_second_order!(
    ::CNSE3D,
    ::ConstantViscosity,
    flux::Grad,
    state::Vars,
    gradflux::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρ += gradflux.μ∇ρ
    flux.ρu += gradflux.ν∇u
    flux.ρθ += gradflux.κ∇θ

    return nothing
end

@inline function source!(
    model::CNSE3D,
    source::Vars,
    state::Vars,
    gradflux::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    coriolis_force!(model, model.coriolis, source, state, aux, t)
    forcing_term!(model, model.forcing, source, state, aux, t)

    return nothing
end

coriolis_force!(::CNSE3D, ::Nothing, _...) = nothing

@inline function coriolis_force!(
    model::CNSE3D,
    coriolis::fPlaneCoriolis,
    source,
    state,
    aux,
    t,
)
    # f × u
    f = [-0, -0, coriolis_parameter(model, coriolis, aux.coords)]
    ρu = state.ρu

    source.ρu -= f × ρu

    return nothing
end

forcing_term!(::CNSE3D, ::Nothing, _...) = nothing

@inline function forcing_term!(
    model::CNSE3D,
    buoy::Buoyancy,
    source,
    state,
    aux,
    t,
)
    α = buoy.α
    g = buoy.g
    ρθ = state.ρθ

    B = α * g * ρθ

    # only in a box, need to generalize for sphere
    source.ρu += @SVector [-0, -0, B]
end

@inline wavespeed(m::CNSE3D, _...) = m.cₛ

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_first_order!(
    ::RoeNumericalFlux,
    model::CNSE3D,
    fluxᵀn::Vars{S},
    n⁻::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
) where {S, A}
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        model,
        fluxᵀn,
        n⁻,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )

    FT = eltype(fluxᵀn)

    # constants and normal vectors
    cₛ = model.cₛ
    ρₒ = model.ρₒ

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n⁻

    # in general thermodynamics
    p⁻ = (cₛ * ρ⁻)^2 / (2 * ρₒ)
    c⁻ = cₛ * sqrt(ρ⁻ / ρₒ)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n⁻

    # in general thermodynamics
    p⁺ = (cₛ * ρ⁺)^2 / (2 * ρₒ)
    c⁺ = cₛ * sqrt(ρ⁺ / ρₒ)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n⁻

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n⁻

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!
    fluxᵀn.ρ -= (w1 + w2 + w3) * 0.5
    fluxᵀn.ρu -=
        (
            w1 * (u - c * n⁻) +
            w2 * (u + c * n⁻) +
            w3 * u +
            w4 * (Δu - Δuₙ * n⁻)
        ) * 0.5
    fluxᵀn.ρθ -= ((w1 + w2) * θ + w5) * 0.5

    return nothing
end

## Boundary Conditions

boundary_conditions(model::CNSE3D) = model.boundary_conditions

"""
    boundary_state!(nf, ::CNSE3D, args...)
applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions
"""
@inline function boundary_state!(nf, bc, model::CNSE3D, args...)
    return _ocean_boundary_state!(nf, bc, model, args...)
end

"""
    ocean_boundary_state!(nf, bc::OceanBC, ::CNSE3D)
splits boundary condition application into velocity
"""
@inline function ocean_boundary_state!(nf, bc::OceanBC, m::CNSE3D, args...)
    ocean_boundary_state!(nf, bc.velocity, m, m.turbulence, args...)
    ocean_boundary_state!(nf, bc.temperature, m, args...)

    return nothing
end

include("bc_momentum.jl")
include("bc_tracer.jl")

end
