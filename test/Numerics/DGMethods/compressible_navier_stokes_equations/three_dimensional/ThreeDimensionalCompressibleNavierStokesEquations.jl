include("../boilerplate.jl")

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
import ClimateMachine.DGMethods: DGModel
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

"""
    ThreeDimensionalCompressibleNavierStokesEquations <: BalanceLaw
A `BalanceLaw` for shallow water modeling.
write out the equations here
# Usage
    ThreeDimensionalCompressibleNavierStokesEquations()
"""
abstract type AbstractFluid3D <: AbstractFluid end
struct Fluid3D <: AbstractFluid3D end

struct ThreeDimensionalCompressibleNavierStokesEquations{
    I,
    D,
    A,
    T,
    C,
    F,
    BC,
    FT,
} <: AbstractFluid3D
    initial_value_problem::I
    domain::D
    advection::A
    turbulence::T
    coriolis::C
    forcing::F
    boundary_conditions::BC
    cₛ::FT
    ρₒ::FT
    function ThreeDimensionalCompressibleNavierStokesEquations{FT}(
        initial_value_problem::I,
        domain::D,
        advection::A,
        turbulence::T,
        coriolis::C,
        forcing::F,
        boundary_conditions::BC;
        cₛ = FT(sqrt(10)),  # m/s
        ρₒ = FT(1),  #kg/m³
    ) where {FT <: AbstractFloat, I, D, A, T, C, F, BC}
        return new{I, D, A, T, C, F, BC, FT}(
            initial_value_problem,
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
    cnse_init_state!(m, state, aux, localgeo, t)
end

# default initial state if IVP == nothing
function cnse_init_state!(model::CNSE3D, state, aux, localgeo, t)

    x = aux.x
    y = aux.y
    z = aux.z

    ρ = model.ρₒ
    state.ρ = ρ
    state.ρu = ρ * @SVector [-0, -0, -0]
    state.ρθ = ρ

    return nothing
end

# user defined initial state
function cnse_init_state!(
    model::CNSE3D{<:InitialValueProblem},
    state,
    aux,
    localgeo,
    t,
)
    x = aux.x
    y = aux.y
    z = aux.z

    params = model.initial_value_problem.params
    ic = model.initial_value_problem.initial_conditions

    state.ρ = ic.ρ(params, x, y, z)
    state.ρu = ic.ρu(params, x, y, z)
    state.ρθ = ic.ρθ(params, x, y, z)

    return nothing
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
        (model, aux, tmp, geom) -> cnse_init_aux!(model, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )
end

function cnse_init_aux!(::CNSE3D, aux, geom)
    @inbounds begin
        aux.x = geom.coord[1]
        aux.y = geom.coord[2]
        aux.z = geom.coord[3]
    end

    return nothing
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

boundary_conditions(model::CNSE3D) = model.boundary_conditions

"""
    boundary_state!(nf, ::CNSE3D, args...)
applies boundary conditions for the hyperbolic fluxes
dispatches to a function in CNSEBoundaryConditions
"""
@inline function boundary_state!(nf, bc, model::CNSE3D, args...)
    return _cnse_boundary_state!(nf, bc, model, args...)
end

"""
    cnse_boundary_state!(nf, bc::FluidBC, ::CNSE3D)
splits boundary condition application into velocity
"""
@inline function cnse_boundary_state!(nf, bc::FluidBC, m::CNSE3D, args...)
    cnse_boundary_state!(nf, bc.momentum, m, m.turbulence, args...)
    cnse_boundary_state!(nf, bc.temperature, m, args...)

    return nothing
end

include("bc_momentum.jl")
include("bc_temperature.jl")

"""
STUFF FOR ANDRE'S WRAPPERS
"""

function get_boundary_conditions(
    model::SpatialModel{BL},
) where {BL <: AbstractFluid3D}
    bcs = model.boundary_conditions

    west_east = (check_bc(bcs, :west), check_bc(bcs, :east))
    south_north = (check_bc(bcs, :south), check_bc(bcs, :north))
    bottom_top = (check_bc(bcs, :bottom), check_bc(bcs, :top))

    return (west_east..., south_north..., bottom_top...)
end

function DGModel(
    model::SpatialModel{BL};
    initial_conditions = nothing,
) where {BL <: AbstractFluid3D}
    params = model.parameters
    physics = model.physics

    Lˣ, Lʸ, Lᶻ = length(model.grid.domain)
    bcs = get_boundary_conditions(model)
    FT = eltype(model.grid.numerical.vgeo)

    if !isnothing(initial_conditions)
        initial_conditions = InitialValueProblem(params, initial_conditions)
    end

    balance_law = CNSE3D{FT}(
        initial_conditions,
        (Lˣ, Lʸ, Lᶻ),
        physics.advection,
        physics.dissipation,
        physics.coriolis,
        physics.buoyancy,
        bcs,
        ρₒ = params.ρₒ,
        cₛ = params.cₛ,
    )

    numerical_flux_first_order = model.numerics.flux # should be a function

    rhs = DGModel(
        balance_law,
        model.grid.numerical,
        numerical_flux_first_order,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    return rhs
end
