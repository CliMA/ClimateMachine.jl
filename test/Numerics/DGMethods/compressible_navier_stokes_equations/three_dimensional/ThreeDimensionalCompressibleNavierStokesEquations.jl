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
    boundary_conditions,
    boundary_state!
import ClimateMachine.DGMethods: DGModel
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!
import ClimateMachine.Orientations: vertical_unit_vector

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
    O,
    A,
    T,
    C,
    G,
    E,
    BC,
    P,
} <: AbstractFluid3D
    initial_value_problem::I
    domain::D
    orientation::O
    advection::A
    turbulence::T
    coriolis::C
    gravity::G
    eos::E
    boundary_conditions::BC
    parameters::P
    function ThreeDimensionalCompressibleNavierStokesEquations{FT}(
        initial_value_problem::I,
        domain::D,
        orientation::O,
        advection::A,
        turbulence::T,
        coriolis::C,
        gravity::G,
        eos::E,
        boundary_conditions::BC,
        parameters::P,
    ) where {FT <: AbstractFloat, I, D, O, A, T, C, G, E, BC, P}
        return new{I, D, O, A, T, C, G, E, BC, P}(
            initial_value_problem,
            domain,
            orientation,
            advection,
            turbulence,
            coriolis,
            gravity,
            eos,
            boundary_conditions,
            parameters,
        )
    end
end

CNSE3D = ThreeDimensionalCompressibleNavierStokesEquations

include("thermodynamics.jl")
include("coriolis.jl")
include("gravity.jl")

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

function vars_state(m::CNSE3D, st::Auxiliary, T)
    @vars begin
        x::T
        y::T
        z::T
        orientation::vars_state(m.orientation, st, T)
    end
end

function init_state_auxiliary!(
    m::CNSE3D,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    # update the geopotential Φ in state_auxiliary.orientation.Φ
    init_state_auxiliary!(
        m,
        (m, aux, tmp, geom) ->
            orientation_nodal_init_aux!(m.orientation, m.domain, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )

    # update ∇Φ in state_auxiliary.orientation.∇Φ
    orientation_gradient(m, m.orientation, state_auxiliary, grid, direction)

    # store coordinates and potentially other stuff
    init_state_auxiliary!(
        m,
        (m, aux, tmp, geom) -> cnse_init_aux!(m, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )

    return nothing
end

function orientation_gradient(
    model::CNSE3D,
    ::Orientation,
    state_auxiliary,
    grid,
    direction,
)
    auxiliary_field_gradient!(
        model,
        state_auxiliary,
        ("orientation.∇Φ",),
        state_auxiliary,
        ("orientation.Φ",),
        grid,
        direction,
    )

    return nothing
end

function orientation_gradient(::CNSE3D, ::NoOrientation, _...)
    return nothing
end

function orientation_nodal_init_aux!(
    ::SphericalOrientation,
    domain::Tuple,
    aux::Vars,
    geom::LocalGeometry,
)
    norm_R = norm(geom.coord)
    @inbounds aux.orientation.Φ = norm_R - domain[1]

    return nothing
end

"""
function orientation_nodal_init_aux!(
    ::SuperSphericalOrientation,
    domain::Tuple,
    aux::Vars,
    geom::LocalGeometry,
)
    norm_R = norm(geom.coord)
    @inbounds aux.orientation.Φ = 1 / norm_R^2 
end
"""

function orientation_nodal_init_aux!(
    ::FlatOrientation,
    domain::Tuple,
    aux::Vars,
    geom::LocalGeometry,
)
    @inbounds aux.orientation.Φ = geom.coord[3]

    return nothing
end

function orientation_nodal_init_aux!(
    ::NoOrientation,
    domain::Tuple,
    aux::Vars,
    geom::LocalGeometry,
)
    return nothing
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
        ∇p::T
    end
end

function compute_gradient_argument!(
    model::CNSE3D,
    grad::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρ = state.ρ
    
    grad.∇p = calc_pressure(model.eos, state)

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
        ∇p::SVector{3, T}
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

    gradflux.∇p = grad.∇p

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
    ρu = state.ρu

    flux.ρ += ρu
    # flux.ρu += calc_pressure(model.eos, state)

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
    source.ρu -= gradflux.∇p

    calc_force!(source, model.coriolis, state, aux, model.orientation, t)
    calc_force!(source, model.gravity, state, aux, model.orientation, t)

    return nothing
end

@inline vertical_unit_vector(::Orientation, aux) = aux.orientation.∇Φ
@inline vertical_unit_vector(::NoOrientation, aux) = @SVector [0, 0, 1]

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

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n⁻

    # in general thermodynamics
    p⁻ = calc_pressure(model.eos, state⁻)
    c⁻ = calc_sound_speed(model.eos, state⁻)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n⁻

    # in general thermodynamics
    p⁺ = calc_pressure(model.eos, state⁺)
    c⁺ = calc_sound_speed(model.eos, state⁺)

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
    
    #=
    max_wavespeed = 10.0 # max(c⁻, c⁺)
    fluxᵀn.ρ -= max_wavespeed * 0.5 * Δρ * 1.0
    fluxᵀn.ρu -=
        (
            max_wavespeed * Δu
        ) * 0.5
    fluxᵀn.ρθ -= max_wavespeed * Δρθ * 1.0
    =#

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
        physics.orientation,
        physics.advection,
        physics.dissipation,
        physics.coriolis,
        physics.gravity,
        physics.eos,
        bcs,
        parameters,
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