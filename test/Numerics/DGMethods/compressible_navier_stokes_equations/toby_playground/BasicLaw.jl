include("boilerplate.jl")

import ClimateMachine.BalanceLaws:
    vars_state,
    init_state_prognostic!,
    init_state_auxiliary!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    BoundaryCondition,
    boundary_conditions,
    boundary_state!
import ClimateMachine.DGMethods: DGModel
import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

abstract type AbstractFluid3D <: AbstractFluid end
struct Fluid3D <: AbstractFluid3D end

struct BasicLaw{I,D,BC,P} <: AbstractFluid3D
    initial_value_problem::I
    domain::D
    boundary_conditions::BC
    parameters::P
    function BasicLaw{FT}(
        initial_value_problem::I,
        domain::D,
        boundary_conditions::BC,
        parameters::P,
    ) where {FT <: AbstractFloat, I, D, BC, P}
        return new{I, D, BC, P}(
            initial_value_problem,
            domain,
            boundary_conditions,
            parameters,
        )
    end
end

# Declaration
vars_state(::BasicLaw, ::Prognostic, FT)     = @vars(ρ::FT, ρu::SVector{3, FT})
vars_state(::BasicLaw, ::Auxiliary, FT)      = @vars(x::FT, y::FT, z::FT)
vars_state(::BasicLaw, ::Gradient, _...)     = @vars()
vars_state(::BasicLaw, ::GradientFlux, _...) = @vars()

struct BasicLawBC <: BoundaryCondition end
boundary_conditions(::BasicLaw,) = (BasicLawBC(),)

# Initialization
function nodal_init_state_auxiliary!(::BasicLaw, aux::Vars, tmp::Vars, geom::LocalGeometry)
    aux.x = geom.coord[1]
    aux.y = geom.coord[2]
    aux.z = geom.coord[3]
end

function init_state_prognostic!(model::BasicLaw, state::Vars, aux::Vars, localgeo, t)
    x = aux.x
    y = aux.y
    z = aux.z

    params = model.initial_value_problem.params
    ic = model.initial_value_problem.initial_conditions

    state.ρ = ic.ρ(params, x, y, z)
    state.ρu = ic.ρu(params, x, y, z)
end

# Equations of motion
function flux_first_order!(model::BasicLaw, flux::Grad, state::Vars, aux::Vars, t::Real, direction)
    cₛ = model.parameters.cₛ 
    
    ρ = state.ρ
    ρu = state.ρu

    flux.ρ += ρu
    flux.ρu += cₛ * ρ * I

    return nothing
end

nodal_update_auxiliary_state!(::BasicLaw, _...) = nothing
compute_gradient_argument!(::BasicLaw, _...)    = nothing 
compute_gradient_flux!(::BasicLaw, _...)        = nothing
flux_second_order!(::BasicLaw, _...)            = nothing
source!(::BasicLaw, _...)                       = nothing

function boundary_state!(
    nf,
    ::BasicLawBC,
    ::BasicLaw,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    ρ⁻  = state⁻.ρ
    ρu⁻ = state⁻.ρu

    state⁺.ρ = ρ⁻
    state⁺.ρu = ρu⁻ - 2 * n⁻ ⋅ ρu⁻ .* SVector(n⁻)
end

@inline wavespeed(model::BasicLaw, _...) = model.parameters.cₛ

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_first_order!(
    ::RoeNumericalFlux,
    model::BasicLaw,
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

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    uₙ⁻ = u⁻' * n⁻

    # in general thermodynamics
    p⁻ = ρ⁻
    c⁻ = model.parameters.cₛ

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    uₙ⁺ = u⁺' * n⁻

    # in general thermodynamics
    p⁺ = ρ⁺
    c⁺ = model.parameters.cₛ

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n⁻

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δuₙ = Δu' * n⁻

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ

    # fluxes!!!
    
    fluxᵀn.ρ -= (w1 + w2 + w3) * 0.5
    fluxᵀn.ρu -=
        (
            w1 * (u - c * n⁻) +
            w2 * (u + c * n⁻) +
            w3 * u +
            w4 * (Δu - Δuₙ * n⁻)
        ) * 0.5

    return nothing
end

function DGModel(model::SpatialModel{BL}; initial_conditions = nothing,) where {BL <: AbstractFluid3D}
    params  = model.parameters
    physics = model.physics

    Lˣ, Lʸ, Lᶻ = length(model.grid.domain)
    bcs = (BasicLawBC, BasicLawBC, BasicLawBC, BasicLawBC, BasicLawBC, BasicLawBC,)
    FT = eltype(model.grid.numerical.vgeo)

    if !isnothing(initial_conditions)
        initial_conditions = InitialValueProblem(params, initial_conditions)
    end

    balance_law = BasicLaw{FT}(
        initial_conditions,
        (Lˣ, Lʸ, Lᶻ),
        bcs,
        parameters,
    )

    numerical_flux_first_order = model.numerics.flux

    rhs = DGModel(
        balance_law,
        model.grid.numerical,
        numerical_flux_first_order,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    return rhs
end