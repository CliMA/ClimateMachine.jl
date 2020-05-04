export LinearHBModel

# Linear model for 1D IMEX
"""
    LinearHBModel <: BalanceLaw

A `BalanceLaw` for modeling vertical diffusion implicitly.

write out the equations here

# Usage

    model = HydrostaticBoussinesqModel(problem)
    linear = LinearHBModel(model)

"""
struct LinearHBModel{M} <: BalanceLaw
    ocean::M
    function LinearHBModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

"""
    Copy over state, aux, and diff variables from HBModel
"""
vars_state_conservative(lm::LinearHBModel, FT) =
    vars_state_conservative(lm.ocean, FT)
vars_state_gradient(lm::LinearHBModel, FT) = vars_state_gradient(lm.ocean, FT)
vars_state_gradient_flux(lm::LinearHBModel, FT) =
    vars_state_gradient_flux(lm.ocean, FT)
vars_state_auxiliary(lm::LinearHBModel, FT) = vars_state_auxiliary(lm.ocean, FT)
vars_integrals(lm::LinearHBModel, FT) = @vars()

"""
    No integration, hyperbolic flux, or source terms
"""
@inline integrate_aux!(::LinearHBModel, _...) = nothing
@inline flux_first_order!(::LinearHBModel, _...) = nothing
@inline source!(::LinearHBModel, _...) = nothing

"""
    No need to init, initialize by full model
"""
init_state_auxiliary!(lm::LinearHBModel, A::Vars, geom::LocalGeometry) = nothing
init_state_conservative!(lm::LinearHBModel, Q::Vars, A::Vars, coords, t) =
    nothing

"""
    compute_gradient_argument!(::LinearHBModel)

copy u and θ to var_gradient
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `G`: array of gradient variables
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function compute_gradient_argument!(
    m::LinearHBModel,
    states::NamedTuple,
    t,
)
    u = states.conservative.u
    θ = states.conservative.θ
    ∇u = states.arguments.∇u
    ∇θ = states.arguments.∇θ

    ∇u = u
    ∇θ = θ

    return nothing
end

"""
    compute_gradient_flux!(::LinearHBModel)

copy ν∇u and κ∇θ to var_diffusive
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `D`: array of diffusive variables
- `G`: array of gradient variables
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function compute_gradient_flux!(
    lm::LinearHBModel,
    states::NamedTuple,
    t,
)
    ∇u = states.gradient.∇u
    ∇θ = states.gradient.∇θ
    ν∇u = states.gradient_flux.ν∇u
    κ∇θ = states.gradient_flux.κ∇θ
    ν = viscosity_tensor(lm.ocean)
    κ = diffusivity_tensor(lm.ocean, ∇θ[3])

    ν∇u = ν * ∇u
    κ∇θ = κ * ∇θ

    return nothing
end

"""
    flux_second_order!(::HBModel)

calculates the parabolic flux contribution to state variables
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `F`: array of fluxes for each state variable
- `Q`: array of state variables
- `D`: array of diff variables
- `A`: array of aux variables
- `t`: time, not used

# computations
∂ᵗu = -∇⋅(ν∇u)
∂ᵗθ = -∇⋅(κ∇θ)
"""
@inline function flux_second_order!(lm::LinearHBModel, states::NamedTuple, t)
    Fᵘ = states.flux.u
    Fᶿ = states.flux.θ
    ν∇u = states.gradient_flux.ν∇u
    κ∇θ = states.gradient_flux.κ∇θ

    Fᵘ -= ν∇u
    Fᶿ -= κ∇θ

    return nothing
end

"""
    wavespeed(::LinaerHBModel)

calculates the wavespeed for rusanov flux
"""
function wavespeed(lm::LinearHBModel, n⁻, _...)
    C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
    return C
end
