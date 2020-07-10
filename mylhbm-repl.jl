export myLHBModel

import ClimateMachine.Ocean.HydrostaticBoussinesq: viscosity_tensor
import ClimateMachine.Ocean.HydrostaticBoussinesq: diffusivity_tensor

struct myLHBModel{M} <: BalanceLaw
    ocean::M
    function myLHBModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

vars_state_conservative(lm::myLHBModel, FT) =
    vars_state_conservative(lm.ocean, FT)
vars_state_gradient(lm::myLHBModel, FT) = vars_state_gradient(lm.ocean, FT)
vars_state_gradient_flux(lm::myLHBModel, FT) =
    vars_state_gradient_flux(lm.ocean, FT)
vars_state_auxiliary(lm::myLHBModel, FT) = vars_state_auxiliary(lm.ocean, FT)
vars_integrals(lm::myLHBModel, FT) = @vars()

"""
    No integration, hyperbolic flux, or source terms
"""
@inline integrate_aux!(::myLHBModel, _...) = nothing
@inline flux_first_order!(::myLHBModel, _...) = nothing
@inline source!(::myLHBModel, _...) = nothing

"""
    No need to init, initialize by full model
"""
# init_state_auxiliary!(lm::myLHBModel, A::Vars, geom::LocalGeometry) = nothing
init_state_auxiliary!(lm::myLHBModel, A::Vars, _...) = nothing
init_state_conservative!(lm::myLHBModel, Q::Vars, A::Vars, coords, t) = nothing

"""
    compute_gradient_argument!(::myLHBModel)

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
    m::myLHBModel,
    G::Vars,
    Q::Vars,
    A,
    t,
)
    G.∇u = Q.u
    G.∇θ = Q.θ

    return nothing
end

"""
    compute_gradient_flux!(::myLHBModel)

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
    lm::myLHBModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    ν = viscosity_tensor(lm.ocean)
    D.ν∇u = -ν * G.∇u

    κ = diffusivity_tensor(lm.ocean, G.∇θ[3])
    D.κ∇θ = -κ * G.∇θ

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
@inline function flux_second_order!(
    lm::myLHBModel,
    F::Grad,
    Q::Vars,
    D::Vars,
    HD::Vars,
    A::Vars,
    t::Real,
)
    F.u += D.ν∇u
    F.θ += D.κ∇θ

    return nothing
end

"""
    wavespeed(::LinaerHBModel)

calculates the wavespeed for rusanov flux
"""
function wavespeed(lm::myLHBModel, n⁻, _...)
    C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
    return C
end
