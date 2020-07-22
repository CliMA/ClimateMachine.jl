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
vars_state(lm::LinearHBModel, ::Prognostic, FT) =
    vars_state(lm.ocean, Prognostic(), FT)
vars_state(lm::LinearHBModel, st::Gradient, FT) = vars_state(lm.ocean, st, FT)
vars_state(lm::LinearHBModel, ::GradientFlux, FT) =
    vars_state(lm.ocean, GradientFlux(), FT)
vars_state(lm::LinearHBModel, st::Auxiliary, FT) = vars_state(lm.ocean, st, FT)
vars_state(lm::LinearHBModel, ::UpwardIntegrals, FT) = @vars()

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
init_state_prognostic!(lm::LinearHBModel, Q::Vars, A::Vars, coords, t) = nothing

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
    lm::LinearHBModel,
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
function wavespeed(lm::LinearHBModel, n⁻, _...)
    C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
    return C
end
