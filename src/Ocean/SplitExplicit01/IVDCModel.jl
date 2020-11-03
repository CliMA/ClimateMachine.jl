# Linear model equations, for split-explicit ocean model implicit vertical diffusion
# convective adjustment step.
#
# In this version the operator is tweked to be the indentity for testing

using ClimateMachine.DGMethods.NumericalFluxes:
    NumericalFluxFirstOrder, NumericalFluxSecondOrder, NumericalFluxGradient

"""
 IVDCModel{M} <: BalanceLaw

 This code defines DG `BalanceLaw` terms for an operator, L, that is evaluated from iterative
 implicit solver to solve an equation of the form

 (L + 1/Δt) ϕ^{n+1} = ϕ^{n}/Δt

  where L is a vertical diffusion operator with a spatially varying diffusion
  coefficient.

 # Usage

 parent_model  = OceanModel{FT}(prob...)
 linear_model  = IVDCModel( parent_model )
 	
"""

# Create a new child linear model instance, attached to whatever parent
# BalanceLaw instantiates this.
# (Not sure we need parent, but maybe we will get some parameters from it)
struct IVDCModel{M} <: AbstractOceanModel
    parent_om::M
    function IVDCModel(parent_om::M;) where {M}
        return new{M}(parent_om)
    end
end

"""
 Set model state variables and operators
"""

# State variable and initial value, just one for now, θ

vars_state(m::IVDCModel, ::Prognostic, FT) = @vars(θ::FT)

function init_state_prognostic!(m::IVDCModel, Q::Vars, A::Vars, localgeo, t)
    @inbounds begin
        Q.θ = -0
    end
    return nothing
end

vars_state(m::IVDCModel, ::Auxiliary, FT) = @vars(θ_init::FT)
function init_state_auxiliary!(m::IVDCModel, A::Vars, _...)
    @inbounds begin
        A.θ_init = -0
    end
    return nothing
end

# Variables and operations used in differentiating first derivatives

vars_state(m::IVDCModel, ::Gradient, FT) = @vars(∇θ::FT, ∇θ_init::FT,)

@inline function compute_gradient_argument!(
    m::IVDCModel,
    G::Vars,
    Q::Vars,
    A,
    t,
)
    G.∇θ = Q.θ
    G.∇θ_init = A.θ_init

    return nothing
end

# Variables and operations used in differentiating second derivatives

vars_state(m::IVDCModel, ::GradientFlux, FT) = @vars(κ∇θ::SVector{3, FT})

@inline function compute_gradient_flux!(
    m::IVDCModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)

    κ = diffusivity_tensor(m, G.∇θ_init[3])
    D.κ∇θ = -κ * G.∇θ

    return nothing
end

@inline function diffusivity_tensor(m::IVDCModel, ∂θ∂z)
    κᶻ = m.parent_om.κᶻ * 0.5
    κᶜ = m.parent_om.κᶜ
    ∂θ∂z < 0 ? κ = (@SVector [0, 0, κᶜ]) : κ = (@SVector [0, 0, κᶻ])
    # ∂θ∂z <= 1e-10 ? κ = (@SVector [0, 0, κᶜ]) : κ = (@SVector [0, 0, κᶻ])

    return Diagonal(-κ)
end

# Function to apply I to state variable

@inline function source!(
    m::IVDCModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t,
    direction,
)
    #ivdc_dt = m.ivdc_dt
    ivdc_dt = m.parent_om.ivdc_dt
    @inbounds begin
        S.θ = Q.θ / ivdc_dt
        # S.θ = 0
    end

    return nothing
end

## Numerical fluxes and boundaries

function flux_first_order!(::IVDCModel, _...) end

function flux_second_order!(
    ::IVDCModel,
    F::Grad,
    S::Vars,
    D::Vars,
    H::Vars,
    A::Vars,
    t,
)
    F.θ += D.κ∇θ
    # F.θ = 0

end

function wavespeed(m::IVDCModel, n⁻, _...)
    C = abs(SVector(m.parent_om.cʰ, m.parent_om.cʰ, m.parent_om.cᶻ)' * n⁻)
    # C = abs(SVector(m.parent_om.cʰ, m.parent_om.cʰ, 50)' * n⁻)
    # C = abs(SVector(1, 1, 1)' * n⁻)
    ### C = abs(SVector(10, 10, 10)' * n⁻)
    # C = abs(SVector(50, 50, 50)' * n⁻)
    # C = abs(SVector( 0,  0,  0)' * n⁻)
    return C
end

function boundary_state!(
    nf::Union{
        NumericalFluxFirstOrder,
        NumericalFluxGradient,
        CentralNumericalFluxGradient,
    },
    m::IVDCModel,
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    bctype,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ

    return nothing
end

###    From -  function numerical_boundary_flux_gradient! , DGMethods/NumericalFluxes.jl
###    boundary_state!(
###        numerical_flux,
###        balance_law,
###        state_conservative⁺,
###        state_auxiliary⁺,
###        normal_vector,
###        state_conservative⁻,
###        state_auxiliary⁻,
###        bctype,
###        t,
###        state1⁻,
###        aux1⁻,
###    )

function boundary_state!(
    nf::Union{NumericalFluxSecondOrder, CentralNumericalFluxSecondOrder},
    m::IVDCModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    bctype,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ
    D⁺.κ∇θ = n⁻ * -0
    # D⁺.κ∇θ = n⁻ * -0 + 7000
    # D⁺.κ∇θ = -D⁻.κ∇θ

    return nothing
end

###    boundary_state!(
###        numerical_flux,
###        balance_law,
###        state_conservative⁺,
###        state_gradient_flux⁺,
###        state_auxiliary⁺,
###        normal_vector,
###        state_conservative⁻,
###        state_gradient_flux⁻,
###        state_auxiliary⁻,
###        bctype,
###        t,
###        state1⁻,
###        diff1⁻,
###        aux1⁻,
###    )

###    boundary_flux_second_order!(
###        numerical_flux,
###        balance_law,
###        Grad{S}(flux),
###        state_conservative⁺,
###        state_gradient_flux⁺,
###        state_hyperdiffusive⁺,
###        state_auxiliary⁺,
###        normal_vector,
###        state_conservative⁻,
###        state_gradient_flux⁻,
###        state_hyperdiffusive⁻,
###        state_auxiliary⁻,
###        bctype,
###        t,
###        state1⁻,
###        diff1⁻,
###        aux1⁻,
###    )
