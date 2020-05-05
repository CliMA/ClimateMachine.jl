module HydrostaticBoussinesq

export HydrostaticBoussinesqModel, AbstractHydrostaticBoussinesqProblem

using StaticArrays
using LinearAlgebra: dot, Diagonal
using CLIMAParameters.Planet: grav

using ..VariableTemplates
using ..MPIStateArrays
using ..Mesh.Filters: apply!
using ..Mesh.Grids: VerticalDirection
using ..DGmethods:
    BalanceLaw,
    LocalGeometry,
    DGModel,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    nodal_update_auxiliary_state!,
    copy_stack_field_down!
using ..DGmethods.NumericalFluxes: RusanovNumericalFlux

import ..DGmethods.NumericalFluxes: update_penalty!
import ..DGmethods:

    vars_state_conservative,
    init_state_conservative!,
    vars_state_auxiliary,
    init_state_auxiliary!,
    vars_state_gradient,
    compute_gradient_argument!,
    vars_state_gradient_flux,
    compute_gradient_flux!,
    vars_integrals,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    vars_reverse_integrals,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)

abstract type AbstractHydrostaticBoussinesqProblem end

"""
    HydrostaticBoussinesqModel <: BalanceLaw

A `BalanceLaw` for ocean modeling.

write out the equations here

ρₒ = reference density of sea water
cʰ = maximum horizontal wave speed
cᶻ = maximum vertical wave speed
αᵀ = thermal expansitivity coefficient
νʰ = horizontal viscosity
νᶻ = vertical viscosity
κʰ = horizontal diffusivity
κᶻ = vertical diffusivity
fₒ = first coriolis parameter (constant term)
β  = second coriolis parameter (linear term)

# Usage

    HydrostaticBoussinesqModel(problem)

"""
struct HydrostaticBoussinesqModel{PS, P, T} <: BalanceLaw
    param_set::PS
    problem::P
    ρₒ::T
    cʰ::T
    cᶻ::T
    αᵀ::T
    νʰ::T
    νᶻ::T
    κʰ::T
    κᶻ::T
    fₒ::T
    β::T
    function HydrostaticBoussinesqModel{FT}(
        param_set::PS,
        problem;
        ρₒ = FT(1000),  # kg / m^3
        cʰ = FT(0),     # m/s
        cᶻ = FT(0),     # m/s
        αᵀ = FT(2e-4),  # (m/s)^2 / K
        νʰ = FT(5e3),   # m^2 / s
        νᶻ = FT(5e-3),  # m^2 / s
        κʰ = FT(1e3),   # m^2 / s
        κᶻ = FT(1e-4),  # m^2 / s
        fₒ = FT(1e-4),  # Hz
        β = FT(1e-11), # Hz / m
    ) where {FT <: AbstractFloat, PS}
        return new{PS, typeof(problem), FT}(
            param_set,
            problem,
            ρₒ,
            cʰ,
            cᶻ,
            αᵀ,
            νʰ,
            νᶻ,
            κʰ,
            κᶻ,
            fₒ,
            β,
        )
    end
end
HBModel = HydrostaticBoussinesqModel

"""
    vars_state_conservative(::HBModel)

prognostic variables evolved forward in time

u = (u,v) = (zonal velocity, meridional velocity)
η = sea surface height
θ = temperature
"""
# If this order is changed check the filter usage!
function vars_state_conservative(m::HBModel, T)
    @vars begin
        u::SVector{2, T}
        η::T # real a 2-D variable TODO: should be 2D
        θ::T
    end
end

"""
    init_state_conservative!(::HBModel)

sets the initial value for state variables
dispatches to ocean_init_state! which is defined in a problem file such as SimpleBoxProblem.jl
"""
function ocean_init_state! end
function init_state_conservative!(m::HBModel, Q::Vars, A::Vars, coords, t)
    return ocean_init_state!(m.problem, Q, A, coords, t)
end

"""
    vars_state_auxiliary(::HBModel)
helper variables for computation

first half is because there is no dedicated integral kernels
these variables are used to compute vertical integrals
w = vertical velocity
wz0 = w at z = 0
pkin = bulk hydrostatic pressure contribution


second half of these are fields that are used for computation
y = north-south coordinate

"""
# If this order is changed check update_auxiliary_state!
function vars_state_auxiliary(m::HBModel, T)
    @vars begin
        w::T     # ∫(-∇⋅u)
        pkin::T  # ∫(-αᵀθ)
        wz0::T   # w at z=0
        y::T     # y-coordinate of the box
    end
end

"""
    init_state_auxiliary!(::HBModel)

sets the initial value for auxiliary variables (those that aren't related to vertical integrals)
dispatches to ocean_init_aux! which is defined in a problem file such as SimpleBoxProblem.jl
"""
function ocean_init_aux! end
function init_state_auxiliary!(m::HBModel, A::Vars, geom::LocalGeometry)
    return ocean_init_aux!(m, m.problem, A, geom)
end

"""
    vars_state_gradient(::HBModel)

variables that you want to take a gradient of
these are just copies in our model
"""
function vars_state_gradient(m::HBModel, T)
    @vars begin
        ∇u::SVector{2, T}
        ∇θ::T
    end
end

"""
    compute_gradient_argument!(::HBModel)

copy u and θ to var_gradient
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `G`: array of gradient variables
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function compute_gradient_argument!(m::HBModel, G::Vars, Q::Vars, A, t)
    G.∇u = Q.u
    G.∇θ = Q.θ

    return nothing
end

"""
    vars_state_gradient_flux(::HBModel)

the output of the gradient computations
multiplies ∇u by viscosity tensor and ∇θ by the diffusivity tensor
"""
function vars_state_gradient_flux(m::HBModel, T)
    @vars begin
        ν∇u::SMatrix{3, 2, T, 6}
        κ∇θ::SVector{3, T}
    end
end

"""
    compute_gradient_flux!(::HBModel)

copy ∇u and ∇θ to var_diffusive
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
    m::HBModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    ν = viscosity_tensor(m)
    D.ν∇u = ν * G.∇u

    κ = diffusivity_tensor(m, G.∇θ[3])
    D.κ∇θ = κ * G.∇θ

    return nothing
end

"""
    viscosity_tensor(::HBModel)

uniform viscosity with different values for horizontal and vertical directions

# Arguments
- `m`: model object to dispatch on and get viscosity parameters
"""
@inline viscosity_tensor(m::HBModel) = Diagonal(@SVector [m.νʰ, m.νʰ, m.νᶻ])

"""
    diffusivity_tensor(::HBModel)

uniform diffusivity in the horizontal direction
applies convective adjustment in the vertical, bump by 1000 if ∂θ∂z < 0

# Arguments
- `m`: model object to dispatch on and get diffusivity parameters
- `∂θ∂z`: value of the derivative of temperature in the z-direction
"""
@inline function diffusivity_tensor(m::HBModel, ∂θ∂z)
    ∂θ∂z < 0 ? κ = (@SVector [m.κʰ, m.κʰ, 1000 * m.κᶻ]) : κ =
        (@SVector [m.κʰ, m.κʰ, m.κᶻ])

    return Diagonal(κ)
end

"""
    vars_integral(::HBModel)

location to store integrands for bottom up integrals
∇hu = the horizontal divegence of u, e.g. dw/dz
"""
function vars_integrals(m::HBModel, T)
    @vars begin
        ∇hu::T
        αᵀθ::T
    end
end

"""
    integral_load_auxiliary_state!(::HBModel)

copy w to var_integral
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
I -> array of integrand variables
Q -> array of state variables
A -> array of aux variables
"""
@inline function integral_load_auxiliary_state!(
    m::HBModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∇hu = A.w # borrow the w value from A...
    I.αᵀθ = -m.αᵀ * Q.θ # integral will be reversed below

    return nothing
end

"""
    integral_set_auxiliary_state!(::HBModel)

copy integral results back out to aux
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
A -> array of aux variables
I -> array of integrand variables
"""
@inline function integral_set_auxiliary_state!(m::HBModel, A::Vars, I::Vars)
    A.w = I.∇hu
    A.pkin = I.αᵀθ

    return nothing
end

"""
    vars_reverse_integral(::HBModel)

location to store integrands for top down integrals
αᵀθ = density perturbation
"""
function vars_reverse_integrals(m::HBModel, T)
    @vars begin
        αᵀθ::T
    end
end

"""
    reverse_integral_load_auxiliary_state!(::HBModel)

copy αᵀθ to var_reverse_integral
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
I -> array of integrand variables
A -> array of aux variables
"""
@inline function reverse_integral_load_auxiliary_state!(
    m::HBModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.αᵀθ = A.pkin

    return nothing
end

"""
    reverse_integral_set_auxiliary_state!(::HBModel)

copy reverse integral results back out to aux
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
A -> array of aux variables
I -> array of integrand variables
"""
@inline function reverse_integral_set_auxiliary_state!(
    m::HBModel,
    A::Vars,
    I::Vars,
)
    A.pkin = I.αᵀθ

    return nothing
end

"""
    flux_first_order!(::HBModel)

calculates the hyperbolic flux contribution to state variables
this computation is done pointwise at each nodal point

# arguments:
m -> model in this case HBModel
F -> array of fluxes for each state variable
Q -> array of state variables
A -> array of aux variables
t -> time, not used

# computations
∂ᵗu = ∇⋅(g*η + g∫αᵀθdz + v⋅u)
∂ᵗθ = ∇⋅(vθ) where v = (u,v,w)
"""
@inline function flux_first_order!(
    m::HBModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
    direction,
)
    FT = eltype(Q)
    _grav::FT = grav(m.param_set)
    @inbounds begin
        u = Q.u # Horizontal components of velocity
        η = Q.η
        θ = Q.θ
        w = A.w   # vertical velocity
        pkin = A.pkin

        v = @SVector [u[1], u[2], w]
        Iʰ = @SMatrix [
            1 -0
            -0 1
            -0 -0
        ]

        # ∇h • (g η)
        F.u += _grav * η * Iʰ

        # ∇h • (- ∫(αᵀ θ))
        F.u += _grav * pkin * Iʰ

        # ∇h • (v ⊗ u)
        # F.u += v * u'

        # ∇ • (u θ)
        F.θ += v * θ
    end

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
    m::HBModel,
    F::Grad,
    Q::Vars,
    D::Vars,
    HD::Vars,
    A::Vars,
    t::Real,
)
    F.u -= D.ν∇u
    F.θ -= D.κ∇θ

    return nothing
end

"""
    source!(::HBModel)
    calculates the source term contribution to state variables
    this computation is done pointwise at each nodal point

    arguments:
    m -> model in this case HBModel
    F -> array of fluxes for each state variable
    Q -> array of state variables
    A -> array of aux variables
    t -> time, not used

    computations
    ∂ᵗu = -f×u
    ∂ᵗη = w|(z=0)
"""
@inline function source!(
    m::HBModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
    direction,
)
    @inbounds begin
        u, v = Q.u # Horizontal components of velocity
        wz0 = A.wz0

        # f × u
        f = coriolis_force(m, A.y)
        S.u -= @SVector [-f * v, f * u]

        S.η += wz0
    end

    return nothing
end

"""
    coriolis_force(::HBModel)

northern hemisphere coriolis

# Arguments
- `m`: model object to dispatch on and get coriolis parameters
- `y`: y-coordinate in the box
"""
@inline coriolis_force(m::HBModel, y) = m.fₒ + m.β * y

"""
    wavespeed(::HBModel)

calculates the wavespeed for rusanov flux
"""
@inline wavespeed(m::HBModel, n⁻, _...) = abs(SVector(m.cʰ, m.cʰ, m.cᶻ)' * n⁻)

"""
    update_penalty(::HBModel)
    set Δη = 0 when computing numerical fluxes
"""
# We want not have jump penalties on η (since not a flux variable)
function update_penalty!(
    ::RusanovNumericalFlux,
    ::HBModel,
    n⁻,
    λ,
    ΔQ::Vars,
    Q⁻,
    A⁻,
    Q⁺,
    A⁺,
    t,
)
    ΔQ.η = -0

    return nothing
end

"""
    update_auxiliary_state!(::HBModel)

    applies the vertical filter to the zonal and meridional velocities to preserve numerical incompressibility
    applies an exponential filter to θ to anti-alias the non-linear advective term

    doesn't actually touch the aux variables any more, but we need a better filter interface than this anyways
"""
function update_auxiliary_state!(
    dg::DGModel,
    m::HBModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    MD = dg.modeldata

    # `update_aux!` gets called twice, once for the real elements and once for
    # the ghost elements.  Only apply the filters to the real elems.
    if elems == dg.grid.topology.realelems
        # required to ensure that after integration velocity field is divergence free
        vert_filter = MD.vert_filter
        # Q[1] = u[1] = u, Q[2] = u[2] = v
        apply!(Q, (1, 2), dg.grid, vert_filter, VerticalDirection())

        exp_filter = MD.exp_filter
        # Q[4] = θ
        apply!(Q, (4,), dg.grid, exp_filter, VerticalDirection())
    end

    return true
end

"""
    update_auxiliary_state_gradient!(::HBModel)

    ∇hu to w for integration
    performs integration for w and pkin (should be moved to its own integral kernels)
    copies down w and wz0 because we don't have 2D structures

    now for actual update aux stuff
    implements convective adjustment by bumping the vertical diffusivity up by a factor of 1000 if dθdz < 0
"""
function update_auxiliary_state_gradient!(
    dg::DGModel,
    m::HBModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    A = dg.state_auxiliary

    # store ∇ʰu as integrand for w
    function f!(m::HBModel, Q, A, D, t)
        @inbounds begin
            ν = viscosity_tensor(m)
            ∇u = ν \ D.ν∇u
            A.w = -(∇u[1, 1] + ∇u[2, 2])
        end

        return nothing
    end
    nodal_update_auxiliary_state!(f!, dg, m, Q, t, elems; diffusive = true)

    # compute integrals for w and pkin
    indefinite_stack_integral!(dg, m, Q, A, t, elems) # bottom -> top
    reverse_indefinite_stack_integral!(dg, m, Q, A, t, elems) # top -> bottom

    # project w(z=0) down the stack
    # Need to be consistent with vars_state_auxiliary
    # A[1] = w, A[3] = wz0
    copy_stack_field_down!(dg, m, A, 1, 3, elems)

    return true
end

include("SimpleBoxProblem.jl")
include("LinearHBModel.jl")
include("BoundaryConditions.jl")
include("Courant.jl")

end
