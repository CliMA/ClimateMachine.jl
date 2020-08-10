module HydrostaticBoussinesq

export HydrostaticBoussinesqModel

using StaticArrays
using LinearAlgebra: dot, Diagonal
using CLIMAParameters.Planet: grav

using ..Ocean
using ...VariableTemplates
using ...MPIStateArrays
using ...Mesh.Filters: apply!
using ...Mesh.Grids: VerticalDirection
using ...Mesh.Geometry
using ...DGMethods
using ...DGMethods: nodal_init_state_auxiliary!
using ...DGMethods.NumericalFluxes
using ...DGMethods.NumericalFluxes: RusanovNumericalFlux
using ...BalanceLaws

import ..Ocean: coriolis_parameter
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
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!
import ..Ocean: ocean_init_state!, ocean_init_aux!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

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
struct HydrostaticBoussinesqModel{C, PS, P, T} <: BalanceLaw
    param_set::PS
    problem::P
    coupling::C
    ρₒ::T
    cʰ::T
    cᶻ::T
    αᵀ::T
    νʰ::T
    νᶻ::T
    κʰ::T
    κᶻ::T
    κᶜ::T
    fₒ::T
    β::T
    function HydrostaticBoussinesqModel{FT}(
        param_set::PS,
        problem::P;
        coupling::C = Uncoupled(),
        ρₒ = FT(1000),  # kg / m^3
        cʰ = FT(0),     # m/s
        cᶻ = FT(0),     # m/s
        αᵀ = FT(2e-4),  # (m/s)^2 / K
        νʰ = FT(5e3),   # m^2 / s
        νᶻ = FT(5e-3),  # m^2 / s
        κʰ = FT(1e3),   # m^2 / s # horizontal diffusivity
        κᶻ = FT(1e-4),  # m^2 / s # background vertical diffusivity
        κᶜ = FT(1e-1),  # m^2 / s # diffusivity for convective adjustment
        fₒ = FT(1e-4),  # Hz
        β = FT(1e-11), # Hz / m
    ) where {FT <: AbstractFloat, PS, P, C}
        return new{C, PS, P, FT}(
            param_set,
            problem,
            coupling,
            ρₒ,
            cʰ,
            cᶻ,
            αᵀ,
            νʰ,
            νᶻ,
            κʰ,
            κᶻ,
            κᶜ,
            fₒ,
            β,
        )
    end
end
HBModel = HydrostaticBoussinesqModel

"""
    vars_state(::HBModel, ::Prognostic)

prognostic variables evolved forward in time

u = (u,v) = (zonal velocity, meridional velocity)
η = sea surface height
θ = temperature
"""
function vars_state(m::HBModel, ::Prognostic, T)
    @vars begin
        u::SVector{2, T}
        η::T # real a 2-D variable TODO: should be 2D
        θ::T
    end
end

"""
    init_state_prognostic!(::HBModel)

sets the initial value for state variables
dispatches to ocean_init_state! which is defined in a problem file such as SimpleBoxProblem.jl
"""
function init_state_prognostic!(m::HBModel, Q::Vars, A::Vars, coords, t)
    return ocean_init_state!(m, m.problem, Q, A, coords, t)
end

"""
    vars_state(::HBModel, ::Auxiliary)
helper variables for computation

second half is because there is no dedicated integral kernels
these variables are used to compute vertical integrals
w = vertical velocity
wz0 = w at z = 0
pkin = bulk hydrostatic pressure contribution


first half of these are fields that are used for computation
y = north-south coordinate

"""
function vars_state(m::HBModel, ::Auxiliary, T)
    @vars begin
        y::T     # y-coordinate of the box
        w::T     # ∫(-∇⋅u)
        pkin::T  # ∫(-αᵀθ)
        wz0::T   # w at z=0
        uᵈ::SVector{2, T}    # velocity deviation from vertical mean
        ΔGᵘ::SVector{2, T}   # vertically averaged tendency
    end
end

function ocean_init_aux! end

"""
    init_state_auxiliary!(::HBModel)

sets the initial value for auxiliary variables (those that aren't related to vertical integrals)
dispatches to ocean_init_aux! which is defined in a problem file such as SimpleBoxProblem.jl
"""
function init_state_auxiliary!(m::HBModel, state_auxiliary::MPIStateArray, grid)
    nodal_init_state_auxiliary!(
        m,
        (m, A, tmp, geom) -> ocean_init_aux!(m, m.problem, A, geom),
        state_auxiliary,
        grid,
    )
end

"""
    vars_state(::HBModel, ::Gradient)

variables that you want to take a gradient of
these are just copies in our model
"""
function vars_state(m::HBModel, ::Gradient, T)
    @vars begin
        ∇u::SVector{2, T}
        ∇uᵈ::SVector{2, T}
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
    G.∇θ = Q.θ

    velocity_gradient_argument!(m, m.coupling, G, Q, A, t)

    return nothing
end

@inline function velocity_gradient_argument!(
    m::HBModel,
    ::Uncoupled,
    G,
    Q,
    A,
    t,
)
    G.∇u = Q.u

    return nothing
end

"""
    vars_state(::HBModel, ::GradientFlux, FT)

the output of the gradient computations
multiplies ∇u by viscosity tensor and ∇θ by the diffusivity tensor
"""
function vars_state(m::HBModel, ::GradientFlux, T)
    @vars begin
        ∇ʰu::T
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
    # store ∇ʰu for continuity equation (convert gradient to divergence)
    D.∇ʰu = G.∇u[1, 1] + G.∇u[2, 2]

    velocity_gradient_flux!(m, m.coupling, D, G, Q, A, t)

    κ = diffusivity_tensor(m, G.∇θ[3])
    D.κ∇θ = -κ * G.∇θ

    return nothing
end

@inline function velocity_gradient_flux!(m::HBModel, ::Uncoupled, D, G, Q, A, t)
    ν = viscosity_tensor(m)
    D.ν∇u = -ν * G.∇u

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
    ∂θ∂z < 0 ? κ = m.κᶜ : κ = m.κᶻ

    return Diagonal(@SVector [m.κʰ, m.κʰ, κ])
end

"""
    vars_integral(::HBModel)

location to store integrands for bottom up integrals
∇hu = the horizontal divegence of u, e.g. dw/dz
"""
function vars_state(m::HBModel, ::UpwardIntegrals, T)
    @vars begin
        ∇ʰu::T
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
    I.∇ʰu = A.w # borrow the w value from A...
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
    A.w = I.∇ʰu
    A.pkin = I.αᵀθ

    return nothing
end

"""
    vars_reverse_integral(::HBModel)

location to store integrands for top down integrals
αᵀθ = density perturbation
"""
function vars_state(m::HBModel, ::DownwardIntegrals, T)
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
    @inbounds begin
        # ∇h • (g η)
        hydrostatic_pressure!(m, m.coupling, F, Q, A, t)

        # ∇h • (- ∫(αᵀ θ))
        pkin = A.pkin
        Iʰ = @SMatrix [
            1 -0
            -0 1
            -0 -0
        ]
        F.u += grav(m.param_set) * pkin * Iʰ

        # ∇h • (v ⊗ u)
        # F.u += v * u'

        # ∇ • (u θ)
        θ = Q.θ
        v = @SVector [Q.u[1], Q.u[2], A.w]
        F.θ += v * θ
    end

    return nothing
end

@inline function hydrostatic_pressure!(m::HBModel, ::Uncoupled, F, Q, A, t)
    η = Q.η
    Iʰ = @SMatrix [
        1 -0
        -0 1
        -0 -0
    ]

    F.u += grav(m.param_set) * η * Iʰ

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
    F.u += D.ν∇u
    F.θ += D.κ∇θ

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
    # explicit forcing for SSH
    wz0 = A.wz0
    S.η += wz0

    coriolis_force!(m, m.coupling, S, Q, A, t)

    return nothing
end

@inline function coriolis_force!(m::HBModel, ::Uncoupled, S, Q, A, t)
    # f × u
    f = coriolis_parameter(m, A.y)
    u, v = Q.u # Horizontal components of velocity
    S.u -= @SVector [-f * v, f * u]

    return nothing
end

"""
    coriolis_parameter(::HBModel)

northern hemisphere coriolis

# Arguments
- `m`: model object to dispatch on and get coriolis parameters
- `y`: y-coordinate in the box
"""
@inline coriolis_parameter(m::HBModel, y) = m.fₒ + m.β * y

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
    FT = eltype(Q)
    MD = dg.modeldata

    # `update_aux!` gets called twice, once for the real elements and once for
    # the ghost elements.  Only apply the filters to the real elems.
    if elems == dg.grid.topology.realelems
        # required to ensure that after integration velocity field is divergence free
        vert_filter = MD.vert_filter
        apply!(Q, (:u,), dg.grid, vert_filter, direction = VerticalDirection())

        exp_filter = MD.exp_filter
        apply!(Q, (:θ,), dg.grid, exp_filter, direction = VerticalDirection())
    end

    compute_flow_deviation!(dg, m, m.coupling, Q, t)

    return true
end

@inline compute_flow_deviation!(dg, ::HBModel, ::Uncoupled, _...) = nothing

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
    FT = eltype(Q)
    A = dg.state_auxiliary

    # store ∇ʰu as integrand for w
    function f!(m::HBModel, Q, A, D, t)
        @inbounds begin
            # load -∇ʰu as ∂ᶻw
            A.w = -D.∇ʰu
        end

        return nothing
    end
    nodal_update_auxiliary_state!(f!, dg, m, Q, t, elems; diffusive = true)

    # compute integrals for w and pkin
    indefinite_stack_integral!(dg, m, Q, A, t, elems) # bottom -> top
    reverse_indefinite_stack_integral!(dg, m, Q, A, t, elems) # top -> bottom

    # We are unable to use vars (ie A.w) for this because this operation will
    # return a SubArray, and adapt (used for broadcasting along reshaped arrays)
    # has a limited recursion depth for the types allowed.
    number_aux = number_states(m, Auxiliary())
    index_w = varsindex(vars_state(m, Auxiliary(), FT), :w)
    index_wz0 = varsindex(vars_state(m, Auxiliary(), FT), :wz0)
    Nq, Nqk, _, _, nelemv, nelemh, nhorzrealelem, _ = basic_grid_info(dg)

    # project w(z=0) down the stack
    data = reshape(A.data, Nq^2, Nqk, number_aux, nelemv, nelemh)
    flat_wz0 = @view data[:, end:end, index_w, end:end, 1:nhorzrealelem]
    boxy_wz0 = @view data[:, :, index_wz0, :, 1:nhorzrealelem]
    boxy_wz0 .= flat_wz0

    return true
end

include("LinearHBModel.jl")
include("BoundaryConditions.jl")
include("Courant.jl")

end
