module HydrostaticBoussinesq

export HydrostaticBoussinesqModel, AbstractHydrostaticBoussinesqProblem, OceanDGModel,
       LinearHBModel, calculate_dt

using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using ..VariableTemplates
using ..MPIStateArrays
using ..DGmethods: init_ode_state
using ..PlanetParameters: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids: polynomialorder, VerticalDirection, HorizontalDirection, min_node_distance

using ..DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                   CentralNumericalFluxDiffusive,
                                   CentralNumericalFluxNonDiffusive

import ..DGmethods.NumericalFluxes: update_penalty!, numerical_flux_diffusive!,
                                    NumericalFluxNonDiffusive

import ..DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                    vars_diffusive, flux_nondiffusive!,
                    flux_diffusive!, source!, wavespeed,
                    boundary_state!, update_aux!, update_aux_diffusive!,
                    gradvariables!, init_aux!, init_state!,
                    LocalGeometry, DGModel, nodal_update_aux!, diffusive!,
                    copy_stack_field_down!, create_state, calculate_dt,
                    vars_integrals, vars_reverse_integrals,
                    indefinite_stack_integral!,
                    reverse_indefinite_stack_integral!,
                    integral_load_aux!, integral_set_aux!,
                    reverse_integral_load_aux!,
                    reverse_integral_set_aux!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

abstract type AbstractHydrostaticBoussinesqProblem end

"""
    HydrostaticBoussinesqModel <: BalanceLaw

A `BalanceLaw` for ocean modeling.

write out the equations here

# Usage

    HydrostaticBoussinesqModel(problem)

"""
struct HydrostaticBoussinesqModel{P,T} <: BalanceLaw
  problem::P
  cʰ::T
  cᶻ::T
  αᵀ::T
  νʰ::T
  νᶻ::T
  κʰ::T
  κᶻ::T
  function HydrostaticBoussinesqModel{FT}(problem;
                                      cʰ = FT(0),     # m/s
                                      cᶻ = FT(0),     # m/s
                                      αᵀ = FT(2e-4),  # (m/s)^2 / K
                                      νʰ = FT(5e3),   # m^2 / s
                                      νᶻ = FT(5e-3),  # m^2 / s
                                      κʰ = FT(1e3),   # m^2 / s
                                      κᶻ = FT(1e-4),  # m^2 / s
                                      ) where {FT <: AbstractFloat}
    return new{typeof(problem),FT}(problem, cʰ, cᶻ, αᵀ, νʰ, νᶻ, κʰ, κᶻ)
  end
end
HBModel = HydrostaticBoussinesqModel

"""
    calculate_dt(grid, model::HBModel)

calculates the time step based on grid spacing and model parameters
takes minimum of gravity wave, diffusive, and viscous CFL
factor of 1000 in the diffusive CFL is to handle the convective adjustment

"""
function calculate_dt(grid, model::HBModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())
    minΔz = min_node_distance(grid, VerticalDirection())

    CFL_gravity = minΔx / model.cʰ
    CFL_diffusive = minΔz^2 / (1000 * model.κᶻ)
    CFL_viscous = minΔz^2 / model.νᶻ

    dt = 1//2 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

"""
    OceanDGModel()

helper function to add required filtering
not used in the Driver+Config setup
"""
function OceanDGModel(bl::HBModel, grid, numfluxnondiff, numfluxdiff,
                      gradnumflux; kwargs...)
  vert_filter = CutoffFilter(grid, polynomialorder(grid)-1)
  exp_filter  = ExponentialFilter(grid, 1, 8)

  modeldata = (vert_filter = vert_filter, exp_filter=exp_filter)

  return DGModel(bl, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 kwargs..., modeldata=modeldata)
end

"""
    vars_state(::HBModel)

prognostic variables evolved forward in time

u = (u,v) = (zonal velocity, meridional velocity)
η = sea surface height
θ = temperature
"""
# If this order is changed check the filter usage!
function vars_state(m::HBModel, T)
  @vars begin
    u::SVector{2, T}
    η::T # real a 2-D variable TODO: should be 2D
    θ::T
  end
end

"""
    init_state!(::HBModel)

sets the initial value for state variables
dispatches to ocean_init_state! which is defined in a problem file such as SimpleBoxProblem.jl
"""
function ocean_init_state! end
function init_state!(m::HBModel, Q::Vars, A::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, A, coords, t)
end

"""
    vars_aux(::HBModel)
helper variables for computation

first half is because there is no dedicated integral kernels
these variables are used to compute vertical integrals
w = vertical velocity
w_reverse = w but integrated in the opposite direction
wz0 = w at z = 0
pkin = bulk hydrostatic pressure contribution
pkin_reverse = pkin but integrated in the opposite direction

second half of these are fields that are used for computation
θʳ = relaxation value of the sea surface temperature
f = coriolis force
τ = wind stress
ν = vector of viscosities (zonal, meridional, vertical)
κ = vector of diffusivities (zonal, meridional, vertical)
"""
# If this order is changed check update_aux!
function vars_aux(m::HBModel, T)
  @vars begin
    w::T
    pkin::T         # ∫(-αᵀ θ)
    wz0::T          # w at z=0
    θʳ::T           # SST given    # TODO: Should be 2D
    f::T            # coriolis
    τ::T            # wind stress  # TODO: Should be 2D
    ν::SVector{3, T}
    κ::SVector{3, T}
  end
end

"""
    init_aux!(::HBModel)

sets the initial value for auxiliary variables (those that aren't related to vertical integrals)
dispatches to ocean_init_aux! which is defined in a problem file such as SimpleBoxProblem.jl
"""
function ocean_init_aux! end
function init_aux!(m::HBModel, A::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m, m.problem, A, geom)
end

"""
    vars_gradient(::HBModel)

variables that you want to take a gradient of
these are just copies in our model
"""
function vars_gradient(m::HBModel, T)
  @vars begin
    u::SVector{2, T}
    θ::T
  end
end

"""
    gradvariables!(::HBModel)
    
copy u and θ to var_gradient
this computation is done pointwise at each nodal point

# arguments:
- `m`: model in this case HBModel
- `G`: array of gradient variables
- `Q`: array of state variables
- `A`: array of aux variables
- `t`: time, not used
"""
@inline function gradvariables!(m::HBModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

"""
    vars_diffusive(::HBModel)

the output of the gradient computations
once again just copies, we don't do any transforms or reductions
"""
function vars_diffusive(m::HBModel, T)
  @vars begin
    ∇u::SMatrix{3, 2, T, 6}
    ∇θ::SVector{3, T}
  end
end

"""
    diffusive!(::HBModel)

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
@inline function diffusive!(m::HBModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end

"""
    vars_integral(::HBModel)

location to store integrands for bottom up integrals
∇hu = the horizontal divegence of u, e.g. dw/dz
"""
function vars_integrals(m::HBModel, T)
  @vars begin
    ∇hu::T
  end
end

"""
    integral_load_aux!(::HBModel)

copy w to var_integral
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
I -> array of integrand variables
Q -> array of state variables
A -> array of aux variables
"""
@inline function integral_load_aux!(m::HBModel, I::Vars, Q::Vars, A::Vars)
  I.∇hu = A.w # borrow the w value from A...

  return nothing
end

"""
    integral_set_aux!(::HBModel)

copy integral results back out to aux
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
A -> array of aux variables
I -> array of integrand variables
"""
@inline function integral_set_aux!(m::HBModel, A::Vars, I::Vars)
  A.w = I.∇hu

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
    reverse_integral_load_aux!(::HBModel)

copy αᵀθ to var_reverse_integral
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
I -> array of integrand variables
A -> array of aux variables
"""
@inline function reverse_integral_load_aux!(m::HBModel, I::Vars, Q::Vars, A::Vars)
  I.αᵀθ = A.pkin

  return nothing
end

"""
    reverse_integral_set_aux!(::HBModel)

copy reverse integral results back out to aux
this computation is done pointwise at each nodal point

arguments:
m -> model in this case HBModel
A -> array of aux variables
I -> array of integrand variables
"""
@inline function reverse_integral_set_aux!(m::HBModel, A::Vars, I::Vars)
  A.pkin = I.αᵀθ

  return nothing
end

"""
    flux_nondiffusive!(::HBModel)

calculates the hyperbolic flux contribution to state variables
this computation is done pointwise at each nodal point

# arguments:
m -> model in this case HBModel
F -> array of fluxes for each state variable
Q -> array of state variables
A -> array of aux variables
t -> time, not used

# computations
∂ᵗu = ∇∘(g*η + g∫αᵀθdz + v∘u)
∂ᵗθ = ∇∘(vθ) where v = (u,v,w)
"""
@inline function flux_nondiffusive!(m::HBModel, F::Grad, Q::Vars,
                                    A::Vars, t::Real)
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    η = Q.η
    θ = Q.θ
    w = A.w   # vertical velocity
    pkin = A.pkin

    v = @SVector [u[1], u[2], w]
    Ih = @SMatrix [ 1 -0;
                   -0  1;
                   -0 -0]

    # ∇h • (g η)
    F.u += grav * η * Ih

    # ∇h • (- ∫(αᵀ θ))
    F.u += grav * pkin * Ih

    # ∇h • (v ⊗ u)
    # F.u += v * u'

    # ∇ • (u θ)
    F.θ += v * θ
  end

  return nothing
end

"""
    flux_diffusive!(::HBModel)

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
∂ᵗu = -∇∘(ν∇u)
∂ᵗθ = -∇∘(κ∇θ)
"""
@inline function flux_diffusive!(m::HBModel, F::Grad, Q::Vars, D::Vars,
                                 HD::Vars, A::Vars, t::Real)
  F.u -= Diagonal(A.ν) * D.∇u
  F.θ -= Diagonal(A.κ) * D.∇θ

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
@inline function source!(m::HBModel{P}, source::Vars, Q::Vars,
                         diffusive::Vars, A::Vars, t::Real) where P
  @inbounds begin
    u,v = Q.u # Horizontal components of velocity
    f = A.f
    wz0 = A.wz0

    # f × u
    source.u -= @SVector [-f * v, f * u]

    source.η += wz0
  end

  return nothing
end

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
function update_penalty!(::Rusanov, ::HBModel, n⁻, λ, ΔQ::Vars,
                         Q⁻, A⁻, Q⁺, A⁺, t)
  ΔQ.η = -0

  return nothing
end

"""
    update_aux!(::HBModel)

    applies the vertical filter to the zonal and meridional velocities to preserve numerical incompressibility
    applies an exponential filter to θ to anti-alias the non-linear advective term

    doesn't actually touch the aux variables any more, but we need a better filter interface than this anyways
"""
function update_aux!(dg::DGModel, m::HBModel, Q::MPIStateArray, t::Real)
  MD = dg.modeldata

  # required to ensure that after integration velocity field is divergence free
  vert_filter = MD.vert_filter
  # Q[1] = u[1] = u, Q[2] = u[2] = v
  apply!(Q, (1, 2), dg.grid, vert_filter, VerticalDirection())

  exp_filter = MD.exp_filter
  # Q[4] = θ
  apply!(Q, (4,), dg.grid, exp_filter, VerticalDirection())

  return true
end

"""
    update_aux_diffusive!(::HBModel)

    ∇hu to w for integration
    performs integration for w and pkin (should be moved to its own integral kernels)
    copies down w and wz0 because we don't have 2D structures

    now for actual update aux stuff
    implements convective adjustment by bumping the vertical diffusivity up by a factor of 1000 if dθdz < 0
"""
function update_aux_diffusive!(dg::DGModel, m::HBModel, Q::MPIStateArray, t::Real)
  A  = dg.auxstate

  # store ∇ʰu as integrand for w
  # update vertical diffusivity for convective adjustment
  function f!(m::HBModel, Q, A, D, t)
    @inbounds begin
      A.w = -(D.∇u[1,1] + D.∇u[2,2])
      A.pkin = -m.αᵀ * Q.θ

      D.∇θ[3] < 0 ? A.κ = (m.κʰ, m.κʰ, 1000 * m.κᶻ) : A.κ = (m.κʰ, m.κʰ, m.κᶻ)
    end

    return nothing
  end
  nodal_update_aux!(f!, dg, m, Q, t; diffusive=true)

  # compute integrals for w and pkin
  indefinite_stack_integral!(dg, m, Q, A, t) # bottom -> top
  reverse_indefinite_stack_integral!(dg, m, Q, A, t) # top -> bottom

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  # A[1] = w, A[3] = wz0
  copy_stack_field_down!(dg, m, A, 1, 3)

  return true
end

"""
    boundary_state!(nf, ::HBModel, Q⁺, A⁺, Q⁻, A⁻, bctype)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, m::HBModel, Q⁺::Vars, A⁺::Vars, n⁻,
                                 Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(m, m.problem, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end

"""
    boundary_state!(nf, ::HBModel, Q⁺, D⁺, A⁺, Q⁻, D⁻, A⁻, bctype)

applies boundary conditions for the parabolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(nf, m::HBModel,
                                 Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻,
                                 Q⁻::Vars, D⁻::Vars, A⁻::Vars,
                                 bctype, t, _...)
  return ocean_boundary_state!(m, m.problem, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end

include("OceanBoundaryConditions.jl")
include("LinearHBModel.jl")

end
