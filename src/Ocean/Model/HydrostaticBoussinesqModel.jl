module HydrostaticBoussinesq

export HydrostaticBoussinesqModel, HydrostaticBoussinesqProblem, OceanDGModel

using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using ..VariableTemplates
using ..MPIStateArrays
using ..DGmethods: init_ode_state
using ..PlanetParameters: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids: polynomialorder, VerticalDirection

using ..DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                   CentralNumericalFluxDiffusive,
                                   CentralNumericalFluxNonDiffusive

import ..DGmethods.NumericalFluxes: update_penalty!, numerical_flux_diffusive!,
                                    NumericalFluxNonDiffusive

import ..DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                    vars_diffusive, vars_integrals, flux_nondiffusive!,
                    flux_diffusive!, source!, wavespeed,
                    boundary_state!, update_aux!, update_aux_diffusive!,
                    gradvariables!, init_aux!, init_state!,
                    LocalGeometry, indefinite_stack_integral!,
                    reverse_indefinite_stack_integral!, integrate_aux!,
                    DGModel, nodal_update_aux!, diffusive!,
                    copy_stack_field_down!, create_state

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

abstract type OceanBoundaryCondition end
struct CoastlineFreeSlip             <: OceanBoundaryCondition end
struct CoastlineNoSlip               <: OceanBoundaryCondition end
struct OceanFloorFreeSlip            <: OceanBoundaryCondition end
struct OceanFloorNoSlip              <: OceanBoundaryCondition end
struct OceanSurfaceNoStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressNoForcing   <: OceanBoundaryCondition end
struct OceanSurfaceNoStressForcing   <: OceanBoundaryCondition end
struct OceanSurfaceStressForcing     <: OceanBoundaryCondition end

abstract type HydrostaticBoussinesqProblem end

struct HydrostaticBoussinesqModel{P,T} <: BalanceLaw
  problem::P
  c₁::T
  c₂::T
  c₃::T
  αᵀ::T
  νʰ::T
  νᶻ::T
  κʰ::T
  κᶻ::T
end

HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

function OceanDGModel(bl::HBModel, grid, numfluxnondiff, numfluxdiff,
                      gradnumflux; kwargs...)
  vert_filter = CutoffFilter(grid, polynomialorder(grid)-1)
  exp_filter  = ExponentialFilter(grid, 1, 8)

  modeldata = (vert_filter = vert_filter, exp_filter=exp_filter)

  return DGModel(bl, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 kwargs..., modeldata=modeldata)
end

# If this order is changed check the filter usage!
function vars_state(m::HBModel, T)
  @vars begin
    u::SVector{2, T}
    η::T # real a 2-D variable TODO: should be 2D
    θ::T
  end
end

# If this order is changed check update_aux!
function vars_aux(m::HBModel, T)
  @vars begin
    w::T
    pkin_reverse::T # ∫(-αᵀ θ) # TODO: remove me after better integral interface
    w_reverse::T               # TODO: remove me after better integral interface
    pkin::T         # ∫(-αᵀ θ)
    wz0::T          # w at z=0
    θʳ::T           # SST given    # TODO: Should be 2D
    f::T            # coriolis
    τ::T            # wind stress  # TODO: Should be 2D
    # κ::SMatrix{3, 3, T, 9} # diffusivity tensor (for convective adjustment)
    κᶻ::T
  end
end

function vars_gradient(m::HBModel, T)
  @vars begin
    u::SVector{2, T}
    θ::T
  end
end

function vars_diffusive(m::HBModel, T)
  @vars begin
    ∇u::SMatrix{3, 2, T, 6}
    ∇θ::SVector{3, T}
  end
end

function vars_integrals(m::HBModel, T)
  @vars begin
    ∇hu::T
    αᵀθ::T
  end
end

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

    # ∇ • (u θ)
    F.θ += v * θ

    # ∇h • (g η)
    F.u += grav * η * Ih

    # ∇h • (- ∫(αᵀ θ))
    F.u += grav * pkin * Ih

    # ∇h • (v ⊗ u)
    # F.u += v * u'

  end

  return nothing
end

@inline wavespeed(m::HBModel, n⁻, _...) = abs(SVector(m.c₁, m.c₂, m.c₃)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
function update_penalty!(::Rusanov, ::HBModel, n⁻, λ, ΔQ::Vars,
                         Q⁻, A⁻, Q⁺, A⁺, t)
  ΔQ.η = -0

  #=
  θ⁻ = Q⁻.θ
  u⁻ = Q⁻.u
  w⁻ = A⁻.w
  @inbounds v⁻ = @SVector [u⁻[1], u⁻[2], w⁻]
  n̂_v⁻ = n⁻∘v⁻

  θ⁺ = Q⁺.θ
  u⁺ = Q⁺.u
  w⁺ = A⁺.w
  @inbounds v⁺ = @SVector [u⁺[1], u⁺[2], w⁺]
  n̂_v⁺ = n⁻∘v⁺

  # max velocity
  # n̂∘v = (abs(n̂∘v⁺) > abs(n̂∘v⁻) ? n̂∘v⁺ : n̂∘v⁻

  # average velocity
  n̂_v = (n̂_v⁻ + n̂_v⁺) / 2

  ΔQ.θ = ((n̂_v > 0) ? 1 : -1) * (n̂_v⁻ * θ⁻ - n̂_v⁺ * θ⁺)
  # ΔQ.θ = abs(n̂_v⁻) * θ⁻ - abs(n̂_v⁺) * θ⁺
  =#

  return nothing
end

@inline function flux_diffusive!(m::HBModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
  ν = Diagonal(@SVector [m.νʰ, m.νʰ, m.νᶻ])
  F.u -= ν * D.∇u

  κ = Diagonal(@SVector [m.κʰ, m.κʰ, A.κᶻ])
  F.θ -= κ * D.∇θ

  return nothing
end

@inline function gradvariables!(m::HBModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

@inline function diffusive!(m::HBModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end

@inline function source!(m::HBModel{P}, source::Vars, Q::Vars, A::Vars,
                         t::Real) where P
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    f = A.f
    wz0 = A.wz0

    # f × u
    source.u -= @SVector [-f * u[2], f * u[1]]

    source.η += wz0
  end

  return nothing
end

@inline function integrate_aux!(m::HBModel, integrand::Vars, Q::Vars, A::Vars)
  αᵀ = m.αᵀ
  integrand.αᵀθ = -αᵀ * Q.θ
  integrand.∇hu = A.w # borrow the w value from A...

  return nothing
end

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

function update_aux_diffusive!(dg::DGModel, m::HBModel, Q::MPIStateArray, t::Real)
  A  = dg.auxstate

  # store ∇ʰu as integrand for w
  # update vertical diffusivity for convective adjustment
  function f!(::HBModel, Q, A, D, t)
    @inbounds begin
      A.w = -(D.∇u[1,1] + D.∇u[2,2])

      # κʰ = m.κʰ
      # A.κ = @SMatrix [κʰ -0 -0; -0 κʰ -0; -0 -0 κᶻ]
      D.∇θ[3] < 0 ? A.κᶻ = 1000 * m.κᶻ : A.κᶻ = m.κᶻ
    end

    return nothing
  end
  nodal_update_aux!(f!, dg, m, Q, t; diffusive=true)

  # compute integrals for w and pkin
  indefinite_stack_integral!(dg, m, Q, A, t) # bottom -> top
  reverse_indefinite_stack_integral!(dg, m, A, t) # top -> bottom

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  # A[1] = w, A[5] = wz0
  copy_stack_field_down!(dg, m, A, 1, 5)

  return true
end

function ocean_init_aux! end
function init_aux!(m::HBModel, A::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m, m.problem, A, geom)
end

function ocean_init_state! end
function init_state!(m::HBModel, Q::Vars, A::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, A, coords, t)
end

@inline function boundary_state!(nf, m::HBModel, Q⁺::Vars, A⁺::Vars, n⁻,
                                 Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(m, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end
@inline function boundary_state!(nf, m::HBModel,
                                 Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻,
                                 Q⁻::Vars, D⁻::Vars, A⁻::Vars,
                                 bctype, t, _...)
  return ocean_boundary_state!(m, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip,
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = -D⁻.∇u

  D⁺.∇θ = -D⁻.∇θ

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  Q⁺.u = -Q⁻.u

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  Q⁺.u = -Q⁻.u

  D⁺.∇θ = -D⁻.∇θ

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  A⁺.w = -A⁻.w

  return nothing
end


@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  A⁺.w = -A⁻.w
  D⁺.∇u = -D⁻.∇u

  D⁺.∇θ = -D⁻.∇θ

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  Q⁺.u = -Q⁻.u
  A⁺.w = -A⁻.w

  return nothing
end


@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)

  Q⁺.u = -Q⁻.u
  A⁺.w = -A⁻.w

  D⁺.∇θ = -D⁻.∇θ

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::Union{
                                       OceanSurfaceNoStressNoForcing,
                                       OceanSurfaceStressNoForcing,
                                       OceanSurfaceNoStressForcing,
                                       OceanSurfaceStressForcing},
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceNoStressNoForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = -D⁻.∇u

  D⁺.∇θ = -D⁻.∇θ

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceStressNoForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  τ = A⁻.τ
  D⁺.∇u = -D⁻.∇u + 2 * @SMatrix [ -0 -0;
                                  -0 -0;
                                  τ / 1000 -0]

  D⁺.∇θ = -D⁻.∇θ

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceNoStressForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = -D⁻.∇u

  θ  = Q⁻.θ
  θʳ = A⁻.θʳ
  λʳ = m.problem.λʳ
  D⁺.∇θ = -D⁻.∇θ + 2 * @SVector [-0, -0, λʳ * (θʳ - θ)]

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceStressForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  τ = A⁻.τ
  D⁺.∇u = -D⁻.∇u + 2 * @SMatrix [ -0 -0;
                                  -0 -0;
                                  τ / 1000 -0]

  θ  = Q⁻.θ
  θʳ = A⁻.θʳ
  λʳ = m.problem.λʳ
  D⁺.∇θ = -D⁻.∇θ + 2 * @SVector [-0, -0, λʳ * (θʳ - θ)]

  return nothing
end

end
