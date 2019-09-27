module Ocean3D

export HydrostaticBoussinesqModel, HydrostaticBoussinesqProblem

using StaticArrays
using ..VariableTemplates
using LinearAlgebra: I, dot, Diagonal
using ..DGmethods: init_ode_state
using ..PlanetParameters: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids: polynomialorder

using ..DGmethods.NumericalFluxes: Rusanov, CentralFlux, CentralGradPenalty,
                                   CentralNumericalFluxDiffusive

import ..DGmethods.NumericalFluxes: update_jump!

import ..DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                    vars_diffusive, vars_integrals, flux_nondiffusive!,
                    flux_diffusive!, source!, wavespeed,
                    boundary_state!, update_aux!,
                    gradvariables!, init_aux!, init_state!,
                    LocalGeometry, indefinite_stack_integral!,
                    reverse_indefinite_stack_integral!, integrate_aux!,
                    init_ode_param, DGModel, nodal_update_aux!, diffusive!,
                    copy_stack_field_down!, surface_flux!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)

abstract type OceanBoundaryCondition end
struct Coastline    <: OceanBoundaryCondition end
struct OceanFloor   <: OceanBoundaryCondition end
struct OceanSurface <: OceanBoundaryCondition end

abstract type HydrostaticBoussinesqProblem end

struct HydrostaticBoussinesqModel{P,T} <: BalanceLaw
  problem::P
  c₁::T
  c₂::T
  c₃::T
  αᵀ::T
  λʳ::T
  νʰ::T
  νᶻ::T
  κʰ::T
  κᶻ::T
end
HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

struct HBVerticalSupplementModel <: BalanceLaw end

function init_ode_param(dg::DGModel, m::HydrostaticBoussinesqModel)
  vert_dg     = DGModel(dg, HBVerticalSupplementModel())
  vert_param  = init_ode_param(vert_dg)
  vert_dQ     = init_ode_state(vert_dg, 948)
  vert_filter = CutoffFilter(dg.grid, polynomialorder(dg.grid)-1)
  exp_filter  = ExponentialFilter(dg.grid, 1, 32)

  return (vert_dg = vert_dg, vert_param = vert_param, vert_dQ = vert_dQ,
          vert_filter = vert_filter, exp_filter=exp_filter)
end

# If this order is changed check the filter usage!
function vars_state(m::Union{HBModel, HBVerticalSupplementModel}, T)
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
    ν∇u::SMatrix{3, 2, T, 6}
    κ∇θ::SVector{3, T}
  end
end

function vars_integrals(m::HBModel, T)
  @vars begin
    ∇hu::T
    αᵀθ::T
  end
end

@inline function flux_nondiffusive!(m::HBModel, F::Grad, Q::Vars,
                                    α::Vars, t::Real)
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    η = Q.η
    θ = Q.θ
    w = α.w   # vertical velocity
    pkin = α.pkin
    v = @SVector [u[1], u[2], w]
    Ih = @SMatrix [ 1 -0;
                   -0  1;
                   -0 -0]

    # ∇ • (u θ)
    F.θ += v * θ

    # ∇h • (g η)
    F.u += grav * η * Ih

    # ∇h • (- ∫(αᵀ θ))
    F.u += pkin * Ih

    # ∇h • (v ⊗ u)
    # F.u += v * u'
  end

  return nothing
end

@inline function flux_diffusive!(::HBModel, F::Grad, Q::Vars, σ::Vars,
                                 α::Vars, t::Real)
  F.u += σ.ν∇u
  F.θ += σ.κ∇θ
  return nothing
end

@inline function gradvariables!(m::HBModel, G::Vars, Q::Vars, α, t)
  G.u = Q.u
  G.θ = Q.θ
  return nothing
end

@inline function diffusive!(m::HBModel, σ::Vars, G::Grad, Q::Vars,
                            α::Vars, t)
  ν = Diagonal(@SVector [m.νʰ, m.νʰ, m.νᶻ])
  σ.ν∇u = -ν * G.u

  κ = Diagonal(@SVector [m.κʰ, m.κʰ, m.κᶻ])
  σ.κ∇θ = -κ * G.θ
  return nothing
end

@inline wavespeed(m::HBModel, n⁻, _...) = abs(SVector(m.c₁, m.c₂, m.c₃)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
update_jump!(::Rusanov, ::HBModel, Qjump::Vars, _...) = Qjump.η = -0

@inline function source!(m::HBModel{P}, source::Vars, Q::Vars, α::Vars,
                         t::Real) where P
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    f = α.f
    wz0 = α.wz0

    # f × u
    source.u -= @SVector [-f * u[2], f * u[1]]

    source.η += wz0
  end

  return nothing
end

function update_aux!(dg, m::HydrostaticBoussinesqModel, Q, α, t, params)
  # Compute DG gradient of u -> α.w
  vert_dg = params.vert_dg
  vert_param = params.vert_param
  vert_dQ = params.vert_dQ
  vert_filter = params.vert_filter
  # apply!(Q, (1, 2), dg.grid, vert_filter; horizontal=false)

  exp_filter = params.exp_filter
  # apply!(Q, (4,), dg.grid, exp_filter; horizontal=false)

  vert_dg(vert_dQ, Q, vert_param, t; increment = false)

  # Copy from vert_dQ.η which is realy ∇h•u into α.w (which will be
  # integrated)
  function f!(::HBModel, vert_dQ, α, t)
    α.w = vert_dQ.θ
  end
  nodal_update_aux!(f!, dg, m, vert_dQ, α, t)

  # compute integrals for w and pkin
  indefinite_stack_integral!(dg, m, Q, α, t) # bottom -> top
  reverse_indefinite_stack_integral!(dg, m, α, t) # top -> bottom

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  copy_stack_field_down!(dg, m, α, 1, 5)
end

surface_flux!(m::HydrostaticBoussinesqModel, _...) = nothing

@inline function integrate_aux!(m::HydrostaticBoussinesqModel,
                                integrand::Vars,
                                Q::Vars,
                                α::Vars)
  αᵀ = m.αᵀ
  integrand.αᵀθ = -αᵀ * Q.θ
  integrand.∇hu = α.w # borrow the w value from α...
end


function ocean_init_aux! end
function init_aux!(m::HBModel, α::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m.problem, α, geom)
end

function ocean_init_state! end
function init_state!(m::HBModel, Q::Vars, α::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, α, coords, t)
end

@inline function boundary_state!(nf, m::HBModel, Q⁺::Vars, α⁺::Vars, n⁻,
                                 Q⁻::Vars, α⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(m, bctype, nf, Q⁺, α⁺, n⁻, Q⁻, α⁻, t)
end
@inline function boundary_state!(nf, m::HBModel,
                                 Q⁺::Vars, σ⁺::Vars, α⁺::Vars,
                                 n⁻,
                                 Q⁻::Vars, σ⁻::Vars, α⁻::Vars,
                                 bctype, t, _...)
  return ocean_boundary_state!(m, bctype, nf, Q⁺, σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)
end

@inline function ocean_boundary_state!(::HBModel, ::Coastline,
                                       ::Union{Rusanov, CentralFlux, CentralGradPenalty},
                                       Q⁺, α⁺, n⁻, Q⁻, α⁻, t)
  Q⁺.u = -Q⁻.u

  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::Coastline,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)
  Q⁺.u = -Q⁻.u

  σ⁺.κ∇θ = -σ⁻.κ∇θ

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::OceanFloor,
                                       ::Union{Rusanov, CentralFlux, CentralGradPenalty},
                                       Q⁺, α⁺, n⁻, Q⁻, α⁻, t)
  Q⁺.u = -Q⁻.u
  α⁺.w =  α⁻.w

  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::OceanFloor,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)

  Q⁺.u = -Q⁻.u
  α⁺.w =  α⁻.w

  σ⁺.κ∇θ = -σ⁻.κ∇θ

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::OceanSurface,
                                       ::Union{Rusanov, CentralFlux, CentralGradPenalty},
                                       Q⁺, α⁺, n⁻, Q⁻, α⁻, t)
  α⁺.w = α⁻.w

  return nothing
end


@inline function ocean_boundary_state!(m::HBModel, ::OceanSurface,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)
  α⁺.w = α⁻.w

  τ = α⁻.τ
  σ⁺.ν∇u = -σ⁻.ν∇u - 2 * @SMatrix [ -0 -0;
                                    -0 -0;
                                    τ / 1000 -0]

  θ  = Q⁻.θ
  θʳ = α⁻.θʳ
  λʳ = m.λʳ
  σ⁺.κ∇θ = -σ⁻.κ∇θ + 2 * λʳ * (θ - θʳ)

  return nothing
end

# HBVerticalSupplementModel is used to compute the horizontal divergence of u
vars_aux(::HBVerticalSupplementModel, T)  = @vars()
vars_gradient(::HBVerticalSupplementModel, T)  = @vars()
vars_diffusive(::HBVerticalSupplementModel, T)  = @vars()
vars_integrals(::HBVerticalSupplementModel, T)  = @vars()
init_aux!(::HBVerticalSupplementModel, _...) = nothing
@inline flux_diffusive!(::HBVerticalSupplementModel, _...) = nothing
@inline source!(::HBVerticalSupplementModel, _...) = nothing

# This allows the balance law framework to compute the horizontal gradient of u
# (which will be stored back in the field θ)
@inline function flux_nondiffusive!(m::HBVerticalSupplementModel, F::Grad,
                                    Q::Vars, α::Vars, t::Real)
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    v = @SVector [u[1], u[2], -0]

    # ∇ • (v)
    # Just using θ to store w = ∇h • u
    F.θ += v
  end

  return nothing
end


# This is zero because when taking the horizontal gradient we're piggy-backing
# on θ and want to ensure we do not use it's jump
@inline wavespeed(m::HBVerticalSupplementModel, n⁻, _...) = -zero(eltype(n⁻))

boundary_state!(::CentralNumericalFluxDiffusive, m::HBVerticalSupplementModel,
                _...) = nothing

@inline function boundary_state!(::Union{Rusanov, CentralFlux},
                                 ::HBVerticalSupplementModel,
                                 Q⁺, α⁺, n⁻, Q⁻, α⁻, t, _...)
  Q⁺.u = -Q⁻.u

  return nothing
end

end
