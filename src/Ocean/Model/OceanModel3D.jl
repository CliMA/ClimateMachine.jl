module Ocean3D

export HydrostaticBoussinesqModel, HydrostaticBoussinesqProblem

using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using ..VariableTemplates
using ..MPIStateArrays
using ..DGmethods: init_ode_state
using ..PlanetParameters: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids: polynomialorder

using ..DGmethods.NumericalFluxes: Rusanov, CentralFlux, CentralGradPenalty,
                                   CentralNumericalFluxDiffusive

import ..DGmethods.NumericalFluxes: update_penalty!, numerical_flux_diffusive!,
                                    NumericalFluxNonDiffusive

import ..DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                    vars_diffusive, vars_integrals, flux_nondiffusive!,
                    flux_diffusive!, source!, wavespeed,
                    boundary_state!, update_aux!,
                    gradvariables!, init_aux!, init_state!,
                    LocalGeometry, indefinite_stack_integral!,
                    reverse_indefinite_stack_integral!, integrate_aux!,
                    init_ode_param, DGModel, nodal_update_aux!, diffusive!,
                    copy_stack_field_down!, surface_flux!, nodal_update_state!

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

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

struct HBVerticalSupplementModel <: BalanceLaw end

HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem
VSModel   = HBVerticalSupplementModel

function init_ode_param(dg::DGModel, m::HBModel)
  vert_dg     = DGModel(dg, VSModel())
  vert_param  = init_ode_param(vert_dg)
  vert_dQ     = init_ode_state(vert_dg, 948)
  vert_filter = CutoffFilter(dg.grid, polynomialorder(dg.grid)-1)
  exp_filter  = ExponentialFilter(dg.grid, 1, 32)

  return (vert_dg = vert_dg, vert_param = vert_param, vert_dQ = vert_dQ,
          vert_filter = vert_filter, exp_filter=exp_filter)
end

# If this order is changed check the filter usage!
function vars_state(m::Union{HBModel, VSModel}, T)
  @vars begin
    u::SVector{2, T}
    wₒ::T # copy of aux.w
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

    # diagnostics (should be ported elsewhere eventually)
    ∂w∂z::T
    div2D::T
    div3D::T
    θu::T
    θv::T
    θw::T
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

    F.wₒ = -0
  end

  return nothing
end

@inline function flux_diffusive!(m::HBModel, F::Grad, Q::Vars, σ::Vars,
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
function update_penalty!(::Rusanov, ::HBModel, ΔQ::Vars,
                         n⁻, λ, Q⁻, Q⁺, α⁻, α⁺, t)
  ΔQ.η = -0

  θ⁻ = Q⁻.θ
  u⁻ = Q⁻.u
  w⁻ = α⁻.w
  @inbounds v⁻ = @SVector [u⁻[1], u⁻[2], w⁻]
  n̂_v⁻ = n⁻∘v⁻

  θ⁺ = Q⁺.θ
  u⁺ = Q⁺.u
  w⁺ = α⁺.w
  @inbounds v⁺ = @SVector [u⁺[1], u⁺[2], w⁺]
  n̂_v⁺ = n⁻∘v⁺

  # max velocity
  # n̂∘v = (abs(n̂∘v⁺) > abs(n̂∘v⁻) ? n̂∘v⁺ : n̂∘v⁻

  # average velocity
  n̂_v = (n̂_v⁻ + n̂_v⁺) / 2

  ΔQ.θ = ((n̂_v > 0) ? 1 : -1) * (n̂_v⁻ * θ⁻ - n̂_v⁺ * θ⁺)
  # ΔQ.θ = abs(n̂_v⁻) * θ⁻ - abs(n̂_v⁺) * θ⁺

  return nothing
end

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

function update_aux!(dg, m::HBModel, Q::MPIStateArray,
                     α::MPIStateArray, σ::MPIStateArray, t, params)
  # Compute DG gradient of u -> α.w
  vert_dg = params.vert_dg
  vert_param = params.vert_param
  vert_dQ = params.vert_dQ

  # required to ensure that after integration velocity field is divergence free
  vert_filter = params.vert_filter
  # Q[1] = u[1] = u, Q[2] = u[2] = v
  apply!(Q, (1, 2), dg.grid, vert_filter; horizontal=false)

  exp_filter = params.exp_filter
  # Q[4] = θ
  # apply!(Q, (4,), dg.grid, exp_filter; horizontal=false)

  # calculate ∇ʰ⋅u
  vert_dg(vert_dQ, Q, vert_param, t; increment = false)

  # Copy from vert_dQ.θ which is realy ∇h•u into α.w (which will be
  # integrated)
  function f!(::HBModel, vert_dQ, α, σ, t)
    α.w  = vert_dQ.θ
    α.div2D = vert_dQ.θ

    return nothing
  end
  nodal_update_aux!(f!, dg, m, vert_dQ, α, σ, t)

  # compute integrals for w and pkin
  indefinite_stack_integral!(dg, m, Q, α, t) # bottom -> top
  reverse_indefinite_stack_integral!(dg, m, α, t) # top -> bottom

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  # α[1] = w, α[5] = wz0
  copy_stack_field_down!(dg, m, α, 1, 5)

  # copy w back to state of HBModel
  function h!(::HBModel, Q, α, σ, t)
    Q.wₒ  = α.w

    return nothing
  end
  nodal_update_state!(h!, dg, m, Q, α, σ, t)

  # get ∂w/∂z
  vert_dg(vert_dQ, Q, vert_param, t; increment = false)


  function j!(m::HBModel, vert_dQ, α, σ, t)
    α.∂w∂z = vert_dQ.η
    α.div3D = α.div2D + vert_dQ.η

    return nothing
  end
  nodal_update_aux!(j!, dg, m, vert_dQ, α, σ, t)

  #  store some diagnostic variables
  function g!(m::HBModel, Q, α, σ, t)
    @inbounds begin
      α.θu = Q.θ * Q.u[1]
      α.θv = Q.θ * Q.u[2]
      α.θw = Q.θ * α.w

      return nothing
    end
  end
  nodal_update_aux!(g!, dg, m, Q, α, σ, t)
end

surface_flux!(m::HBModel, _...) = nothing

@inline function integrate_aux!(m::HBModel,
                                integrand::Vars, Q::Vars, α::Vars)
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
  α⁺.w = -α⁻.w

  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::OceanFloor,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)

  Q⁺.u = -Q⁻.u
  α⁺.w = -α⁻.w

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

# VSModel is used to compute the horizontal divergence of u
vars_aux(::VSModel, T)  = @vars()
vars_gradient(::VSModel, T)  = @vars()
vars_diffusive(::VSModel, T)  = @vars()
vars_integrals(::VSModel, T)  = @vars()
init_aux!(::VSModel, _...) = nothing

@inline flux_diffusive!(::VSModel, _...) = nothing
@inline source!(::VSModel, _...) = nothing

# This allows the balance law framework to compute the horizontal gradient of u
# (which will be stored back in the field θ)
# now also using η to store the full 3D gradient of (u,v,w)
@inline function flux_nondiffusive!(m::VSModel, F::Grad, Q::Vars,
                                    α::Vars, t::Real)
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    v = @SVector [u[1], u[2], -0]

    # ∇ • (v)
    # Just using θ to store w = ∇h • u
    F.θ += v

    w = @SVector [-0, -0, Q.wₒ]
    # ∇ • (ṽ)
    # Just using η to store ∇•(0,0,w)
    F.η += w
  end

  return nothing
end


# This is zero because when taking the horizontal gradient we're piggy-backing
# on θ and want to ensure we do not use it's jump
@inline wavespeed(m::VSModel, n⁻, _...) = -zero(eltype(n⁻))

boundary_state!(::CentralNumericalFluxDiffusive, m::VSModel, _...) = nothing

@inline function boundary_state!(::Union{Rusanov, CentralFlux}, ::VSModel,
                                 Q⁺, α⁺, n⁻, Q⁻, α⁻, t, _...)
  Q⁺.u = -Q⁻.u

  return nothing
end

end
