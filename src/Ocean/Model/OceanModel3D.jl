module Ocean3D

export HydrostaticBoussinesqModel, HydrostaticBoussinesqProblem

using StaticArrays
using ..VariableTemplates
using LinearAlgebra: I, dot, Diagonal
using ..DGmethods: init_ode_state
using ..PlanetParameters: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids: polynomialorder

using ..DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
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
struct NoFlux_NoSlip <: OceanBoundaryCondition end
struct FreeSlip <: OceanBoundaryCondition end

abstract type HydrostaticBoussinesqProblem end

struct HydrostaticBoussinesqModel{P,T} <: BalanceLaw
  problem::P
  c1::T
  c2::T
  c3::T
  αT::T
  λ_relax::T
  νh::T
  νz::T
  κh::T
  κz::T
end
HBModel = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

struct HBVerticalSupplementModel <: BalanceLaw end

function init_ode_param(dg::DGModel, m::HydrostaticBoussinesqModel)
  vert_dg = DGModel(dg, HBVerticalSupplementModel())
  vert_param = init_ode_param(vert_dg)
  vert_dQ = init_ode_state(vert_dg, 948)
  vert_filter = CutoffFilter(dg.grid, polynomialorder(dg.grid)-1)
  exp_filter = ExponentialFilter(dg.grid, 1, 32)

  return (vert_dg = vert_dg, vert_param = vert_param, vert_dQ = vert_dQ,
          vert_filter = vert_filter, exp_filter=exp_filter)
end

# If this order is changed check the filter usage!
function vars_state(m::Union{HBModel, HBVerticalSupplementModel}, T)
  @vars begin
    u::SVector{2, T}
    η::T # real a 2-D variable TODO: store as 2-D not 3-D?
    θ::T
  end
end

# If this order is changed check  update_aux!
function vars_aux(m::HBModel, T)
  @vars begin
    w::T
    pkin_reverse::T # ∫(-αT θ) # TODO: remove me after better integral interface
    w_reverse::T # TODO: remove me after better integral interface
    pkin::T # ∫(-αT θ)
    wz0::T # w at z=0
    SST_relax::T # TODO: Should be 2D
    f::T
    τ_wind::T # TODO: Should be 2D
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
    αTθ::T
  end
end

@inline function flux_nondiffusive!(m::HBModel, flux::Grad, state::Vars,
                                    aux::Vars, t::Real)
  @inbounds begin
    u = state.u # Horizontal components of velocity
    η = state.η
    θ = state.θ
    w = aux.w   # vertical velocity
    pkin = aux.pkin
    v = @SVector [u[1], u[2], w]
    Ih = @SMatrix [ 1 -0;
                   -0  1;
                   -0 -0]

    # ∇ • (u θ)
    flux.θ += v * θ

    # ∇h • (g η)
    flux.u += grav * η * Ih

    # ∇h • (- ∫(αT θ))
    flux.u += pkin * Ih

    # ∇h • (v ⊗ u)
    # flux.u += v * u'
  end

  return nothing
end

@inline function flux_diffusive!(::HBModel, flux::Grad, state::Vars, diff::Vars,
                                 aux::Vars, t::Real)
  flux.u += diff.ν∇u
  flux.θ += diff.κ∇θ
  return nothing
end

@inline function gradvariables!(m::HBModel, grad::Vars, state::Vars, aux, t)
  grad.u = state.u
  grad.θ = state.θ
  return nothing
end

@inline function diffusive!(m::HBModel, diff::Vars, grad::Grad, state::Vars,
                            aux::Vars, t)
  ν = Diagonal(@SVector [m.νh, m.νh, m.νz])
  diff.ν∇u = -ν * grad.u

  κ = Diagonal(@SVector [m.κh, m.κh, m.κz])
  diff.κ∇θ = -κ * grad.θ
  return nothing
end

@inline wavespeed(m::HBModel, n⁻, _...) = abs(SVector(m.c1, m.c2, m.c3)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
update_jump!(::Rusanov, ::HBModel, Qjump::Vars, _...) = Qjump.η = -0

@inline function source!(m::HBModel{P}, source::Vars, state::Vars, aux::Vars,
                         t::Real) where P
  @inbounds begin
    u = state.u # Horizontal components of velocity
    f = aux.f
    wz0 = aux.wz0

    # f × u
    source.u -= @SVector [-f * u[2], f * u[1]]

    source.η += wz0
  end

  return nothing
end

function update_aux!(dg, m::HydrostaticBoussinesqModel, Q, aux, t, params)
  # Compute DG gradient of u -> aux.w
  vert_dg = params.vert_dg
  vert_param = params.vert_param
  vert_dQ = params.vert_dQ
  vert_filter = params.vert_filter
  apply!(Q, (1, 2), dg.grid, vert_filter; horizontal=false)

  exp_filter = params.exp_filter
  apply!(Q, (4,), dg.grid, exp_filter; horizontal=false)

  vert_dg(vert_dQ, Q, vert_param, t; increment = false)

  # Copy from vert_dQ.η which is realy ∇h•u into aux.w (which will be
  # integrated)
  function f!(::HBModel, vert_dQ, aux, t)
    aux.w = vert_dQ.θ
  end
  nodal_update_aux!(f!, dg, m, vert_dQ, aux, t)

  # compute integrals for w and pkin
  indefinite_stack_integral!(dg, m, Q, aux, t) # bottom -> top
  reverse_indefinite_stack_integral!(dg, m, aux, t) # top -> bottom

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  copy_stack_field_down!(dg, m, aux, 1, 5)
end

function surface_flux!(m::HydrostaticBoussinesqModel, flux::Vars, face,
                       state::Vars, diff::Vars, aux::Vars, t)
  if face == 6
    DFloat = eltype(state)
    SST_relax = aux.SST_relax
    τ_wind = aux.τ_wind
    λ_relax = m.λ_relax
    θ = state.θ

    flux.θ -= λ_relax * (SST_relax - θ)

    flux.u -= @SVector [τ_wind / 1000, -0]
  end
end

@inline function integrate_aux!(m::HydrostaticBoussinesqModel,
                                integrand::Vars,
                                state::Vars,
                                aux::Vars)
  αT = m.αT
  integrand.αTθ = -αT * state.θ
  integrand.∇hu = aux.w # borrow the w value from aux...
end


function ocean_init_aux! end
function init_aux!(m::HBModel, aux::Vars, geom::LocalGeometry)
  ocean_init_aux!(m.problem, aux, geom)
end

function ocean_init_state! end
function init_state!(m::HBModel, state::Vars, aux::Vars, coords, t)
  ocean_init_state!(m.problem, state, aux, coords, t)
end



@inline function boundary_state!(nf, m::HBModel, state⁺::Vars, aux⁺::Vars, n⁻,
                                 state⁻::Vars, aux⁻::Vars, bctype, t, _...)
  ocean_boundary_state!(m, bctype, nf, state⁺, aux⁺, n⁻, state⁻, aux⁻, t)
end
@inline function boundary_state!(nf, m::HBModel,
                                 state⁺::Vars, diff⁺::Vars, aux⁺::Vars,
                                 n⁻,
                                 state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
                                 bctype, t, _...)
  ocean_boundary_state!(m, bctype, nf, state⁺, diff⁺, aux⁺, n⁻, state⁻, diff⁻,
                        aux⁻, t)
end

@inline function ocean_boundary_state!(::HBModel, ::NoFlux_NoSlip,
                                       ::Union{Rusanov, CentralGradPenalty},
                                       state⁺, aux⁺, n⁻, state⁻, aux⁻, t)

  state⁺.η = state⁻.η
  state⁺.θ = state⁻.θ
  state⁺.u = -state⁻.u

  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::NoFlux_NoSlip,
                                       ::CentralNumericalFluxDiffusive, state⁺,
                                       diff⁺, aux⁺, n⁻, state⁻, diff⁻, aux⁻, t)

  state⁺.η = state⁻.η
  state⁺.θ = state⁻.θ
  state⁺.u = -state⁻.u
  diff⁺.ν∇u = diff⁻.ν∇u
  diff⁺.κ∇θ = -diff⁻.κ∇θ

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::FreeSlip,
                                       ::Union{Rusanov, CentralGradPenalty},
                                       state⁺, aux⁺, n⁻, state⁻, aux⁻, t)

  state⁺.η = state⁻.η
  state⁺.θ = state⁻.θ
  # TODO: in current setup this doesn't matter (check later) b/c vertical
  state⁺.u = state⁻.u
  # aux⁺.w = -aux⁻.w

  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::FreeSlip,
                                       ::CentralNumericalFluxDiffusive, state⁺,
                                       diff⁺, aux⁺, n⁻, state⁻, diff⁻, aux⁻, t)

  state⁺.η = state⁻.η
  state⁺.θ = state⁻.θ
  # TODO: in current setup this doesn't matter (check later) b/c vertical
  state⁺.u = state⁻.u
  # aux⁺.w = -aux⁻.w
  diff⁺.ν∇u = diff⁻.ν∇u
  diff⁺.κ∇θ = -diff⁻.κ∇θ

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
@inline function flux_nondiffusive!(m::HBVerticalSupplementModel, flux::Grad,
                                    state::Vars, aux::Vars, t::Real)
  @inbounds begin
    u = state.u # Horizontal components of velocity
    v = @SVector [u[1], u[2], -0]

    # ∇ • (v)
    # Just using θ to store w = ∇h • u
    flux.θ += v
  end

  return nothing
end


# This is zero because when taking the horizontal gradient we're piggy-backing
# on θ and want to ensure we do not use it's jump
@inline wavespeed(m::HBVerticalSupplementModel, n⁻, _...) = -zero(eltype(n⁻))

boundary_state!(::CentralNumericalFluxDiffusive, m::HBVerticalSupplementModel,
                _...) = nothing

@inline function boundary_state!(::Rusanov, ::HBVerticalSupplementModel,
                                 state⁺, aux⁺, n⁻, state⁻, aux⁻, t, _...)

  state⁺.η = state⁻.η
  state⁺.θ = state⁻.θ
  state⁺.u = -state⁻.u

  return nothing
end

end
