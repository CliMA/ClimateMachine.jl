module Ocean2D

export HB2DModel, HB2DProblem

using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using ..VariableTemplates
using ..MPIStateArrays
using ..PlanetParameters: grav

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, vars_integrals, flux_nondiffusive!,
                        flux_diffusive!, source!, wavespeed,
                        boundary_state!, DGModel,
                        gradvariables!, init_aux!, init_state!,
                        LocalGeometry, diffusive!

using ..DGmethods.NumericalFluxes: Rusanov, CentralFlux, CentralGradPenalty,
                                   CentralNumericalFluxDiffusive

import ..DGmethods.NumericalFluxes: update_penalty!, numerical_flux_diffusive!,
                                    NumericalFluxNonDiffusive

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type HB2DProblem end

struct HB2DModel{P, S} <: BalanceLaw
  problem::P
  cʰ::S
  cᵛ::S
  κʰ::S
  κᵛ::S
end

function vars_state(m::HB2DModel, T)
  @vars begin
    θ::T
  end
end

function vars_aux(m::HB2DModel, T)
  @vars begin
    u::SVector{3, T}
  end
end

function vars_gradient(m::HB2DModel, T)
  @vars begin
    θ::T
  end
end

function vars_diffusive(m::HB2DModel, T)
  @vars begin
    κ∇θ::SVector{3, T}
  end
end

@inline function flux_nondiffusive!(m::HB2DModel, F::Grad, Q::Vars,
                                    α::Vars, t::Real)
  θ = Q.θ
  u = α.u

  F.θ += u * θ

  return nothing
end

@inline wavespeed(m::HB2DModel, n⁻, _...) = abs(SVector(m.cʰ, m.cᵛ, -0)' * n⁻)

function update_penalty!(::Rusanov, ::HB2DModel, ΔQ::Vars,
                         n⁻, λ, Q⁻, Q⁺, α⁻, α⁺, t)
  θ⁻ = Q⁻.θ
  u⁻ = α⁻.u
  n̂_u⁻ = n⁻∘u⁻

  θ⁺ = Q⁺.θ
  u⁺ = α⁺.u
  n̂_u⁺ = n⁻∘u⁺

  n̂_u = (n̂_u⁻ + n̂_u⁺) / 2

  ΔQ.θ = ((n̂_u > 0) ? 1 : -1) * (n̂_u⁻ * θ⁻ - n̂_u⁺ * θ⁺)

  return nothing
end

@inline function flux_diffusive!(m::HB2DModel, F::Grad, Q::Vars, σ::Vars,
                                 α::Vars, t::Real)
  F.θ += σ.κ∇θ

  return nothing
end

@inline function gradvariables!(m::HB2DModel, G::Vars, Q::Vars,
                                α::Vars, t::Real)
  G.θ = Q.θ

  return nothing
end

@inline function diffusive!(m::HB2DModel, σ::Vars, ∇G::Grad, Q::Vars,
                            α::Vars, t::Real)
  κ = Diagonal(@SVector [m.κʰ, m.κᵛ, -0])
  σ.κ∇θ = -κ * ∇G.θ
end

source!(m::HB2DModel, S::Vars, Q::Vars, α::Vars,
                         t::Real) = nothing

function init_ode_param(dg::DGModel, m::HB2DModel)
  # filter = CutoffFilter(dg.grid, div(polynomialorder(dg.grid),2))
  filter = ExponentialFilter(dg.grid, 1, 16)

  return (filter = filter)
end

function update_aux!(dg::DGModel, m::HB2DModel, Q::MPIStateArray,
                     α::MPIStateArray, t, params)
  apply!(Q, (1, ), dg.grid, params.filter)
  apply!(α, (1,2), dg.grid, params.filter)

  return nothing
end

function hb2d_init_aux! end
function init_aux!(m::HB2DModel, α::Vars, geom::LocalGeometry)
  return hb2d_init_aux!(m.problem, α, geom)
end

function hb2d_init_state! end
function init_state!(m::HB2DModel, Q::Vars, α::Vars, coords, t)
  return hb2d_init_state!(m.problem, Q, α, coords, t)
end

@inline function boundary_state!(nf, m::HB2DModel, Q⁺::Vars, α⁺::Vars, n⁻,
                                 Q⁻::Vars, α⁻::Vars, bctype, t, _...)
  return hb2d_boundary_state!(m, Q⁺, α⁺, n⁻, Q⁻, α⁻, t)
end

@inline function boundary_state!(nf, m::HB2DModel,
                                 Q⁺::Vars, σ⁺::Vars, α⁺::Vars,
                                 n⁻,
                                 Q⁻::Vars, σ⁻::Vars, α⁻::Vars,
                                 bctype, t, _...)
  return hb2d_boundary_state!(m, Q⁺, σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)
end


@inline function hb2d_boundary_state!(m::HB2DModel, Q⁺, α⁺, n⁻, Q⁻, α⁻, t)
  if m.κʰ == 0 && m.κᵛ == 0
    Q⁺.θ = -Q⁻.θ
  end

  return nothing
end

@inline function hb2d_boundary_state!(m::HB2DModel, Q⁺, σ⁺, α⁺, n⁻, Q⁻, σ⁻, α⁻, t)
  if m.κʰ == 0 && m.κᵛ == 0
    Q⁺.θ = -Q⁻.θ
  else
    σ⁺.κ∇θ = -σ⁻.κ∇θ
  end
  
  return nothing
end

end
