module ShallowWater

export SWModel, SWProblem

using StaticArrays
using ..VariableTemplates
using LinearAlgebra: I, dot
using ..PlanetParameters: grav

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, vars_integrals, flux_nondiffusive!,
                        flux_diffusive!, source!, wavespeed,
                        boundary_state!,
                        gradvariables!, init_aux!, init_state!,
                        LocalGeometry, diffusive!

using ..DGmethods.NumericalFluxes

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type SWProblem end

abstract type TurbulenceClosure end
struct LinearDrag{L} <: TurbulenceClosure
  λ::L
end
struct ConstantViscosity{L} <: TurbulenceClosure
  ν::L
end

abstract type AdvectionTerm end
struct NonLinearAdvection <: AdvectionTerm end

struct SWModel{P, T, A, S} <: BalanceLaw
  problem::P
  turbulence::T
  advection::A
  c::S
end

function vars_state(m::SWModel, T)
  @vars begin
    U::SVector{3, T}
    η::T
  end
end

function vars_aux(m::SWModel, T)
  @vars begin
    f::SVector{3, T}
    τ::SVector{3, T}  # value includes τₒ, g, and ρ
  end
end

function vars_gradient(m::SWModel, T)
  @vars begin
    U::SVector{3, T}
  end
end

function vars_diffusive(m::SWModel, T)
  @vars begin
    ν∇U::SMatrix{3, 3, T, 9}
  end
end

@inline function flux_nondiffusive!(m::SWModel, F::Grad, q::Vars,
                                    α::Vars, t::Real)
  U = q.U
  η = q.η
  H = m.problem.H

  F.η += U
  F.U += grav * H * η * I

  advective_flux!(m, m.advection, F, q, α, t)

  return nothing
end

advective_flux!(::SWModel, ::Nothing, _...) = nothing

@inline function advective_flux!(m::SWModel, A::NonLinearAdvection, F::Grad,
                                 q::Vars, α::Vars, t::Real)
  U = q.U
  H = m.problem.H

  F.U += 1 / H * U ⊗ U

  return nothing
end

function gradvariables!(m::SWModel, f::Vars, q::Vars, α::Vars, t::Real)
  gradvariables!(m.turbulence, f, q, α, t)
end

gradvariables!(::LinearDrag, _...) = nothing

@inline function gradvariables!(T::ConstantViscosity, G::Vars, q::Vars,
                                α::Vars, t::Real)
  G.U = q.U

  return nothing
end

function diffusive!(m::SWModel, σ::Vars, ∇G::Grad, q::Vars, α::Vars, t::Real)
  diffusive!(m.turbulence, σ, ∇G, q, α, t)
end

diffusive!(::LinearDrag, _...) = nothing

@inline function diffusive!(T::ConstantViscosity, σ::Vars, ∇G::Grad, q::Vars,
                            α::Vars, t::Real)
  ν  = T.ν
  ∇U = ∇G.U

  σ.ν∇U = ν * ∇U

  return nothing
end

function flux_diffusive!(m::SWModel, G::Grad, q::Vars, σ::Vars,
                         α::Vars, t::Real)
  flux_diffusive!(m.turbulence, G, q, σ, α, t)
end

flux_diffusive!(::LinearDrag, _...) = nothing

@inline function flux_diffusive!(::ConstantViscosity, G::Grad, q::Vars,
                                 σ::Vars, α::Vars, t::Real)
  G.U -= σ.ν∇U

  return nothing
end

@inline wavespeed(m::SWModel, n⁻, q::Vars, α::Vars, t::Real) = m.c

@inline function source!(m::SWModel{P}, S::Vars, q::Vars, α::Vars,
                         t::Real) where P
  τ = α.τ
  f = α.f
  U = q.U
  S.U += τ - f × U

  linear_drag!(m.turbulence, S, q, α, t)

  return nothing
end

linear_drag!(::ConstantViscosity, _...) = nothing

@inline function linear_drag!(T::LinearDrag, S::Vars, q::Vars, α::Vars, t::Real)
  λ = T.λ
  U = q.U

  S.U -= λ * U

  return nothing
end

function shallow_init_aux! end
function init_aux!(m::SWModel, α::Vars, geom::LocalGeometry)
  return shallow_init_aux!(m.problem, α, geom)
end

function shallow_init_state! end
function init_state!(m::SWModel, q::Vars, α::Vars, coords, t)
  return shallow_init_state!(m.problem, m.turbulence, q, α, coords, t)
end

function boundary_state!(nf, m::SWModel, q⁺::Vars, α⁺::Vars, n⁻,
                         q⁻::Vars, α⁻::Vars, bctype, t, _...)
  return shallow_boundary_state!(nf, m, m.turbulence, q⁺, α⁺, n⁻, q⁻, α⁻, t)
end

@inline function shallow_boundary_state!(::Rusanov, m::SWModel, ::LinearDrag,
                                         q⁺, α⁺, n⁻, q⁻, α⁻, t)
  U⁻ = q⁻.U
  n⁻ = SVector(n⁻)

  q⁺.η = q⁻.η
  q⁺.U = U⁻ - 2 * (n⁻⋅U⁻) * n⁻

  return nothing
end

shallow_boundary_state!(::CentralGradPenalty, m::SWModel,
                        ::LinearDrag, _...) = nothing

shallow_boundary_state!(::CentralNumericalFluxDiffusive, m::SWModel,
                        ::LinearDrag, _...) = nothing

function boundary_state!(nf, m::SWModel, q⁺::Vars, σ⁺::Vars, α⁺::Vars, n⁻,
                         q⁻::Vars, σ⁻::Vars, α⁻::Vars, bctype, t, _...)
  return shallow_boundary_state!(nf, m, m.turbulence,
                                 q⁺, σ⁺, α⁺, n⁻, q⁻, σ⁻, α⁻, t)
end

@inline function shallow_boundary_state!(::Rusanov, m::SWModel,
                                         ::ConstantViscosity,
                                         q⁺, α⁺, n⁻, q⁻, α⁻, t)
  q⁺.η =  q⁻.η
  q⁺.U = -q⁻.U

  return nothing
end

@inline function shallow_boundary_state!(::CentralGradPenalty, m::SWModel,
                                         ::ConstantViscosity,
                                         q⁺, α⁺, n⁻, q⁻, α⁻, t)
  q⁺.U = -q⁻.U

  return nothing
end

@inline function shallow_boundary_state!(::CentralNumericalFluxDiffusive,
                                         m::SWModel, ::ConstantViscosity,
                                         q⁺, σ⁺, α⁺, n⁻, q⁻, σ⁻, α⁻, t)
  q⁺.U   = -q⁻.U
  σ⁺.ν∇U =  σ⁻.ν∇U

  return nothing
end

end
