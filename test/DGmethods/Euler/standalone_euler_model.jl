using CLIMA.VariableTemplates
using StaticArrays

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux!, source!, wavespeed,
                        boundarycondition!, gradvariables!, diffusive!,
                        init_aux!, init_state!, init_ode_param, init_ode_state

abstract type EulerProblem end

struct EulerModel{P} <: BalanceLaw
  problem::P
  function EulerModel(problem)
    P = typeof(problem)
    new{P}(problem)
  end
end

init_state!(m::EulerModel, x...) = initial_condition!(m, m.problem, x...)

function vars_state(::EulerModel, T)
  NamedTuple{(:ρ, :ρu⃗, :ρe), Tuple{T, SVector{3, T}, T}}
end

vars_aux(m::EulerModel, T) = Tuple{}
vars_gradient(::EulerModel, T) = Tuple{}
vars_diffusive(::EulerModel, T) = Tuple{}

const γ_exact = 7 // 5

function pressure(ρ::T, ρinv, ρe, ρu⃗, ϕ) where T
  γ::T = γ_exact
  (γ - 1) * (ρe - ρinv * ρu⃗' * ρu⃗ / 2 - ϕ * ρ)
end

function flux!(m::EulerModel, flux::Grad, state::Vars, _::Vars, aux::Vars,
               t::Real)

  (ρ, ρu⃗, ρe) = (state.ρ, state.ρu⃗, state.ρe)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  ϕ = -zero(eltype(ρ)) # FIXME: need gravity model
  p = pressure(ρ, ρinv, ρe, ρu⃗, ϕ)

  # compute the flux!
  flux.ρ  = ρu⃗
  flux.ρu⃗ = ρu⃗ .* u⃗' + p * I
  flux.ρe = u⃗ * (ρe + p)
end

gradvariables!(::EulerModel, _...) = nothing
diffusive!(::EulerModel, _...) = nothing

function source!(m::EulerModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρ = 0
  source.ρu⃗ = @SVector zeros(eltype(source.ρu⃗), 3)
  source.ρe = 0
end

init_aux!(m::EulerModel, aux::Vars, (x1, x2, x3)) = nothing

function wavespeed(m::EulerModel, nM, state::Vars, aux::Vars, t::Real)
  T = eltype(state)
  γ = T(γ_exact)

  (ρ, ρu⃗, ρe) = (state.ρ, state.ρu⃗, state.ρe)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  ϕ = -zero(eltype(ρ)) # FIXME: need gravity model
  p = pressure(ρ, ρinv, ρe, ρu⃗, ϕ)
  @inbounds n⃗ = SVector{3, T}(nM[1], nM[2], nM[3])
  abs(n⃗' * u⃗) + sqrt(ρinv * γ * p)
end

boundarycondition!(::EulerModel, _...) = nothing
