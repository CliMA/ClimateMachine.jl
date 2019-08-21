using CLIMA.VariableTemplates
using StaticArrays
using CLIMA.PlanetParameters: grav, Omega, cv_d, T_0
using GPUifyLoops: @unroll
using LinearAlgebra: norm

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux!, source!, wavespeed,
                        boundarycondition!, gradvariables!, diffusive!,
                        init_aux!, init_state!, init_ode_param, init_ode_state

abstract type EulerProblem end

abstract type AbstractEulerModel <: BalanceLaw end

init_state!(m::AbstractEulerModel, x...) = initial_condition!(m, m.problem, x...)
vars_gradient(::AbstractEulerModel, _) = Tuple{}
vars_diffusive(::AbstractEulerModel, _) = Tuple{}
gradvariables!(::AbstractEulerModel, _...) = nothing
diffusive!(::AbstractEulerModel, _...) = nothing

struct EulerModel{P, G} <: AbstractEulerModel
  problem::P
  gravity::G
end
function EulerModel(problem)
  gravity = gravitymodel(problem)
  EulerModel{typeof(problem), typeof(gravity)}(problem, gravity)
end

function vars_state(::EulerModel, DFloat)
  @vars begin
    ρ::DFloat
    ρu⃗::SVector{3, DFloat}
    ρe::DFloat
  end
end

function vars_aux(m::EulerModel, DFloat)
  @vars begin
    gravity::vars_aux(m.gravity, DFloat)
  end
end

const γ_exact = 7 // 5

function pressure(ρ, ρinv, ρe, ρu⃗, ϕ)
  DFloat = eltype(ρu⃗)
  γ = DFloat(γ_exact)
  (γ - 1) * (ρe + ρ * cv_d * T_0 - ρinv * ρu⃗' * ρu⃗ / 2 - ϕ * ρ)
end

function nofluxbc!(scalars, stateP, nM, stateM)
    ## scalars are preserved
    @unroll for s in scalars
      setproperty!(stateP, s, getproperty(stateM, s))
    end

    ## reflect velocities
    ρu⃗M = stateM.ρu⃗
    n⃗ = SVector(nM)
    n⃗_ρu⃗M = n⃗' * ρu⃗M
    stateP.ρu⃗ = ρu⃗M - 2n⃗_ρu⃗M * n⃗
end

function boundarycondition!(::EulerModel, stateP::Vars, _, auxP::Vars, normalM,
                            stateM::Vars, _, auxM::Vars, bctype, t)
  if bctype == 1
    nofluxbc!((:ρ, :ρe), stateP, normalM, stateM)
  else
    error("unknown boundary condition type!")
  end
end

function flux!(m::EulerModel, flux::Grad, state::Vars, _::Vars, aux::Vars,
               t::Real)

  (ρ, ρu⃗, ρe) = (state.ρ, state.ρu⃗, state.ρe)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  ϕ = geopotential(m.gravity, aux)
  p = pressure(ρ, ρinv, ρe, ρu⃗, ϕ)

  # compute the flux!
  flux.ρ  = ρu⃗
  flux.ρu⃗ = ρu⃗ .* u⃗' + p * I
  flux.ρe = u⃗ * (ρe + p)
end

function source!(m::EulerModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρ = 0
  source.ρu⃗ = @SVector zeros(eltype(source.ρu⃗), 3)
  source.ρe = 0
  geopotential_source!(m.gravity, state.ρ, source, state, aux)
end

function init_aux!(m::EulerModel, aux::Vars, x⃗)
  init_aux!(m.gravity, aux, x⃗)
end

function wavespeed_euler(ϕ, ρ, ρu⃗, ρe, nM)
  DFloat = eltype(ρu⃗)
  γ = DFloat(γ_exact)
  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  p = pressure(ρ, ρinv, ρe, ρu⃗, ϕ)
  n⃗ = SVector{3, DFloat}(nM)
  abs(n⃗' * u⃗) + sqrt(ρinv * γ * p)
end

function wavespeed(m::EulerModel, nM, state::Vars, aux::Vars, t::Real)
  (ρ, ρu⃗, ρe) = (state.ρ, state.ρu⃗, state.ρe)
  ϕ = geopotential(m.gravity, aux)
  wavespeed_euler(ϕ, ρ, ρu⃗, ρe, nM)
end

abstract type GravityModel end
vars_aux(m::GravityModel, DFloat) = @vars(ϕ::DFloat, ∇ϕ::SVector{3, DFloat})
geopotential(::GravityModel, aux) = aux.gravity.ϕ
function geopotential_source!(::GravityModel, ρ, source, state, aux)
  source.ρu⃗ -= ρ * aux.gravity.∇ϕ
end

struct NoGravity <: GravityModel end
vars_aux(m::NoGravity, _) = @vars()
init_aux!(::NoGravity, _...) = nothing
geopotential(::NoGravity, _...) = 0
geopotential_source!(::NoGravity, _...) = nothing

struct SphereGravity{DFloat} <: GravityModel
  h::DFloat
end
function init_aux!(g::SphereGravity, aux, x⃗)
  x⃗ = SVector(x⃗)
  r = norm(x⃗, 2)
  aux.gravity.ϕ = grav * (r-g.h)
  DFloat = eltype(aux.gravity.∇ϕ)
  aux.gravity.∇ϕ = DFloat(grav) * x⃗ / r
end

struct BoxGravity{dim} <: GravityModel end
function init_aux!(::BoxGravity{dim}, aux, x⃗) where dim
  @inbounds aux.gravity.ϕ = grav * x⃗[dim]

  DFloat = eltype(aux.gravity.∇ϕ)
  if dim == 2
    aux.gravity.∇ϕ = SVector{3, DFloat}(0, grav, 0)
  else
    aux.gravity.∇ϕ = SVector{3, DFloat}(0, 0, grav)
  end
end

struct NoGravity <: GravityModel end
vars_aux(m::NoGravity, _) = @vars()
init_aux!(::NoGravity, _...) = nothing
geopotential(::NoGravity, _...) = 0
geopotential_source!(::NoGravity, _...) = nothing

struct ReferenceStateEulerModel{P, C, G} <: AbstractEulerModel
  pde_level_hydrostatic_balance::Bool
  problem::P
  coriolis::C
  gravity::G
end

function ReferenceStateEulerModel(balanced, problem, coriolis)
  gravity = gravitymodel(problem)
  args = problem, coriolis, gravity
  ReferenceStateEulerModel{typeof.(args)...}(balanced, args...)
end

function vars_state(::ReferenceStateEulerModel, DFloat)
  @vars begin
    δρ::DFloat
    ρu⃗::SVector{3, DFloat}
    δρe::DFloat
  end
end

function vars_aux(m::ReferenceStateEulerModel, DFloat)
  @vars begin
    coord::SVector{3, DFloat}
    ρ_ref::DFloat
    ρe_ref::DFloat
    gravity::vars_aux(m.gravity, DFloat)
  end
end

function init_aux!(m::ReferenceStateEulerModel, aux::Vars, x⃗)
  aux.coord = SVector(x⃗)
  init_aux!(m.gravity, aux, x⃗)
  reference_state!(m, m.problem, aux, x⃗)
end

function wavespeed(m::ReferenceStateEulerModel, nM, state::Vars, aux::Vars,
                   t::Real)
  δρ, ρu⃗, δρe = state.δρ, state.ρu⃗, state.δρe
  ρ_ref, ρe_ref = aux.ρ_ref, aux.ρe_ref
  ρ, ρe = ρ_ref + δρ, ρe_ref + δρe
  ϕ = geopotential(m.gravity, aux)
  wavespeed_euler(ϕ, ρ, ρu⃗, ρe, nM)
end

function flux!(m::ReferenceStateEulerModel, flux::Grad, state::Vars, _::Vars,
               aux::Vars, t::Real)

  δρ, ρu⃗, δρe = state.δρ, state.ρu⃗, state.δρe
  ρ_ref, ρe_ref = aux.ρ_ref, aux.ρe_ref
  ρ, ρe = ρ_ref + δρ, ρe_ref + δρe

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  ϕ = geopotential(m.gravity, aux)
  p = pressure(ρ, ρinv, ρe, ρu⃗, ϕ)

  p_ρu⃗ = p
  if m.pde_level_hydrostatic_balance
    ρinv_ref = 1 / ρ_ref
    u⃗_ref = @SVector zeros(eltype(ρu⃗), 3)
    p_ref = pressure(ρ_ref, ρinv_ref, ρe_ref, u⃗_ref, ϕ)
    p_ρu⃗ -= p_ref
  end

  # compute the flux!
  flux.δρ = ρu⃗
  flux.ρu⃗ = ρu⃗ .* u⃗' + p_ρu⃗ * I
  flux.δρe = u⃗ * (ρe + p)
end

function source!(m::ReferenceStateEulerModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.δρ = 0
  source.ρu⃗ = @SVector zeros(eltype(source.ρu⃗), 3)
  source.δρe = 0
  if m.pde_level_hydrostatic_balance
    geopotential_source!(m.gravity, state.δρ, source, state, aux)
  else
    ρ = aux.ρ_ref + state.δρ
    geopotential_source!(m.gravity, ρ, source, state, aux)
  end
  coriolis_source!(m.coriolis, source, state, aux)
  
  parametrized_source!(m, m.problem, source, state, aux)
end
parametrized_source!(::ReferenceStateEulerModel, ::EulerProblem, _...) = nothing

abstract type CoriolisModel end
vars_aux(m::CoriolisModel, _) = @vars()
function coriolis_source!(::CoriolisModel, source, state, aux)
   Ω⃗ = SVector(0, 0, Omega)
   source.ρu⃗ += -2Ω⃗ × state.ρu⃗
end
struct Coriolis <: CoriolisModel end

struct NoCoriolis <: CoriolisModel end
coriolis_source!(::NoCoriolis, _...) = nothing

function boundarycondition!(::ReferenceStateEulerModel, stateP::Vars, _, auxP::Vars, normalM,
                            stateM::Vars, _, auxM::Vars, bctype, t)
  if bctype == 1
    nofluxbc!((:δρ, :δρe), stateP, normalM, stateM)
  else
    error("unknown boundary condition type!")
  end
end
