using CLIMA.VariableTemplates
using StaticArrays
using CLIMA.PlanetParameters: grav, Omega
using CLIMA.MoistThermodynamics: PhaseDry, air_pressure, soundspeed_air
using LinearAlgebra: norm

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux!, source!, wavespeed,
                        boundarycondition!, gradvariables!, diffusive!,
                        init_aux!, init_state!, init_ode_param, init_ode_state

abstract type EulerProblem end
pde_level_referencestate_hydrostatic_balance(p::EulerProblem) = false
coriolisforce(::EulerProblem) = false
gravitymodel(::EulerProblem) = NoGravity()
referencestate(::EulerProblem) = ZeroReferenceState()

struct EulerModel{P, G, R} <: BalanceLaw
  pde_level_hydrostatic_balance::Bool
  coriolis::Bool
  problem::P
  gravity::G
  refstate::R
end
function EulerModel(problem)
  pde_level_hydrostatic_balance = pde_level_referencestate_hydrostatic_balance(problem)
  coriolis = coriolisforce(problem)
  gravity = gravitymodel(problem)
  if (pde_level_hydrostatic_balance && gravity isa NoGravity)
    @warn string("pde_level_referencestate_hydrostatic_balance ",
                 "has no effect when gravitymodel == $gravity")
  end
  refstate = referencestate(problem)
  if (pde_level_hydrostatic_balance && refstate isa ZeroReferenceState)
    @warn string("pde_level_referencestate_hydrostatic_balance ",
                 "has no effect when referencestate == $refstate")
  end
  args = (problem, gravity, refstate)
  EulerModel{typeof.(args)...}(pde_level_hydrostatic_balance, coriolis, args...)
end

function vars_state(m::EulerModel, DFloat)
  @vars begin
    δρ::DFloat
    δρu⃗::SVector{3, DFloat}
    δρe::DFloat
  end
end
function vars_aux(m::EulerModel, DFloat)
  @vars begin
    coord::SVector{3, DFloat} # FIXME: these are not always necessary
    gravity::vars_aux(m.gravity, DFloat)
    refstate::vars_aux(m.refstate, DFloat)
  end
end
vars_gradient(::EulerModel, _) = Tuple{}
vars_diffusive(::EulerModel, _) = Tuple{}

gradvariables!(::EulerModel, _...) = nothing
diffusive!(::EulerModel, _...) = nothing

init_state!(m::EulerModel, rest...) = initial_condition!(m, m.problem, rest...)
function init_aux!(m::EulerModel, aux, x⃗)
  aux.coord = SVector(x⃗)
  init_aux!(m.gravity, aux, x⃗)
  init_aux!(m.problem, m.refstate, aux, x⃗)
end

fullstate(m::EulerModel, state, aux) = fullstate(m.refstate, state, aux.refstate)
removerefstate!(m::EulerModel, state, aux) = removerefstate!(m.refstate, state, aux.refstate)

abstract type EulerReferenceState end
init_aux!(problem, m::EulerReferenceState, rest...) = referencestate!(problem, m, rest...)

struct ZeroReferenceState <: EulerReferenceState end
vars_aux(::ZeroReferenceState, _) = @vars()
init_aux!(_, ::ZeroReferenceState, _...) = nothing
fullstate(m::ZeroReferenceState, state, _) = (ρ = state.δρ, ρu⃗ = state.δρu⃗, ρe = state.δρe)
removerefstate!(::ZeroReferenceState, _...) = nothing
pressure_perturbation(::ZeroReferenceState, _, p, _) = p

struct DensityEnergyReferenceState <: EulerReferenceState end
function vars_aux(::DensityEnergyReferenceState, DFloat)
  @vars begin
    ρ::DFloat
    ρe::DFloat
  end
end
fullstate(::DensityEnergyReferenceState, state, refstate) =
  (ρ = state.δρ + refstate.ρ, ρu⃗ = state.δρu⃗, ρe = state.δρe + refstate.ρe)
function removerefstate!(::DensityEnergyReferenceState, state, refstate)
  state.δρ -= refstate.ρ
  state.δρe -= refstate.ρe
end
removerefstate!(::ZeroReferenceState, _...) = nothing
function pressure_perturbation(::DensityEnergyReferenceState, refstate, p, ϕ)
  ρ_ref = refstate.ρ 
  invρ_ref = 1 / ρ_ref
  e_ref = refstate.ρe * invρ_ref
  p - air_pressure(PhaseDry(e_ref - ϕ, ρ_ref))
end

struct FullReferenceState <: EulerReferenceState end
function vars_aux(::FullReferenceState, DFloat)
  @vars begin
    ρ::DFloat
    ρu⃗::SVector{3, DFloat}
    ρe::DFloat
  end
end
fullstate(::FullReferenceState, state, refstate) =
  (ρ = state.δρ + refstate.ρ, ρu⃗ = state.δρu⃗ + refstate.ρu⃗, ρe = state.δρe + refstate.ρe)
function removerefstate!(::FullReferenceState, state, refstate)
  state.δρ -= refstate.ρ
  state.δρu⃗ -= refstate.ρu⃗
  state.δρe -= refstate.ρe
end
function pressure_perturbation(::FullReferenceState, refstate, p, ϕ)
  ρ_ref = refstate.ρ 
  invρ_ref = 1 / ρ_ref
  e_ref = refstate.ρe * invρ_ref
  u⃗_ref = refstate.ρu⃗ * invρ_ref
  p - air_pressure(PhaseDry(e_ref - u⃗_ref' * u⃗_ref / 2 - ϕ, ρ_ref))
end

function nofluxbc!(stateP, nM, stateM)
    ## scalars are preserved
    stateP.δρ = stateM.δρ
    stateP.δρe = stateM.δρe

    ## reflect velocities
    δρu⃗M = stateM.δρu⃗
    n⃗ = SVector(nM)
    n⃗_δρu⃗M = n⃗' * δρu⃗M
    stateP.δρu⃗ = δρu⃗M - 2n⃗_δρu⃗M * n⃗
end

function boundarycondition!(::EulerModel, stateP::Vars, _, auxP::Vars, normalM,
                            stateM::Vars, _, auxM::Vars, bctype, t)
  if bctype == 1
    nofluxbc!(stateP, normalM, stateM)
  else
    error("unknown boundary condition type!")
  end
end

function wavespeed(m::EulerModel, nM, state::Vars, aux::Vars, _)
  DFloat = eltype(state)
  ρ, ρu⃗, ρe = fullstate(m, state, aux)
  n⃗ = SVector{3, DFloat}(nM)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  e = ρinv * ρe
  ϕ = geopotential(m.gravity, aux)
  abs(n⃗' * u⃗) + soundspeed_air(PhaseDry(e - u⃗' * u⃗ / 2 - ϕ, ρ))
end

function flux!(m::EulerModel, flux::Grad, state::Vars, _::Vars, aux::Vars,
               t::Real)
  ρ, ρu⃗, ρe = fullstate(m, state, aux)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  e = ρinv * ρe
  ϕ = geopotential(m.gravity, aux)
  p = air_pressure(PhaseDry(e - u⃗' * u⃗ / 2 - ϕ, ρ))
  δp_or_p = m.pde_level_hydrostatic_balance ? pressure_perturbation(m.refstate, aux.refstate, p, ϕ) : p

  # compute the flux!
  flux.δρ  = ρu⃗
  flux.δρu⃗ = ρu⃗ .* u⃗' + δp_or_p * I
  flux.δρe = u⃗ * (ρe + p)
end

problem_specific_source!(::EulerModel, ::EulerProblem, _...) = nothing
function source!(m::EulerModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.δρ = 0
  source.δρu⃗ = @SVector zeros(eltype(source.δρu⃗), 3)
  source.δρe = 0

  geopotential_source!(m, m.gravity, source, state, aux)
  m.coriolis && coriolis_source!(source, state)
  problem_specific_source!(m, m.problem, source, state, aux)
end

abstract type GravityModel end
vars_aux(m::GravityModel, DFloat) = @vars(ϕ::DFloat, ∇ϕ::SVector{3, DFloat})
geopotential(::GravityModel, aux) = aux.gravity.ϕ
function geopotential_source!(m::EulerModel, ::GravityModel, source, state, aux)
  δρ_or_ρ = m.pde_level_hydrostatic_balance ? state.δρ : fullstate(m, state, aux).ρ
  source.δρu⃗ -= δρ_or_ρ * aux.gravity.∇ϕ
end

struct NoGravity <: GravityModel end
vars_aux(m::NoGravity, _) = @vars()
init_aux!(::NoGravity, _...) = nothing
geopotential(::NoGravity, _...) = 0
geopotential_source!(::EulerModel, ::NoGravity, _...) = nothing

struct SphereGravity{DFloat} <: GravityModel
  h::DFloat
end
function init_aux!(g::SphereGravity, aux, x⃗)
  x⃗ = SVector(x⃗)
  r = norm(x⃗, 2)
  aux.gravity.ϕ = grav * (r - g.h)
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

function coriolis_source!(source, state)
   Ω⃗ = SVector(0, 0, Omega)
   source.δρu⃗ += -2Ω⃗ × state.δρu⃗
end
