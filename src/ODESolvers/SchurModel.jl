using CLIMA.VariableTemplates
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.PlanetParameters: kappa_d, R_d, T_0

using LinearAlgebra

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux_nondiffusive!, flux_diffusive!,
                        source!, boundary_state!,
                        gradvariables!,
                        diffusive!, init_aux!, init_state!,
                        LocalGeometry

import CLIMA.DGmethods.NumericalFluxes: NumericalFluxDiffusive,
                                        NumericalFluxNonDiffusive,
                                        numerical_flux_diffusive!,
                                        numerical_flux_nondiffusive!


function schur_update!(mQ::BalanceLaw, mP::BalanceLaw,
                      Qtt::Vars, Qhat::Vars,
                      P::Vars, auxP::Vars, diffP::Vars,
                      dt::Real)
  γ = 1 / (1 - kappa_d)
  p = P.p
  h0 = auxP.h0
  ∇h0 = auxP.∇h0
  Φ = auxP.Φ - R_d * T_0 / (γ - 1)
 
  Qtt.ρu = Qhat.ρu - dt * diffP.∇p
  Qtt.ρ = (p / (γ - 1) + dt * dot(∇h0, Qtt.ρu) + h0 * Qhat.ρ - Qhat.ρe) / (h0 - Φ)
  Qtt.ρe = Qhat.ρe + h0 * (Qtt.ρ - Qhat.ρ) - dt * dot(∇h0, Qtt.ρu)
end

# LHS
struct SchurLHSModel <: BalanceLaw end

vars_aux(::SchurLHSModel, T) = @vars begin
  h0::T
  ∇h0::SVector{3, T}
  Φ::T
end
vars_state(::SchurLHSModel, T) = @vars(p::T)
vars_gradient(::SchurLHSModel, T) = @vars(p::T)
vars_diffusive(::SchurLHSModel, T) = @vars(∇p::SVector{3,T})

function flux_nondiffusive!(::SchurLHSModel, flux::Grad, state::Vars,
                            auxstate::Vars, t::Real)
  nothing
end

function flux_diffusive!(m::SchurLHSModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, dt::Real)
  flux.p = dt * diffusive.∇p
end

struct ZeroNumFluxNonDiffusive <: NumericalFluxNonDiffusive end
function numerical_flux_nondiffusive!(::ZeroNumFluxNonDiffusive, ::SchurLHSModel, F::MArray, nM,
                                      QM, auxM, QP, auxP, t)
  nothing
end
function boundary_state!(::ZeroNumFluxNonDiffusive, ::SchurLHSModel,
                         stateP::Vars, auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype, t, _...)
  nothing
end

boundary_state!(::CentralGradPenalty, ::SchurLHSModel, _...) = nothing

function gradvariables!(::SchurLHSModel, transformstate::Vars, state::Vars, auxstate::Vars, t::Real)
  transformstate.p = state.p
end

function diffusive!(::SchurLHSModel, diffusive::Vars,
                    ∇transform::Grad, state::Vars, auxstate::Vars, t::Real)
  diffusive.∇p = ∇transform.p
end

function source!(m::SchurLHSModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, dt::Real)
  γ = 1 / (1 - kappa_d)
  Φ = aux.Φ - R_d * T_0 / (γ - 1)
  h0 = aux.h0
  ∇h0 = aux.∇h0
  p = state.p
  source.p = p / (dt * (γ - 1) * (h0 - Φ)) - dt * dot(∇h0, diffusive.∇p) / (h0 - Φ)
end

function init_aux!(::SchurLHSModel, aux::Vars, g::LocalGeometry)
end
function init_state!(::SchurLHSModel, state::Vars, aux::Vars, coords, t)
end


# RHS

struct SchurRHSModel <: BalanceLaw end

vars_aux(::SchurRHSModel, T) = @vars begin
  ρ::T
  ρu::SVector{3, T}
  ρe::T
  h0::T
  ∇h0::SVector{3, T}
  Φ::T
end


vars_state(::SchurRHSModel, T) = @vars(p::T)
vars_gradient(::SchurRHSModel, T) = @vars()
vars_diffusive(::SchurRHSModel, T) = @vars()

function flux_nondiffusive!(m::SchurRHSModel, flux::Grad, state::Vars,
                            auxstate::Vars, t::Real)
  flux.p = auxstate.ρu
end

function flux_diffusive!(::SchurRHSModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, t::Real)
  nothing
end

function source!(m::SchurRHSModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, dt::Real)
  γ = 1 / (1 - kappa_d)
  ρe = aux.ρe
  ρu = aux.ρu
  ρ = aux.ρ
  Φ = aux.Φ - R_d * T_0 / (γ - 1)
  h0 = aux.h0
  ∇h0 = aux.∇h0

  source.p = (ρe - Φ * ρ) / (dt * (h0 - Φ)) - dot(∇h0, ρu) / (h0 - Φ)
end

function init_aux!(::SchurRHSModel, aux::Vars, g::LocalGeometry)
end
function init_state!(::SchurRHSModel, state::Vars, aux::Vars, coords, t)
end
