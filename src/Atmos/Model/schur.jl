#using CLIMA.VariableTemplates
#using CLIMA.DGmethods
#using CLIMA.DGmethods.NumericalFluxes
#using CLIMA.PlanetParameters: kappa_d, R_d, T_0
#
#import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_gradient,
#                        vars_state, vars_instate, vars_outstate,
#                        vars_diffusive, flux_nondiffusive!, flux_diffusive!,
#                        source!, boundary_state!,
#                        gradvariables!,
#                        diffusive!, init_aux!, init_state!,
#                        LocalGeometry
#
#import CLIMA.DGmethods.NumericalFluxes: NumericalFluxDiffusive,
#                                        NumericalFluxNonDiffusive,
#                                        numerical_flux_diffusive!,
#                                        numerical_flux_nondiffusive!


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


#struct ZeroNumFluxNonDiffusive <: NumericalFluxNonDiffusive end
#function numerical_flux_nondiffusive!(::ZeroNumFluxNonDiffusive, ::SchurLHSModel, F::MArray, nM,
#                                      QM, auxM, QP, auxP, t)
#  nothing
#end
#function boundary_state!(::ZeroNumFluxNonDiffusive, ::SchurLHSModel,
#                         stateP::Vars, auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype, t, _...)
#  nothing
#end

boundary_state!(::CentralGradPenalty, ::SchurLHSModel, _...) = nothing

function boundary_state!(::CentralNumericalFluxDiffusive, ::SchurLHSModel,
                               stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  #stateP.p = stateM.p
  #diffP.∇p = diffM.∇p
  diffP.∇p = diffM.∇p - 2 * dot(diffM.∇p, nM) * nM
end

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
  h0::T
  ∇h0::SVector{3, T}
  Φ::T
end

vars_instate(::SchurRHSModel, T) = @vars begin
  ρ::T
  ρu::SVector{3, T}
  ρe::T
end
vars_outstate(::SchurRHSModel, T) = @vars(p::T)
vars_gradient(::SchurRHSModel, T) = @vars()
vars_diffusive(::SchurRHSModel, T) = @vars()

function flux_nondiffusive!(m::SchurRHSModel, flux::Grad, state::Vars,
                            auxstate::Vars, t::Real)
  flux.p = state.ρu
end
function boundary_state!(::CentralNumericalFluxNonDiffusive, ::SchurRHSModel,
                         stateP::Vars,
                         auxP::Vars, nM, stateM::Vars,
                         auxM::Vars, bctype, t, _...)
  stateP.ρu = stateM.ρu - 2 * dot(stateM.ρu, nM) * nM
end

boundary_state!(::CentralNumericalFluxDiffusive, ::SchurRHSModel, _...) = nothing

function flux_diffusive!(::SchurRHSModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, t::Real)
  nothing
end

function source!(m::SchurRHSModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, dt::Real)
  γ = 1 / (1 - kappa_d)
  ρe = state.ρe
  ρu = state.ρu
  ρ = state.ρ
  Φ = aux.Φ - R_d * T_0 / (γ - 1)
  h0 = aux.h0
  ∇h0 = aux.∇h0

  source.p = (ρe - Φ * ρ) / (dt * (h0 - Φ)) - dot(∇h0, ρu) / (h0 - Φ)
end

function init_aux!(::SchurRHSModel, aux::Vars, g::LocalGeometry)
end
function init_state!(::SchurRHSModel, state::Vars, aux::Vars, coords, t)
end


# UPDATE
struct SchurUpdateModel <: BalanceLaw end

vars_aux(::SchurUpdateModel, T) = @vars begin
  h0::T
  ∇h0::SVector{3, T}
  Φ::T
  ρ::T
  ρu::SVector{3, T}
  ρe::T
end
vars_instate(::SchurUpdateModel, T) = @vars(p::T)
vars_outstate(::SchurUpdateModel, T) = @vars begin
  ρ::T
  ρu::SVector{3, T}
  ρe::T
end
vars_gradient(::SchurUpdateModel, T) = @vars(p::T)
vars_diffusive(::SchurUpdateModel, T) = @vars(∇p::SVector{3,T})

function flux_nondiffusive!(::SchurUpdateModel, flux::Grad, state::Vars,
                            auxstate::Vars, t::Real)
  nothing
end

function flux_diffusive!(m::SchurUpdateModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, dt::Real)
  flux.ρ = dt * (auxstate.ρu - dt * diffusive.∇p)
  flux.ρu += dt * state.p * I
  flux.ρe = dt * (auxstate.h0 * (auxstate.ρu - dt * diffusive.∇p))
end

#function numerical_flux_nondiffusive!(::ZeroNumFluxNonDiffusive, ::SchurUpdateModel, F::MArray, nM,
#                                      QM, auxM, QP, auxP, t)
#  nothing
#end
#function boundary_state!(::ZeroNumFluxNonDiffusive, ::SchurUpdateModel,
#                         stateP::Vars, auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype, t, _...)
#  nothing
#end
boundary_state!(::CentralGradPenalty, ::SchurUpdateModel, _...) = nothing

function boundary_state!(::CentralNumericalFluxDiffusive, ::SchurUpdateModel,
                               stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  stateP.p = stateM.p
  diffP.∇p = diffM.∇p - 2 * dot(diffM.∇p, nM) * nM
  auxP.ρu = auxM.ρu - 2 * dot(auxM.ρu, nM) * nM
end

function gradvariables!(::SchurUpdateModel, transformstate::Vars, state::Vars, auxstate::Vars, t::Real)
  transformstate.p = state.p
end

function diffusive!(::SchurUpdateModel, diffusive::Vars,
                    ∇transform::Grad, state::Vars, auxstate::Vars, t::Real)
  diffusive.∇p = ∇transform.p
end

function source!(m::SchurUpdateModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, dt::Real)
  source.ρ = aux.ρ
  source.ρu = aux.ρu
  source.ρe = aux.ρe
end

init_aux!(::SchurUpdateModel, aux::Vars, g::LocalGeometry) = nothing
init_state!(::SchurUpdateModel, state::Vars, aux::Vars, coords, t) = nothing

function schur_aux_init!(::Union{SchurLHSModel, SchurUpdateModel}, schur_state::Vars, schur_aux::Vars,
                         lin, atmos_state::Vars, atmos_aux::Vars, t::Real)
  Φ = gravitational_potential(lin.atmos.orientation, atmos_aux)
  schur_aux.h0 = (atmos_aux.ref_state.ρe + atmos_aux.ref_state.p) / atmos_aux.ref_state.ρ - Φ
  schur_aux.Φ = Φ
end

function schur_pressure_init!(::SchurLHSModel, schur_state::Vars, schur_aux::Vars,
                              lin, atmos_state::Vars, atmos_aux::Vars, t::Real)
  γ = 1 / (1 - kappa_d)
  Φ = gravitational_potential(lin.atmos.orientation, atmos_aux)
  schur_state.p = (γ - 1) * (atmos_state.ρe - atmos_state.ρ * (Φ - R_d * T_0 / (γ - 1)))
end

function schur_copy_state!(::SchurUpdateModel, schur_state::Vars, schur_aux::Vars,
                           lin, atmos_state::Vars, atmos_aux::Vars, t::Real)
  schur_aux.ρ = atmos_state.ρ
  schur_aux.ρu = atmos_state.ρu
  schur_aux.ρe = atmos_state.ρe
end
