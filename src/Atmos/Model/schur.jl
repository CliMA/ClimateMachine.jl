using CLIMA.DGmethods: nodal_transfer_state!, grad_auxiliary_state!, init_ode_state
using CLIMA.LinearSolvers: linearsolve!
import CLIMA.AdditiveRungeKuttaMethod: SchurComplement, ark_linearsolve!

struct AtmosSchurComplement{AT} <: SchurComplement
  lhs!::DGModel
  rhs!::DGModel
  update!::DGModel
  P::AT
  R::AT
end
function AtmosSchurComplement(lineardg::DGModel, Q)
  linearmodel = lineardg.balancelaw
  grid = lineardg.grid

  #TODO: generalize to AtmosAcousticGravityLinearModel 
  @assert linearmodel isa AtmosAcousticLinearModel

  lhs_model = SchurLHSModel(linearmodel)
  rhs_model = SchurRHSModel(linearmodel)
  update_model = SchurUpdateModel(linearmodel)

  lhs_dg = DGModel(lhs_model,
                   grid,
                   nothing,
                   CentralNumericalFluxDiffusive(),
                   CentralGradPenalty())
  
  rhs_dg = DGModel(rhs_model,
                   grid,
                   CentralNumericalFluxNonDiffusive(),
                   CentralNumericalFluxDiffusive(),
                   CentralGradPenalty();
                   auxstate=lhs_dg.auxstate)
  
  update_dg = DGModel(update_model,
                      grid,
                      nothing,
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty())
  P = init_ode_state(lhs_dg, 0)
  R = similar(P)
  AT = typeof(P)

  # initalize the auxiliary state
  nodal_transfer_state!(schur_auxstate_init!, grid,
                        lhs_model, P, lhs_dg.auxstate,
                        linearmodel, Q, lineardg.auxstate, 0)
  grad_auxiliary_state!(lhs_dg, 1, (2, 3, 4))
  nodal_transfer_state!(schur_auxstate_init!, grid,
                        update_model, P, update_dg.auxstate,
                        linearmodel, Q, lineardg.auxstate, 0)

  AtmosSchurComplement{AT}(lhs_dg, rhs_dg, update_dg, P, R)
end

function ark_linearsolve!(schur::AtmosSchurComplement, linearsolver, rhs_linear!, _, Qinit, Qhat, p, t, α)
  nodal_transfer_state!(schur_pressure_init!, rhs_linear!.grid,
                        schur.lhs!.balancelaw, schur.P, schur.lhs!.auxstate,
                        rhs_linear!.balancelaw, Qinit, rhs_linear!.auxstate, 0)
  schur.rhs!(schur.R, Qhat, p, α; increment = false)
  linearoperator! = function(LQ, Q)
    schur.lhs!(LQ, Q, p, α; increment = false)
  end
  linearsolve!(linearoperator!, linearsolver, schur.P, schur.R)
  nodal_transfer_state!(schur_copy_state!, rhs_linear!.grid,
                        schur.update!.balancelaw, schur.P, schur.update!.auxstate,
                        rhs_linear!.balancelaw, Qhat, rhs_linear!.auxstate, 0)
  schur.update!(Qinit, schur.P, p, α; increment = false)
end

struct SchurLHSModel{M} <: BalanceLaw
  linearmodel::M
end
vars_aux(::SchurLHSModel, FT) = @vars begin
  h0::FT
  ∇h0::SVector{3, FT}
  Φ::FT
end
vars_state(::SchurLHSModel, FT) = @vars(p::FT)
vars_gradient(::SchurLHSModel, FT) = @vars(p::FT)
vars_diffusive(::SchurLHSModel, FT) = @vars(∇p::SVector{3,FT})

init_aux!(::SchurLHSModel, aux::Vars, g::LocalGeometry) = nothing
init_state!(::SchurLHSModel, state::Vars, aux::Vars, coords, t) = nothing

function gradvariables!(::SchurLHSModel, transformstate::Vars, state::Vars, auxstate::Vars, t::Real)
  transformstate.p = state.p
end
function diffusive!(::SchurLHSModel, diffusive::Vars,
                    ∇transform::Grad, state::Vars, auxstate::Vars, t::Real)
  diffusive.∇p = ∇transform.p
end
function flux_nondiffusive!(::SchurLHSModel, flux::Grad, state::Vars,
                            auxstate::Vars, t::Real)
  nothing
end
function flux_diffusive!(m::SchurLHSModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, α::Real)
  flux.p = α * diffusive.∇p
end
function source!(m::SchurLHSModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, α::Real)
  γ = 1 / (1 - kappa_d)
  Φ = aux.Φ - R_d * T_0 / (γ - 1)
  h0 = aux.h0
  ∇h0 = aux.∇h0
  p = state.p
  source.p = p / (α * (γ - 1) * (h0 - Φ)) - α * dot(∇h0, diffusive.∇p) / (h0 - Φ)
end

boundary_state!(::CentralGradPenalty, ::SchurLHSModel, _...) = nothing
function boundary_state!(::CentralNumericalFluxDiffusive, ::SchurLHSModel,
                         stateP::Vars, diffP::Vars,
                         auxP::Vars, nM, stateM::Vars, diffM::Vars,
                         auxM::Vars, bctype, α, _...)
  diffP.∇p = diffM.∇p - 2 * dot(diffM.∇p, nM) * nM
end

struct SchurRHSModel{M} <: BalanceLaw
  linearmodel::M
end

vars_aux(::SchurRHSModel, FT) = @vars begin
  h0::FT
  ∇h0::SVector{3, FT}
  Φ::FT
end
vars_instate(::SchurRHSModel, FT) = @vars begin
  ρ::FT
  ρu::SVector{3, FT}
  ρe::FT
end
vars_outstate(::SchurRHSModel, FT) = @vars(p::FT)
vars_gradient(::SchurRHSModel, FT) = @vars()
vars_diffusive(::SchurRHSModel, FT) = @vars()

init_aux!(::SchurRHSModel, aux::Vars, g::LocalGeometry) = nothing
init_state!(::SchurRHSModel, state::Vars, aux::Vars, coords, t) = nothing

function flux_nondiffusive!(m::SchurRHSModel, flux::Grad, state::Vars,
                            auxstate::Vars, α::Real)
  flux.p = state.ρu
end
function flux_diffusive!(::SchurRHSModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, α::Real)
  nothing
end
function source!(m::SchurRHSModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, α::Real)
  γ = 1 / (1 - kappa_d)
  ρe = state.ρe
  ρu = state.ρu
  ρ = state.ρ
  Φ = aux.Φ - R_d * T_0 / (γ - 1)
  h0 = aux.h0
  ∇h0 = aux.∇h0

  source.p = (ρe - Φ * ρ) / (α * (h0 - Φ)) - dot(∇h0, ρu) / (h0 - Φ)
end

function boundary_state!(::CentralNumericalFluxNonDiffusive, ::SchurRHSModel,
                         stateP::Vars,
                         auxP::Vars, nM, stateM::Vars,
                         auxM::Vars, bctype, α, _...)
  stateP.ρu = stateM.ρu - 2 * dot(stateM.ρu, nM) * nM
end
boundary_state!(::CentralNumericalFluxDiffusive, ::SchurRHSModel, _...) = nothing

struct SchurUpdateModel{M} <: BalanceLaw
  linearmodel::M
end
vars_aux(::SchurUpdateModel, FT) = @vars begin
  h0::FT
  ∇h0::SVector{3, FT}
  Φ::FT
  ρ::FT
  ρu::SVector{3, FT}
  ρe::FT
end
vars_instate(::SchurUpdateModel, FT) = @vars(p::FT)
vars_outstate(::SchurUpdateModel, FT) = @vars begin
  ρ::FT
  ρu::SVector{3, FT}
  ρe::FT
end
vars_gradient(::SchurUpdateModel, FT) = @vars(p::FT)
vars_diffusive(::SchurUpdateModel, FT) = @vars(∇p::SVector{3,FT})

function flux_nondiffusive!(::SchurUpdateModel, flux::Grad, state::Vars,
                            auxstate::Vars, α::Real)
  nothing
end

function flux_diffusive!(m::SchurUpdateModel, flux::Grad, state::Vars,
                         diffusive::Vars, auxstate::Vars, α::Real)
  flux.ρ = α * (auxstate.ρu - α * diffusive.∇p)
  flux.ρu += α * state.p * I
  flux.ρe = α * (auxstate.h0 * (auxstate.ρu - α * diffusive.∇p))
end

boundary_state!(::CentralGradPenalty, ::SchurUpdateModel, _...) = nothing
function boundary_state!(::CentralNumericalFluxDiffusive, ::SchurUpdateModel,
                               stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, α, _...)
  stateP.p = stateM.p
  diffP.∇p = diffM.∇p - 2 * dot(diffM.∇p, nM) * nM
  auxP.ρu = auxM.ρu - 2 * dot(auxM.ρu, nM) * nM
end

function gradvariables!(::SchurUpdateModel, transformstate::Vars, state::Vars, auxstate::Vars, α::Real)
  transformstate.p = state.p
end

function diffusive!(::SchurUpdateModel, diffusive::Vars,
                    ∇transform::Grad, state::Vars, auxstate::Vars, α::Real)
  diffusive.∇p = ∇transform.p
end

function source!(m::SchurUpdateModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, α::Real)
  source.ρ = aux.ρ
  source.ρu = aux.ρu
  source.ρe = aux.ρe
end

init_aux!(::SchurUpdateModel, aux::Vars, g::LocalGeometry) = nothing
init_state!(::SchurUpdateModel, state::Vars, aux::Vars, coords, t) = nothing

function schur_auxstate_init!(::Union{SchurLHSModel, SchurUpdateModel}, schur_state::Vars, schur_aux::Vars,
                              linearmodel, atmos_state::Vars, atmos_aux::Vars, ::Real)
  Φ = gravitational_potential(linearmodel.atmos.orientation, atmos_aux)
  schur_aux.h0 = (atmos_aux.ref_state.ρe + atmos_aux.ref_state.p) / atmos_aux.ref_state.ρ - Φ
  schur_aux.Φ = Φ
end

function schur_pressure_init!(::SchurLHSModel, schur_state::Vars, schur_aux::Vars,
                              linearmodel, atmos_state::Vars, atmos_aux::Vars, ::Real)
  γ = 1 / (1 - kappa_d)
  Φ = gravitational_potential(linearmodel.atmos.orientation, atmos_aux)
  schur_state.p = (γ - 1) * (atmos_state.ρe - atmos_state.ρ * (Φ - R_d * T_0 / (γ - 1)))
end

function schur_copy_state!(::SchurUpdateModel, schur_state::Vars, schur_aux::Vars,
                           lin, atmos_state::Vars, atmos_aux::Vars, ::Real)
  schur_aux.ρ = atmos_state.ρ
  schur_aux.ρu = atmos_state.ρu
  schur_aux.ρe = atmos_state.ρe
end
