abstract type AtmosNonlinearModel <: BalanceLaw
end


vars_state(nlm::AtmosNonlinearModel, T) = vars_state(nlm.atmos,T)
vars_gradient(nlm::AtmosNonlinearModel, T) = vars_gradient(nlm.atmos,T)
vars_diffusive(nlm::AtmosNonlinearModel, T) = vars_diffusive(nlm.atmos,T)
vars_aux(nlm::AtmosNonlinearModel, T) = vars_aux(nlm.atmos,T)
vars_integrals(nlm::AtmosNonlinearModel,T) = vars_integrals(nlm.atmos,T)

update_aux!(dg::DGModel, nlm::AtmosNonlinearModel, Q::MPIStateArray, auxstate::MPIStateArray, t::Real) =
  update_aux!(dg, nlm.atmos, Q, auxstate, t)
  
integrate_aux!(nlm::AtmosNonlinearModel, integ::Vars, state::Vars, aux::Vars) =
  integrate_aux!(nlm.atmos, integ, state, aux)

flux_diffusive!(nlm::AtmosNonlinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) =
  flux_diffusive!(nlm.atmos, flux, state, diffusive, aux, t)

gradvariables!(nlm::AtmosNonlinearModel, transform::Vars, state::Vars, aux::Vars, t::Real) =
  gradvariables!(nlm.atmos, transform, state, aux, t)

diffusive!(nlm::AtmosNonlinearModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real) =
  diffusive!(nlm.atmos, diffusive, ∇transform, state, aux, t)

function wavespeed(nlm::AtmosNonlinearModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return wavespeed(nlm.atmos, nM, state, aux, t) - soundspeed_air(ref.T)
end

boundary_state!(nf, nlm::AtmosNonlinearModel, x...) = boundary_state!(nf, nlm.atmos, x...)

init_aux!(nlm::AtmosNonlinearModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(nlm::AtmosNonlinearModel, state::Vars, aux::Vars, coords, t) = nothing

struct AtmosAcousticNonlinearModel{M} <: AtmosNonlinearModel
  atmos::M
end
function flux_nondiffusive!(nlm::AtmosAcousticNonlinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  DF = eltype(state)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  ref = aux.ref_state
  u = ρu / ρ
  e_pot = gravitational_potential(nlm.atmos.orientation, aux)
  p = pressure(nlm.atmos.moisture, nlm.atmos.orientation, state, aux)
  # TODO: use MoistThermodynamics.linearized_air_pressure 
  # need to avoid dividing then multiplying by ρ
  pL = ρ * DF(R_d) * DF(T_0) + DF(R_d) / DF(cv_d) * (ρe - ρ * e_pot)

  flux.ρ = -zero(DF)
  flux.ρu = ρu .* u' + (p - pL) * I
  flux.ρe = ((ρe + p) / ρ - (ref.ρe + ref.p) / ref.ρ + e_pot) * ρu
end
function source!(nlm::AtmosAcousticNonlinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source!(nlm.atmos, source, state, aux, t)
end

#TODO: AtmosAcousticGravityNonlinearModel
