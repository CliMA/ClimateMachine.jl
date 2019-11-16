abstract type AtmosLinearModel <: BalanceLaw
end


vars_state(lm::AtmosLinearModel, FT) = vars_state(lm.atmos,FT)
vars_gradient(lm::AtmosLinearModel, FT) = @vars()
vars_diffusive(lm::AtmosLinearModel, FT) = @vars()
vars_aux(lm::AtmosLinearModel, FT) = vars_aux(lm.atmos,FT)
vars_integrals(lm::AtmosLinearModel, FT) = @vars()


update_aux!(dg::DGModel, lm::AtmosLinearModel, Q::MPIStateArray, auxstate::MPIStateArray, t::Real) = nothing
integrate_aux!(lm::AtmosLinearModel, integ::Vars, state::Vars, aux::Vars) = nothing
flux_diffusive!(lm::AtmosLinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return soundspeed_air(ref.T)
end

function boundary_state!(nf::Rusanov, lm::AtmosLinearModel, x...)
  atmos_boundary_state!(nf, NoFluxBC(), lm.atmos, x...)
end
function boundary_state!(nf::CentralNumericalFluxDiffusive, lm::AtmosLinearModel, x...)
  nothing
end
init_aux!(lm::AtmosLinearModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(lm::AtmosLinearModel, state::Vars, aux::Vars, coords, t) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
  atmos::M
end
function flux_nondiffusive!(lm::AtmosAcousticLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  # TODO: use MoistThermodynamics.linearized_air_pressure 
  # need to avoid dividing then multiplying by ρ
  pL = state.ρ * FT(R_d) * FT(T_0) + FT(R_d) / FT(cv_d) * (state.ρe - state.ρ * e_pot)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ - e_pot)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  nothing
end

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
  atmos::M
end
function flux_nondiffusive!(lm::AtmosAcousticGravityLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  pL = state.ρ * FT(R_d) * FT(T_0) + FT(R_d) / FT(cv_d) * (state.ρe - state.ρ * e_pot)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticGravityLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
  source.ρu = state.ρ * ∇Φ
  nothing
end
