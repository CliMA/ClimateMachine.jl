abstract type AtmosLinearModel <: BalanceLaw
end


vars_state(lm::AtmosLinearModel, T) = vars_state(lm.atmos,T)
vars_gradient(lm::AtmosLinearModel, T) = @vars()
vars_diffusive(lm::AtmosLinearModel, T) = @vars()
vars_aux(lm::AtmosLinearModel, T) = vars_aux(lm.atmos,T)
vars_integrals(lm::AtmosLinearModel,T) = @vars()


update_aux!(dg::DGModel, lm::AtmosLinearModel, Q::MPIStateArray, auxstate::MPIStateArray, t::Real) = nothing
integrate_aux!(lm::AtmosLinearModel, integ::Vars, state::Vars, aux::Vars) = nothing
flux_diffusive!(lm::AtmosLinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
  return soundspeed_air(Float64(T_0))
end

boundary_state!(nf, lm::AtmosLinearModel, x...) = nothing
init_aux!(lm::AtmosLinearModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(lm::AtmosLinearModel, state::Vars, aux::Vars, coords, t) = nothing


function linear_pressure(moisture, orientation, state::Vars, aux::Vars)
  ρ = state.ρ
  invρ = inv(ρ)
  TS = thermo_state(moisture, orientation, state, aux)
  # TODO: avoid dividing then multiplying by ρ
  linearized_air_pressure(ρ, state.ρe * invρ, gravitational_potential(orientation, aux), PhasePartition(TS))
end


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
  atmos::M
end
function flux_nondiffusive!(lm::AtmosAcousticLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  DFloat = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  #pL = linear_pressure(lm.atmos.moisture, lm.atmos.orientation, state, aux)
  pL = state.ρ * DFloat(R_d) * DFloat(T_0) + DFloat(R_d) / DFloat(cv_d) * (state.ρe - state.ρ * e_pot)
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
  ref = aux.ref_state

  flux.ρ = state.ρu
  pL = linear_pressure(lm.atmos.moisture, lm.atmos.orientation, state, aux)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticGravityLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
  source.ρu = state.ρ * ∇Φ
  nothing
end
