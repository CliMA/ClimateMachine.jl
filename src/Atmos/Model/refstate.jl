### Reference state

abstract type AbstractReferenceState{T} end
struct ReferenceState{T} <: AbstractReferenceState{T}
  T_ground::T
  P_ground::T
  q_tot_ground::T
end

vars_state(m::AbstractReferenceState    , DT) = @vars()
vars_gradient(m::AbstractReferenceState , DT) = @vars()
vars_diffusive(m::AbstractReferenceState, DT) = @vars()
vars_aux(m::AbstractReferenceState      , DT) = @vars(ρ::DT, p::DT, T::DT)

"""
    column_integration_kernel!(m::ReferenceState, am::AtmosModel, state::Vars, aux::Vars)

Column-wise integral kernel to compute reference state fields:
  - ρ density
  - p presure
  - T temperature

For accuracy, the logarithm of the expression is in the integrand, which must be exponentiated
after computing the integral.
"""
function column_integration_kernel!(m::ReferenceState, am::AtmosModel, state::Vars, aux::Vars)
  # Unpack ground values from `ReferenceState`
  q_pt_g = PhasePartition(m.q_tot_ground)
  e_int_g = internal_energy(m.T_ground, q_pt_g)
  logp = log(m.P_ground)

  # Use ground values along integral:
  aux.moisture.e_int = e_int_g
  aux.pressure = logp
  state.ρ = air_density(m.T_ground, exp(aux.pressure), q_pt_g)
  state.moisture.ρq_tot = m.q_tot_ground*state.ρ

  # Integrand:
  ts = thermo_state(am.moisture, state, aux)
  R_m = gas_constant_air(ts)
  H = R_m*air_temperature(ts)/grav
  aux.refstate.pressure_integrand = -1/H
end

"""
    post_column_integration_kernel!(m::ReferenceState, am::AtmosModel, state::Vars, aux::Vars)

Column-wise integral kernel to compute reference state fields:
  - ρ density
  - p presure
  - T temperature

This function call occurs once after `column_integration_kernel!`.
"""
function post_column_integration_kernel!(m::ReferenceState, am::AtmosModel, state::Vars, aux::Vars)
  # Unpack ground values from `ReferenceState`
  q_pt_g = PhasePartition(m.q_tot_ground)
  e_int_g = internal_energy(m.T_ground, q_pt_g)
  logp = log(m.P_ground)

  aux.refstate.p = exp(aux.refstate.pressure_integrand)
  aux.moisture.e_int = e_int_g
  state.moisture.ρq_tot = m.q_tot_ground*state.ρ

  # Solve for pressure
  aux.refstate.p = MSLP*exp(aux.refstate.pressure_integrand)

  # Establish thermodynamic state
  ts = TemperatureSHumEquil(m.T_ground, q_pt_g, aux.refstate.p)

  # Solve for remaining reference state fields
  aux.refstate.T = air_temperature(ts)
  aux.refstate.ρ = air_density(aux.refstate.T, aux.refstate.p, PhasePartition(ts))
end
