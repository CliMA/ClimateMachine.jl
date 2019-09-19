export Linear

struct Linear <: ModelPart end

@inline function flux_advective!(m::AtmosModel{Linear},
                                 flux::Grad, state::Vars, aux::Vars, t::Real)
  ρu = state.ρu
  ρ_ref = aux.ref_state.ρ
  ρu_ref = aux.ref_state.ρu
  ρe_ref = aux.ref_state.ρe

  ρ_ref_inv = 1 / ρ_ref
  u_ref = ρ_ref_inv * ρu_ref
  e_ref = ρe_ref * ρ_ref_inv 

  flux.ρ   = ρu
  flux.ρu  = ρu * u_ref'
  flux.ρe  = ρu * e_ref
end

@inline function flux_pressure!(m::AtmosModel{Linear},
                                flux::Grad, state::Vars, aux::Vars, t::Real)
  DFloat = eltype(state)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe

  ρ_inv = 1 / ρ
  u = ρ_inv * ρu

  ρ_ref = aux.ref_state.ρ
  ρ_ref_inv = 1 / ρ_ref
  p_ref = pressure(m.moisture, m.orientation, aux.ref_state, aux)
  
  e_pot = gravitational_potential(m.orientation, aux)
  # FIXME: use linearized_air_pressure, currently it doesn't work here
  # because it doesn't handle ρ == 0 well
  p_L = ρ * DFloat(R_d) * DFloat(T_0) + DFloat(R_d) / DFloat(cv_d) * (ρe - ρ * e_pot)

  #ts = thermo_state(m.moisture, m.orientation, state, aux)
  #e_kin = u' * u / 2
  #p_L = linearized_air_pressure(e_kin, e_pot, ts)

  flux.ρu += p_L * I
  flux.ρe += ρu * p_ref * ρ_ref_inv
end

@inline function wavespeed(m::AtmosModel{Linear}, nM, state::Vars, aux::Vars, t::Real)
  ρ_ref = aux.ref_state.ρ
  ρu_ref = aux.ref_state.ρu
  ρ_ref_inv = 1 / ρ_ref
  u_ref = ρ_ref_inv * ρu_ref

  soundspeed_linear = soundspeed(m.moisture, m.orientation, aux.ref_state, aux) 
  return abs(dot(nM, u_ref)) + soundspeed_linear
end

# prevents errors thrown when calculating virtual potential temperature for
# nonphysical states
atmos_nodal_update_aux!(moist::DryModel, atmos::AtmosModel{Linear},
                        state::Vars, aux::Vars, t::Real) = nothing

# needs to be defined for ambiguity resolution :(
flux_diffusive!(m::AtmosModel{Linear},
                flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real,
                ::NoViscosity) = nothing
