export NonLinear

struct NonLinear <: ModelPart end

# FIXME: this doesn't seem to work ?
const use_linearized_pressure = false

@inline function flux_advective!(m::AtmosModel{NonLinear},
                                 flux::Grad, state::Vars, aux::Vars, t::Real)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  
  ρ_ref = aux.ref_state.ρ
  ρu_ref = aux.ref_state.ρu
  ρe_ref = aux.ref_state.ρe
  
  ρ_inv = 1 / ρ
  u = ρ_inv * ρu
  e = ρ_inv * ρe

  ρ_ref_inv = 1 / ρ_ref
  u_ref = ρ_ref_inv * ρu_ref
  e_ref = ρ_ref_inv * ρe_ref

  flux.ρ   = -zero(eltype(state))
  flux.ρu  = ρu * (u - u_ref)'
  flux.ρe  = ρu * (e - e_ref)
end

@inline function flux_pressure!(m::AtmosModel{NonLinear},
                                flux::Grad, state::Vars, aux::Vars, t::Real)
  DFloat = eltype(state)
  ρ = state.ρ
  ρu = state.ρu
  ρ_ref = aux.ref_state.ρ
  
  ρ_inv = 1 / ρ
  u = ρ_inv * ρu
  ρ_ref_inv = 1 / ρ_ref

  ts = thermo_state(m.moisture, m.orientation, state, aux)
  e_kin = u' * u / 2
  e_pot = gravitational_potential(m.orientation, aux)
  p_L = linearized_air_pressure(e_kin, e_pot, ts)

  if use_linearized_pressure
    p = p_L
  else
    p = pressure(m.moisture, m.orientation, state, aux)
  end

  p_ref = pressure(m.moisture, m.orientation, aux.ref_state, aux)

  flux.ρu += (p - p_L) * I
  flux.ρe += ρu * (ρ_inv * p - ρ_ref_inv * p_ref)
end

@inline function wavespeed(m::AtmosModel{NonLinear}, nM, state::Vars, aux::Vars, t::Real)
  ρ = state.ρ
  ρu = state.ρu
  ρ_ref = aux.ref_state.ρ
  ρu_ref = aux.ref_state.ρu
  
  u = ρu / ρ
  u_ref = ρu_ref / ρ_ref

  soundspeed_full = soundspeed(m.moisture, m.orientation, state, aux)
  soundspeed_linear = soundspeed(m.moisture, m.orientation, aux.ref_state, aux)

  if use_linearized_pressure
    soundspeed_contribution = -zero(eltype(state))
  else
    soundspeed_contribution = soundspeed_full - soundspeed_linear
  end
  return abs(dot(nM, u)) - abs(dot(nM, u_ref)) + soundspeed_contribution
end

# needs to be defined for ambiguity resolution :(
flux_diffusive!(m::AtmosModel{NonLinear},
                flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real,
                ::NoViscosity) = nothing
