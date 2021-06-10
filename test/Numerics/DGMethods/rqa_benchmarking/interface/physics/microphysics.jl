abstract type AbstractMicrophysics  <: AbstractPhysicsComponent end

@Base.kwdef struct ZeroMomentMicrophysics <: AbstractMicrophysics end

@inline calc_component!(source, ::Nothing, state, _...) = nothing

@inline function calc_component!(source, ::ZeroMomentMicrophysics, state, aux, physics) 
  ρ    = state.ρ
  ρq   = state.ρq
  Φ    = aux.Φ
  eos  = physics.eos
  τ    = physics.parameters.τ_precip
  T_0  = physics.parameters.T_0 
  cv_l = physics.parameters.cv_l

  # we need the saturation excess in order to calculate the 
  # source terms for prognostic variables
  T    = calc_air_temperature(eos, state, aux, physics.parameters)
  qᵥₛ  = calc_saturation_specific_humidity(ρ, T, physics.parameters) 
  ρS   = max(0, ρq - ρ * qᵥₛ) # saturation excess
  Iₗ   = cv_l * (T - T_0) # liquid internal energy

  # source terms are proportional to the saturation excess
  source.ρ  -= ρS / τ
  source.ρe -= (Iₗ + Φ) * ρS / τ
  source.ρq -= ρS / τ
end