abstract type AbstractMicrophysics  <: AbstractPhysicsComponent end

@Base.kwdef struct ZeroMomentMicrophysics <: AbstractMicrophysics end

@inline calc_component!(source, ::Nothing, state, _...) = nothing

@inline function calc_component!(source, ::ZeroMomentMicrophysics, state, aux, physics) 
  ρ  = state.ρ
  ρq = state.ρq
  Φ  = aux.Φ
  eos = physics.eos
  parameters = physics.parameters
  τ = parameters.τ_precip
  T_0  = parameters.T_0 
  cv_l = parameters.cv_l

  T = calc_air_temperature(eos, state, aux, parameters) 
  qᵥₛ = calc_saturation_specific_humidity(ρ, T, parameters) 
  ρS = max(0, ρq - ρ * qᵥₛ) # saturation excess
  Iₗ = cv_l * (T - T_0) # liquid internal energy

  source.ρ  -= ρS / τ
  source.ρe -= (Iₗ + Φ) * ρS / τ
  source.ρq -= ρS / τ
end