abstract type AbstractMicrophysics  <: AbstractPhysicsComponent end

@Base.kwdef struct ZeroMomentMicrophysics <: AbstractMicrophysics end

@inline calc_component!(source, ::Nothing, state, _...) = nothing

@inline function calc_component!(source, ::ZeroMomentMicrophysics, state, aux, physics) 
  ρ  = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  ρq = state.ρq
  Φ  = aux.Φ
  τ  = physics.parameters.τ_precip

  q  = ρq / ρ

  # thermodynamic parameters
  pₜᵣ      = get_planet_parameter(:press_triple) 
  R_v      = get_planet_parameter(:R_v)
  Tₜᵣ      = get_planet_parameter(:T_triple)
  T_0      = get_planet_parameter(:T_0)
  cv_d     = get_planet_parameter(:cv_d)
  cv_v     = get_planet_parameter(:cv_v)
  cv_l     = get_planet_parameter(:cv_l)
  cp_v     = get_planet_parameter(:cp_v)
  LH_v0    = get_planet_parameter(:LH_v0)
  cp_v     = get_planet_parameter(:cp_v)
  cp_l     = get_planet_parameter(:cp_l)
  e_int_v0 = get_planet_parameter(:e_int_v0)

  # moist internal energy (dry calculation)
  ρ⁻¹ = 1 / ρ
  ρe_kin = ρ⁻¹ * (ρu ⋅ ρu) / 2
  ρe_pot = ρ * Φ
  ρe_int = ρe - ρe_kin - ρe_pot # - latent_energy
  e_int = ρ⁻¹ * ρe_int

  # temperature
  cv_m = cv_d + (cv_v - cv_d) * q
  T = T_0 + (e_int - q * e_int_v0) / cv_m

  # saturation vapor pressure
  Δcp = cp_v - cp_l
  pᵥₛ = pₜᵣ * (T / Tₜᵣ)^(Δcp / R_v) * exp((LH_v0 - Δcp * T_0) / R_v * (1 / Tₜᵣ - 1 / T))

  # saturation specific humidity
  qᵥₛ = pᵥₛ / (ρ * R_v * T)

  # saturation excess
  S = max(0, q - qᵥₛ)

  # liquid internal energy
  Iₗ = cv_l * (T - T_0)

  source.ρ  -= ρ * S / τ
  source.ρe -= (Iₗ + Φ) * ρ * S / τ
  source.ρq -= ρ * S / τ
end
