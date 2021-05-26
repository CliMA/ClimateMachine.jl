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
  _pₜᵣ      = get_planet_parameter(:press_triple) 
  _R_v      = get_planet_parameter(:R_v)
  _Tₜᵣ      = get_planet_parameter(:T_triple)
  _T_0      = get_planet_parameter(:T_0)
  _cv_d     = get_planet_parameter(:cv_d)
  _cv_v     = get_planet_parameter(:cv_v)
  _cv_l     = get_planet_parameter(:cv_l)
  _cp_v     = get_planet_parameter(:cp_v)
  _LH_v0    = get_planet_parameter(:LH_v0)
  _cp_v     = get_planet_parameter(:cp_v)
  _cp_l     = get_planet_parameter(:cp_l)
  _e_int_v0 = get_planet_parameter(:e_int_v0)

  # moist internal energy (dry calculation)
  ρ⁻¹ = 1 / ρ
  ρe_kin = ρ⁻¹ * (ρu ⋅ ρu) / 2
  ρe_pot = ρ * Φ
  ρe_int = ρe - ρe_kin - ρe_pot # - latent_energy
  e_int = ρ⁻¹ * ρe_int

  # temperature
  cv_m = _cv_d + (_cv_v - _cv_d) * q
  T = _T_0 + (e_int - q * _e_int_v0) / cv_m

  # saturation vapor pressure
  Δcp = _cp_v - _cp_l
  pᵥₛ = _pₜᵣ * (T / _Tₜᵣ)^(Δcp / _R_v) * exp((_LH_v0 - Δcp * _T_0) / _R_v * (1 / _Tₜᵣ - 1 / T))

  # saturation specific humidity
  qᵥₛ = pᵥₛ / (ρ * _R_v * T)

  # saturation excess
  S = max(0, q - qᵥₛ)

  # liquid internal energy
  Iₗ = _cv_l * (T - _T_0)

  source.ρ  -= ρ * S / τ
  source.ρe -= (Iₗ + Φ) * ρ * S / τ
  source.ρq -= ρ * S / τ
end