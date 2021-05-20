abstract type AbstractMicrophysics  <: AbstractPhysicsComponent end

@Base.kwdef struct ZeroMomentMicrophysics{FT} <: AbstractMicrophysics
    τ :: FT # s
end

@inline calc_force!(source, ::Nothing, state, _...) = nothing

@inline function calc_force!(source, microphysics::ZeroMomentMicrophysics, state, aux, _...) 
  ρ  = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  ρq = state.ρq
  Φ  = aux.Φ
  τ  = microphysics.τ

  q  = ρq / ρ

  # thermodynamic parameters
  _pₜᵣ      = press_triple(param_set) 
  _R_v      = R_v(param_set)
  _Tₜᵣ      = T_triple(param_set)
  _T_0      = T_0(param_set)
  _cv_d     = cv_d(param_set)
  _cv_v     = cv_v(param_set)
  _cv_l     = cv_l(param_set)
  _cp_v     = cp_v(param_set)
  _LH_v0    = LH_v0(param_set)
  _cp_v     = cp_v(param_set)
  _cp_l     = cp_l(param_set)
  _e_int_v0 = e_int_v0(param_set)

  # moist internal energy (dry calculation)
  ρ⁻¹ = 1 / ρ
  ρe_kin = ρ⁻¹ * (ρu ⋅ ρu) / 2
  ρe_pot = ρ * Φ
  if total_energy
    ρe_int = ρe - ρe_kin - ρe_pot # - latent_energy
  else
    ρe_int = ρe - ρe_kin
  end
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