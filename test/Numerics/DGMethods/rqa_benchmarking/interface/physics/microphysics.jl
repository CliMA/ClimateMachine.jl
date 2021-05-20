abstract type AbstractMicrophysics  <: AbstractPhysicsComponent end

@Base.kwdef struct ZeroMomentMicrophysics{FT} <: AbstractMicrophysics
    τ  :: FT # s
end

@inline calc_force!(source, ::Nothing, state, _...) = nothing

@inline function calc_force!(source, microphysics::ZeroMomentMicrophysics, state, aux, _...) 
  ρ   = state.ρ
  ρu  = state.ρu
  ρe  = state.ρe
  ρqₜ = state.ρqₜ
  Φ   = aux.Φ
  τ   = microphysics.τ
  qₜ  = ρqₜ / ρ

  # thermodynamic parameters
  pₜᵣ   = press_triple(param_set)
  R_v   = R_v(param_set)
  Tₜᵣ   = T_triple(param_set)
  T_0   = T_0(param_set)
  cv_d  = cv_d(param_set)
  cv_v  = cv_v(param_set)
  cv_l  = cv_l(param_set)
  cp_v  = cp_v(param_set)
  LH_v0 = LH_v0(param_set)
  cp_v  = cp_v(param_set)
  cp_l  = cp_l(param_set)
  e_int_v0 = e_int_v0(param_set)

  # internal energy
  ρinv = 1 / ρ
  ρe_kin = ρinv * sum(abs2, ρu) / 2
  ρe_pot = ρ * Φ
  if total_energy
    ρe_int = ρe - ρe_kin - ρe_pot
  else
    ρe_int = ρe - ρe_kin - ρe_pot
  end
  e_int  = ρinv * ρe_int

  # temperature
  cv_m = cv_d + (cv_v - cv_d) * qₜ
  T = T_0 + (e_int - qₜ * e_int_v0) / cv_m

  # saturation vapor pressure
  Δcp  = cp_v - cp_l
  pᵥₛ = pₜᵣ * (T / Tₜᵣ)^(Δcp / R_v) * exp((LH_v0 - Δcp * T_0) / R_v * (1 / Tₜᵣ - 1 / T))

  # saturation specific humidity
  qᵥₛ = pᵥₛ / (ρ * R_v * T)

  # saturation excess
  S = max(0, qₜ - qᵥₛ)

  # liquid internal energy
  Iₗ = cv_l * (T - T_0)

  source.ρ   -= ρ * S / τ
  source.ρe  -= (Iₗ + Φ) * ρ * S / τ
  source.ρqₜ -= ρ * S / τ
end