abstract type AbstractEquationOfState end
abstract type AbstractIdealGas <: AbstractEquationOfState end

struct BarotropicFluid <: AbstractEquationOfState end
struct DryIdealGas <: AbstractIdealGas end
struct MoistIdealGas <: AbstractIdealGas end

@inline function calc_pressure(::BarotropicFluid, state, aux, params)
    ρ  = state.ρ
    cₛ = params.cₛ
    ρₒ = params.ρₒ

    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_pressure(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ

    R_d = calc_gas_constant(eos, state, params)
    T = calc_air_temperature(eos, state, aux, params)

    return ρ * R_d * T
end

@inline function calc_pressure(eos::MoistIdealGas, state, aux, params)
    ρ  = state.ρ

    R_m = calc_gas_constant(eos, state, params)
    T = calc_air_temperature(eos, state, aux, params)
    
    return ρ * R_m * T 
end

@inline function calc_linear_pressure(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    Φ  = aux.Φ
    T_0  = params.T_0

    γ = calc_heat_capacity_ratio(eos, state, params)
    cv_d = calc_heat_capacity_at_constant_volume(eos, state, params)

    return (γ - 1) * (ρe - ρ * Φ + ρ * cv_d * T_0) 
end

@inline function calc_linear_pressure(::MoistIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    ρ_q_tot = state.ρq 
    ρ_q_liq = 0 # zero for now
    ρ_q_ice = 0 # zero for now
    Φ = aux.Φ
    T_0  = params.T_0

    γ    = calc_heat_capacity_ratio(DryIdealGas(), state, params)
    cv_d = calc_heat_capacity_at_constant_volume(DryIdealGas(), state, params)

    ρ_e_latent = (ρ_q_tot - ρ_q_liq) * params.e_int_v0 - ρ_q_ice * (params.e_int_v0 + params.e_int_i0)
    
    return (γ - 1) * (ρe - ρ * Φ - ρ_e_latent + ρ * cv_d * T_0)
end

@inline function calc_very_linear_pressure(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ

    # Reference states
    ρᵣ  = aux.ref_state.ρ
    ρuᵣ = aux.ref_state.ρu

    γ  = calc_heat_capacity_ratio(eos, state, params)

    return (γ - 1) * (ρe - dot(ρuᵣ, ρu) / ρᵣ + ρ * dot(ρuᵣ, ρuᵣ) / (2*ρᵣ^2) - ρ * Φ)
end

@inline function calc_sound_speed(::BarotropicFluid, state, aux, params)
    ρ = state.ρ
    cₛ = params.cₛ 
    ρₒ = params.ρₒ
    
    return cₛ * sqrt(ρ / ρₒ) 
end

@inline function calc_sound_speed(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ

    γ  = calc_heat_capacity_ratio(eos, state, params)
    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_sound_speed(eos::MoistIdealGas, state, aux, params)
    ρ  = state.ρ

    γ  = calc_heat_capacity_ratio(eos, state, params)
    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(eos::DryIdealGas, state, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ

    γ = calc_heat_capacity_ratio(eos, state, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(::MoistIdealGas, state, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ

    γ = calc_heat_capacity_ratio(DryIdealGas(), state, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_air_temperature(::DryIdealGas, state, aux, params)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ
  T_0 = params.T_0

  cv_d = calc_heat_capacity_at_constant_volume(DryIdealGas(), state, params)

  e_int = (ρe - ρu' * ρu / 2ρ - ρ * Φ) / ρ
  T = T_0 + e_int / cv_d

  return T
end

@inline function calc_air_temperature(eos::MoistIdealGas, state, aux, params)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  q_tot = state.ρq / ρ
  q_liq = 0.0 # zero for now
  q_ice = 0.0 # zero for now
  Φ = aux.Φ
  T_0 = params.T_0
  e_int_v0 = params.e_int_v0
  e_int_i0 = params.e_int_i0

  cv_m = calc_heat_capacity_at_constant_volume(eos, state, params)
  
  e_int = (ρe - ρu' * ρu / 2ρ - ρ * Φ) / ρ
  T = T_0 + (e_int - (q_tot - q_liq) * e_int_v0 + q_ice * (e_int_v0 + e_int_i0)) / cv_m

  return T
end

@inline function calc_total_specific_enthalpy(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe

    p  = calc_pressure(eos, state, aux, params)

    return (ρe + p) / ρ
end

@inline function calc_total_specific_enthalpy(eos::MoistIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe

    p  = calc_pressure(eos, state, aux, params)

    return (ρe + p) / ρ
end

@inline function calc_heat_capacity_at_constant_pressure(::DryIdealGas, state, params)
    return params.cp_d
end

@inline function calc_heat_capacity_at_constant_pressure(::MoistIdealGas, state, params)
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for now
    cp_d  = params.cp_d
    cp_v  = params.cp_v
    cp_l  = params.cp_l
    cp_i  = params.cp_i

    cp_m  = cp_d + (cp_v - cp_d) * q_tot + (cp_l - cp_v) * q_liq + (cp_i - cp_v) * q_ice

    return cp_m
end

@inline function calc_heat_capacity_at_constant_volume(::DryIdealGas, state, params)
    return params.cv_d 
end

@inline function calc_heat_capacity_at_constant_volume(::MoistIdealGas, state, params)
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for now   
    cv_d  = params.cv_d
    cv_v  = params.cv_v
    cv_l  = params.cv_l
    cv_i  = params.cv_i

    cv_m  = cv_d + (cv_v - cv_d) * q_tot + (cv_l - cv_v) * q_liq + (cv_i - cv_v) * q_ice

    return cv_m
end

@inline function calc_gas_constant(::DryIdealGas, state, params)
    return params.R_d
end

@inline function calc_gas_constant(::MoistIdealGas, state, params)
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for nov
    R_d = params.R_d
    molmass_ratio = params.molmass_ratio

    R_m = R_d * (1 + (molmass_ratio - 1) * q_tot - molmass_ratio * (q_liq + q_ice))

    return R_m
end

@inline function calc_heat_capacity_ratio(eos::AbstractEquationOfState, state, params)
    cp = calc_heat_capacity_at_constant_pressure(eos, state, params)
    cv = calc_heat_capacity_at_constant_volume(eos, state, params)
    γ  = cp/cv

    return γ
end

@inline function calc_saturation_vapor_pressure(T, params)
    pₜᵣ   = params.pₜᵣ
    R_v   = params.R_v
    Tₜᵣ   = params.Tₜᵣ
    T_0   = params.T_0
    cp_v  = params.cp_v
    cp_l  = params.cp_l
    LH_v0 = params.LH_v0
    
    Δcp = cp_v - cp_l
    pᵥₛ = pₜᵣ * (T / Tₜᵣ)^(Δcp / R_v) * exp((LH_v0 - Δcp * T_0) / R_v * (1 / Tₜᵣ - 1 / T))

    return pᵥₛ
end

@inline function calc_saturation_specific_humidity(ρ, T, params)
    R_v = params.R_v
    
    pᵥₛ = calc_saturation_vapor_pressure(T, params)
    qt  = pᵥₛ / (ρ * R_v * T)
    
    return qt
end