abstract type AbstractEquationOfState{𝒯} end

struct BarotropicFluid <: AbstractEquationOfState{𝒯} end
struct DryIdealGas{𝒯} <: AbstractEquationOfState{𝒯} end
struct MoistIdealGas{𝒯} <: AbstractEquationOfState{𝒯} end

@inline function calc_pressure(::BarotropicFluid, state, aux, params)
    ρ  = state.ρ
    cₛ = params.cₛ
    ρₒ = params.ρₒ

    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_pressure(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    R_d = params.R_d

    T = calc_air_temperature(eos, state, aux, params)

    return ρ * R_d * T
end

@inline function calc_pressure(eos::MoistIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    R_m = calc_gas_constant(eos, state, params)
    T = calc_air_temperature(eos, state, aux, params)
    
    return ρ * R_m * T 
end

@inline function calc_linear_pressure(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = calc_γ(eos, state, params)

    cv_d = params.cv_d
    T_0  = params.T_0

    return (γ - 1) * (ρe - ρ * Φ + ρ * cv_d * T_0) 
end

@inline function calc_linear_pressure(eos::MoistIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    ρ_q_tot = state.ρq 
    ρ_q_liq = 0 # zero for now
    ρ_q_ice = 0 # zero for now
    Φ = aux.Φ

    T_0  = params.T_0 
    γ    = calc_γ(DryIdealGas{(:ρ, :ρu, :ρe)}(), state, params)
    cv_m = calc_cv(eos, state, params)

    ρ_e_latent = (ρ_q_tot - ρ_q_liq) * params.e_int_v0 - ρ_q_ice * (params.e_int_v0 + params.e_int_i0)
    
    return (γ - 1) * (ρe - ρ * Φ - ρ_e_latent + ρ * cv_m * T_0)
end

@inline function calc_sound_speed(::BarotropicFluid{(:ρ, :ρu)}, state, aux, params)
    ρ = state.ρ
    cₛ = params.cₛ 
    ρₒ = params.ρₒ
    
    return cₛ * sqrt(ρ / ρₒ) 
end

@inline function calc_sound_speed(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    γ  = calc_γ(eos, state, params)

    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_sound_speed(eos::MoistIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    γ  = calc_γ(eos, state, params)

    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(eos::DryIdealGas, state, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ
    γ = calc_γ(eos, state, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(::MoistIdealGas, state, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ
    γ = calc_γ(DryIdealGas{(:ρ, :ρu, :ρe)}(), state, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_air_temperature(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ

  T_0 = params.T_0
  cv_d = params.cv_d

  e_int = (ρe - ρu' * ρu / 2ρ - ρ * Φ) / ρ
  T = T_0 + e_int / cv_d

  return T
end

@inline function calc_air_temperature(eos::MoistIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
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

  cv_m = calc_cv(eos, state, params)
  
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

@inline calc_cp(::DryIdealGas, state, params) = params.cp_d

@inline function calc_cp(::MoistIdealGas, state, params)
    cp_d  = params.cp_d
    cp_v  = params.cp_v
    cp_l  = params.cp_l
    cp_i  = params.cp_i
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for now

    cp_m  = cp_d + (cp_v - cp_d) * q_tot + (cp_l - cp_v) * q_liq + (cp_i - cp_v) * q_ice 
    return cp_m
end

@inline calc_cv(::DryIdealGas, state, params) = params.cv_d 

@inline function calc_cv(::MoistIdealGas, state, params)
    cv_d  = params.cv_d
    cv_v  = params.cv_v
    cv_l  = params.cv_l
    cv_i  = params.cv_i
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for nov

    cv_m  = cv_d + (cv_v - cv_d) * q_tot + (cv_l - cv_v) * q_liq + (cv_i - cv_v) * q_ice
    return cv_m
end

@inline calc_gas_constant(::DryIdealGas, state, params) = params.R_d

@inline function calc_gas_constant(::MoistIdealGas, state, params)
    R_d = params.R_d
    molmass_ratio = params.molmass_ratio
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for nov

    R_m = R_d * (1 + (molmass_ratio - 1) * q_tot - molmass_ratio * (q_liq + q_ice))

    return R_m
end

@inline function calc_γ(eos::AbstractEquationOfState, state, params)
    cp = calc_cp(eos, state, params)
    cv = calc_cv(eos, state, params)
    γ  = cp/cv

    return γ
end

@inline function calc_saturation_vapor_pressure(T, params)
    pₜᵣ   = params.press_triple
    R_v   = params.R_v
    Tₜᵣ   = params.T_triple
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