abstract type AbstractEquationOfState{𝒯} end

struct BarotropicFluid{𝒯} <: AbstractEquationOfState{𝒯} end
struct DryIdealGas{𝒯} <: AbstractEquationOfState{𝒯} end
struct MoistIdealGas{𝒯} <: AbstractEquationOfState{𝒯} end

@inline function calc_pressure(::BarotropicFluid{(:ρ, :ρu)}, state, aux, params)
    ρ  = state.ρ
    cₛ = params.cₛ
    ρₒ = params.ρₒ

    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_pressure(eos::DryIdealGas{(:ρ, :ρu, :ρθ)}, state, aux, params)
    ρθ  = state.ρθ
    R_d = params.R_d
    pₒ  = params.pₒ
    γ   = calc_γ(eos, state, params)

    return pₒ * (R_d / pₒ * ρθ)^γ
end

@inline function calc_pressure(eos::MoistIdealGas{(:ρ, :ρu, :ρθ)}, state, aux, params)
    # TODO: θ = T (p/pₒ)^(R/cₚ) is not conserved when there is phase transition, latent heat source needed
    ρθ  = state.ρθ
    R   = calc_R(eos, state, params)
    pₒ  = params.pₒ
    γ   = calc_γ(eos, state, params)

    return pₒ * (R / pₒ * ρθ)^γ
end

@inline function calc_pressure(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = calc_γ(eos, state, params)

    return (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

@inline function calc_pressure(eos::MoistIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = calc_γ(eos, state, params)

    return (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

@inline function calc_linear_pressure(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = calc_γ(eos, state, params)

    return (γ - 1) * (ρe - ρ * Φ) 
end

@inline function calc_linear_pressure(eos::MoistIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = calc_γ(eos, state, params)

    return (γ - 1) * (ρe - ρ * Φ) 
end

@inline function calc_sound_speed(::BarotropicFluid{(:ρ, :ρu)}, state, aux, params)
    ρ = state.ρ
    cₛ = params.cₛ 
    ρₒ = params.ρₒ
    
    return cₛ * sqrt(ρ / ρₒ) 
end

@inline function calc_sound_speed(eos::DryIdealGas{(:ρ, :ρu, :ρθ)}, state, aux, params)
    ρ   = state.ρ
    γ   = calc_γ(eos, state, params)

    p   = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_sound_speed(eos::MoistIdealGas{(:ρ, :ρu, :ρθ)}, state, aux, params)
    ρ   = state.ρ
    γ   = calc_γ(eos, state, params)

    p   = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
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

@inline function calc_ref_sound_speed(::DryIdealGas, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ
    γ = calc_γ(eos, state, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(::MoistIdealGas, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ
    γ = calc_γ(eos, state, params)

    return sqrt(γ * p / ρ)
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
    cp_l  = params.cp_v
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
    cv_l  = params.cv_v
    cv_i  = params.cv_i
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for nov

    cv_m  = cv_d + (cv_v - cv_d) * q_tot + (cv_l - cv_v) * q_liq + (cv_i - cv_v) * q_ice
    return cv_m
end

@inline calc_R(::DryIdealGas, state, params) = params.R_d

@inline function calc_R(::MoistIdealGas, state, params)
    R_d = params.R_d
    molmass_ratio = params.molmass_ratio
    q_tot = state.ρq / state.ρ
    q_liq = 0 # zero for now
    q_ice = 0 # zero for nov

    R_m = R_d * (1 + (molmass_ratio - 1) * q_tot - molmass_ratio * (q_liq + q_ice))

end

@inline function calc_γ(eos::AbstractEquationOfState, state, params)
    cp = calc_cv(eos, state, params)
    cv = calc_cv(eos, state, params)
    γ  = cp/cv

    return γ
end