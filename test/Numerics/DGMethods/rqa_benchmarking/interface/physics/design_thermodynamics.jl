abstract type AbstractEquationOfState{𝒯} end

struct BarotropicFluid{𝒯} <: AbstractEquationOfState{𝒯} end
struct DryIdealGas{𝒯} <: AbstractEquationOfState{𝒯} end

@inline function calc_pressure(::BarotropicFluid{(:ρ, :ρu)}, state, aux, params)
    ρ  = state.ρ
    cₛ = params.cₛ
    ρₒ = params.ρₒ

    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_pressure(::DryIdealGas{(:ρ, :ρu, :ρθ)}, state, aux, params)
    ρθ  = state.ρθ
    R_d = params.R_d
    pₒ  = params.pₒ
    γ   = params.γ

    return pₒ * (R_d / pₒ * ρθ)^γ
end

@inline function calc_pressure(::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = params.γ

    return (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

@inline function calc_linear_pressure(::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = params.γ

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
    γ   = params.γ

    p   = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_sound_speed(eos::DryIdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    γ  = params.γ

    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end

@inline function calc_ref_sound_speed(::DryIdealGas, aux, params)
    p = aux.ref_state.p
    ρ = aux.ref_state.ρ
    γ = params.γ

    return sqrt(γ * p / ρ)
end

@inline function calc_total_specific_enthalpy(eos::DryIdealGas, state, aux, params)
    ρ  = state.ρ
    ρe = state.ρe

    p  = calc_pressure(eos, state, aux, params)

    return (ρe + p) / ρ
end