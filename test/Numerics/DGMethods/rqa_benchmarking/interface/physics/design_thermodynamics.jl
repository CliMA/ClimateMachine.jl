abstract type AbstractEquationOfState{𝒯} end

struct BarotropicFluid{𝒯} <: AbstractEquationOfState{𝒯} end
struct IdealGas{𝒯} <: AbstractEquationOfState{𝒯} end

@inline function pressure(::BarotropicFluid{(:ρ, :ρu)}, state, aux, params)
    ρ  = state.ρ
    cₛ = params.cₛ
    ρₒ = params.ρₒ

    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function pressure(::IdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = params.γ

    return (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

@inline function sound_speed(::BarotropicFluid{(:ρ, :ρu)}, state, aux, params)
    cₛ = params.cₛ 
    ρₒ = params.ρₒ
    ρ = state.ρ
    
    return cₛ * sqrt(ρ / ρₒ) 
end

@inline function sound_speed(eos::IdealGas{(:ρ, :ρu, :ρe)}, state, aux, params)
    ρ  = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ  = aux.Φ
    γ  = params.γ

    p  = calc_pressure(eos, state, aux, params)

    return sqrt(γ * p / ρ)
end