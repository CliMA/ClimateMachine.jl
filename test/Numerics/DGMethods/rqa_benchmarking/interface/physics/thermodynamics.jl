abstract type AbstractEquationOfState  <: AbstractPhysicsComponent end

Base.@kwdef struct BarotropicFluid{FT} <: AbstractEquationOfState
    ρₒ :: FT
    cₛ :: FT
end

Base.@kwdef struct DryIdealGas{FT} <: AbstractEquationOfState
    R  :: FT
    pₒ :: FT
    γ  :: FT
end

"""
  Thermodynamic relationships
"""
@inline function calc_pressure(eos::BarotropicFluid, state) 
    cₛ = eos.cₛ 
    ρₒ = eos.ρₒ
    ρ = state.ρ
    
    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_pressure(eos::DryIdealGas, state)
    R = eos.R
    pₒ = eos.pₒ
    γ = eos.γ
    ρθ = state.ρθ

    return pₒ * (R / pₒ * ρθ)^γ
end

@inline function calc_sound_speed(eos::BarotropicFluid, state)
    cₛ = eos.cₛ 
    ρₒ = eos.ρₒ
    ρ = state.ρ
    
    return cₛ * sqrt(ρ / ρₒ) 
end

@inline function calc_sound_speed(eos::DryIdealGas, state)
    R = eos.R
    pₒ = eos.pₒ
    γ = eos.γ
    ρ = state.ρ

    return sqrt(γ * calc_pressure(eos, state) / ρ)
end

"""
  Extensions
"""
function info(::AbstractEquationOfState)
    error("Not implemented!")
end

function info(::BarotropicFluid)
    println("The equation of state (eos) is:")
    printstyled("eos = (cₛ ρ)^2 / (2 ρₒ) \n", color = 82)
    println("The sound speed is:")
    printstyled("soundspeed = cₛ sqrt(ρ / ρₒ)  \n", color = 82)
    println("ρ : density ")
    println("cₛ: reference soundspeed ")
    println("ρₒ: reference density ")
end

function info(::DryIdealGas)
    error("Not implemented!")
end
