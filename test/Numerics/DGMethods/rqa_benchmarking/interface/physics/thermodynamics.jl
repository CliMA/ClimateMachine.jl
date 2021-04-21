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

# Energy Prognostic Cases, perhaps abstract type 

Base.@kwdef struct TotalEnergy{FT} <: AbstractEquationOfState
    γ  :: FT
end

Base.@kwdef struct DryEuler{FT} <: AbstractEquationOfState
    γ  :: FT
end

Base.@kwdef struct LinearizedTotalEnergy{FT} <: AbstractEquationOfState
    γ  :: FT
end

Base.@kwdef struct LinearizedDryEuler{FT} <: AbstractEquationOfState
    γ  :: FT
end

# Linearized Equation of States
linearize(eos::TotalEnergy) = LinearizedTotalEnergy(eos.γ)
linearize(eos::DryEuler) = LinearizedDryEuler(eos.γ)

"""
  Thermodynamic relationships
"""

@inline function calc_pressure(eos::BarotropicFluid, state) 
    cₛ = eos.cₛ 
    ρₒ = eos.ρₒ
    ρ = state.ρ
    
    return (cₛ * ρ)^2 / (2 * ρₒ)
end

@inline function calc_sound_speed(eos::BarotropicFluid, state)
    cₛ = eos.cₛ 
    ρₒ = eos.ρₒ
    ρ = state.ρ
    
    return cₛ * sqrt(ρ / ρₒ) 
end

@inline function calc_pressure(eos::DryIdealGas, state)
    R = eos.R
    pₒ = eos.pₒ
    γ = eos.γ
    ρθ = state.ρθ

    return pₒ * (R / pₒ * ρθ)^γ
end

@inline function calc_sound_speed(eos::DryIdealGas, state)
    R = eos.R
    pₒ = eos.pₒ
    γ = eos.γ
    ρ = state.ρ

    return sqrt(γ * calc_pressure(eos, state) / ρ)
end

"""
Modified Maciek's world
"""
@inline function calc_pressure(eos::TotalEnergy, state, aux)
    γ  = eos.γ
    Φ  = aux.Φ
    ρ  = state.ρ
    ρe = state.ρe
    ρu = state.ρu
    return (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

@inline function calc_sound_speed(eos::TotalEnergy, state, aux)
    γ = eos.γ
    ρ = state.ρ
    return sqrt(γ * calc_pressure(eos, state, aux) / ρ)
end

@inline function calc_pressure(eos::LinearizedTotalEnergy, state, aux)
    γ  = eos.γ
    Φ  = aux.Φ
    ρ  = state.ρ
    ρe = state.ρe
    return (γ - 1) * (ρe - ρ * Φ)
end

@inline function calc_sound_speed(eos::LinearizedTotalEnergy, state, aux)
    γ = eos.γ
    ρ = state.ρ
    return sqrt(γ * calc_pressure(eos, state, aux) / ρ)
end

@inline function calc_pressure(eos::DryEuler, state, aux)
    γ  = eos.γ
    ρ  = state.ρ
    ρe = state.ρe
    ρu = state.ρu
    return (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ)
end

@inline function calc_sound_speed(eos::DryEuler, state, aux)
    γ = eos.γ
    ρ = state.ρ
    return sqrt(γ * calc_pressure(eos, state, aux) / ρ)
end

@inline function calc_pressure(eos::LinearizedDryEuler, state, aux)
    γ  = eos.γ
    ρe = state.ρe
    return (γ - 1) * ρe
end

@inline function calc_sound_speed(eos::LinearizedDryEuler, state, aux)
    γ = eos.γ
    ρ = state.ρ
    return sqrt(γ * calc_pressure(eos, state, aux) / ρ)
end

"""
  Maciek's world
"""
function pressure(ρ, ρu, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    if total_energy
        (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
    else
        (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ)
    end
end

function totalenergy(ρ, ρu, p, Φ)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    if total_energy
        return p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ
    else
        return p / (γ - 1) + dot(ρu, ρu) / 2ρ
    end
end

function soundspeed(ρ, p)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    sqrt(γ * p / ρ)
end

@inline function linearized_pressure(ρ, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    if total_energy
        (γ - 1) * (ρe - ρ * Φ)
    else
        (γ - 1) * ρe
    end
end

"""
  Base extensions
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
