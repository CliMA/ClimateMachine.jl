abstract type AbstractPressureForce <: AbstractPhysicsComponent end

@Base.kwdef struct PressureDivergence{T} <: AbstractPressureForce
    eos::T
end

@inline calc_flux!(flux, ::Nothing, state, _...) = nothing
@inline calc_flux!(flux, ::AbstractPressureForce, state, _...) = nothing

@inline function calc_flux!(flux, pressure::PressureDivergence, state, aux, _...)
    eos = pressure.eos
    p = calc_pressure(eos, state, aux)

    flux.ρu += p * I

    return nothing
end