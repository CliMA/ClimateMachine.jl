struct PressureDivergence <: AbstractTerm end
struct LinearPressureDivergence <: AbstractTerm end

@inline calc_component!(flux, ::Nothing, _...) = nothing
@inline calc_component!(flux, ::AbstractTerm, _...) = nothing

@inline function calc_component!(flux, ::PressureDivergence, state, aux, physics)
    eos = physics.eos
    parameters = physics.parameters
    
    p = calc_pressure(eos, state, aux, parameters)

    flux.ρu += p * I

    nothing
end

@inline function calc_component!(flux, ::LinearPressureDivergence, state, aux, physics)
    eos = physics.eos
    parameters = physics.parameters

    p = calc_linear_pressure(eos, state, aux, parameters)

    flux.ρu += p * I

    nothing
end
