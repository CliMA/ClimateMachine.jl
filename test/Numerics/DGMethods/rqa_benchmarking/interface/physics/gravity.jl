abstract type AbstractGravity <: AbstractTerm end

struct Gravity <: AbstractGravity end
struct Buoyancy{𝒯} <: AbstractGravity end
struct FluctuationGravity <: AbstractGravity end

@inline calc_component!(source, ::Nothing, state, _...) = nothing

@inline function calc_component!(source, ::Gravity, state, aux, physics)
    ρ  = state.ρ
    ∇Φ = aux.∇Φ
   
    source.ρu -= ρ * ∇Φ 

    nothing
end

@inline function calc_component!(source, ::Buoyancy{(:ρ, :ρu, :ρθ)}, state, aux, physics)
    ρθ = state.ρθ
    α = physics.parameters.α 
    g = physics.parameters.g
    orientation = physics.orientation

    k = vertical_unit_vector(orientation, aux)
        
    source.ρu -= -α * g * k * ρθ

    nothing
end

# FluctuationGravity Components
@inline calc_fluctuation_component!(source, _...) = nothing
@inline calc_component!(source, ::FluctuationGravity, _...) = nothing

@inline function calc_fluctuation_component!(source, ::FluctuationGravity, state_1, state_2, aux_1, aux_2)
        ρ_1, ρ_2 = state_1.ρ, state_2.ρ
        Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
        α = ave(ρ_1, ρ_2) * 0.5
        source.ρu -= α * (Φ_1 - Φ_2) * I
        
        nothing
end