abstract type AbstractGravity <: AbstractTerm end

struct Gravity <: AbstractGravity end
struct Buoyancy{𝒯} <: AbstractGravity end

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