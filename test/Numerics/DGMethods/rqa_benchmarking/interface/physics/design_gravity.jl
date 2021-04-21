abstract type AbstractTerm{𝒯} end
abstract type AbstractGravity{𝒯} <: AbstractTerm{𝒯} end

struct Gravity{𝒯} <: AbstractGravity{𝒯}
struct Buoyancy{𝒯} <: AbstractGravity{𝒯}

@inline calc_component!(source, ::Nothing, state, _...) = nothing

@inline function calc_component!(source, ::Gravity, state, aux, physics)
    ρ  = state.ρ
    ∇Φ = aux.∇Φ
   
    source.ρu -= ρ * ∇Φ 

    nothing
end

# really ρe should be ρeᵢₙₜ
@inline function calc_component!(source, ::Gravity{(:ρ, :ρu, :ρe)}, state, aux, physics)
    ρ  = state.ρ
    ρu = state.ρu
    ∇Φ = aux.∇Φ
   
    source.ρu -= ρ * ∇Φ 
    source.ρe -= ρu' * ∇Φ

    nothing
end

@inline function calc_component!(source, ::Buoyancy, state, aux, physics)
    ρθ = state.ρθ
    k = vertical_unit_vector(aux.orientation, aux)
    α = physics.params.α 
    g = physics.params.g
        
    source.ρu -= -α * g * k * ρθ

    nothing
end