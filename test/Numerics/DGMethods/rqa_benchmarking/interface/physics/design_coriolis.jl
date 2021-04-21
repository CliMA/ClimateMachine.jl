abstract type AbstractTerm{𝒯} end
abstract type AbstractCoriolis{𝒯} <: AbstractTerm{𝒯} end

struct DeepShellCoriolis <: AbstractCoriolis{𝒯} end
struct ThinShellCoriolis <: AbstractCoriolis{𝒯} end
struct BetaPlaneCoriolis <: AbstractCoriolis{𝒯} end

@inline calc_component!(source, ::Nothing, state, _...) = nothing
@inline calc_component!(source, ::AbstractTerm, _...) = nothing

@inline function calc_component!(source, ::DeepShellCoriolis, state, aux, physics)
    ρu = state.ρu
    Ω  = @SVector [-0, -0, physics.params.Ω]

    source.ρu -= 2Ω × ρu

    nothing
end

@inline function calc_component!(source, ::ThinShellCoriolis, state, aux, physics)
    ρu = state.ρu
    k  = vertical_unit_vector(aux.orientation, aux)
    Ω  = @SVector [-0, -0, physics.params.Ω]

    source.ρu -= (2Ω ⋅ k) * (k × ρu)
    
    nothing
end

@inline function calc_component!(source, ::BetaPlaneCoriolis, state, aux, physics)
    ρu = state.ρu
    y  = aux.y
    k  = vertical_unit_vector(aux.orientation, aux)
    f₀ = physics.params.f₀
    β  = physics.params.β

    f = f₀ + β * y
    
    source.ρu -= f * (k × ρu)

    nothing
end