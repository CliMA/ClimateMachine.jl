abstract type AbstractCoriolis <: AbstractTerm end

struct DeepShellCoriolis <: AbstractCoriolis end
struct ThinShellCoriolis <: AbstractCoriolis end
struct BetaPlaneCoriolis <: AbstractCoriolis end

@inline calc_component!(source, ::Nothing, state, _...) = nothing
@inline calc_component!(source, ::AbstractTerm, _...) = nothing

@inline function calc_component!(source, ::DeepShellCoriolis, state, aux, physics)
    ρu = state.ρu

    Ω  = @SVector [-0, -0, physics.parameters.Ω]

    source.ρu -= 2Ω × ρu

    nothing
end

@inline function calc_component!(source, ::ThinShellCoriolis, state, aux, physics)
    ρu = state.ρu
    
    k  = vertical_unit_vector(physics.orientation, aux)
    Ω  = @SVector [-0, -0, physics.parameters.Ω]

    source.ρu -= (2Ω ⋅ k) * (k × ρu)
    
    nothing
end

@inline function calc_component!(source, ::BetaPlaneCoriolis, state, aux, physics)
    ρu = state.ρu
    y  = aux.y
    f₀ = physics.parameters.f₀
    β  = physics.parameters.β

    k  = vertical_unit_vector(physics.orientation, aux)

    f = f₀ + β * y
    
    source.ρu -= f * (k × ρu)

    nothing
end