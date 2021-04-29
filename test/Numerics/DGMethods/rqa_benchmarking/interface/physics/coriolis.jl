abstract type AbstractCoriolis  <: AbstractPhysicsComponent end

@Base.kwdef struct DeepShellCoriolis{FT} <: AbstractCoriolis
    Ω :: FT # s⁻¹
end

@Base.kwdef struct ThinShellCoriolis{FT} <: AbstractCoriolis
    Ω :: FT # s⁻¹
end

@Base.kwdef struct BetaPlaneCoriolis{FT} <: AbstractCoriolis
    f₀ :: FT # s⁻¹
    β  :: FT # s⁻¹m⁻¹
end

@inline calc_force!(source, ::Nothing, state, _...) = nothing

@inline function calc_force!(source, coriolis::DeepShellCoriolis, state, _...)
    Ω⃗  = @SVector [-0, -0, coriolis.Ω]
    ρu⃗ = state.ρu

    source.ρu -= 2Ω⃗ × ρu⃗

    return nothing
end

@inline function calc_force!(source, coriolis::ThinShellCoriolis, state, aux, orientation, _...)
    Ω⃗  = @SVector [-0, -0, coriolis.Ω]
    ρu⃗ = state.ρu

    k̂ = vertical_unit_vector(orientation, aux)

    source.ρu -= (2Ω⃗ ⋅ k̂) * (k̂ × ρu⃗)
    
    return nothing
end

@inline function calc_force!(source, coriolis::BetaPlaneCoriolis, state, aux, orientation, _...)
    f₀ = coriolis.f₀
    β  = coriolis.β
    ρu⃗ = state.ρu
    y  = aux.y

    f = f₀ + β * y
    k̂ = vertical_unit_vector(orientation, aux)
    
    source.ρu -= f * (k̂ × ρu⃗)

    return nothing
end