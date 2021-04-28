abstract type AbstractGravity  <: AbstractPhysicsComponent end

@Base.kwdef struct DeepShellGravity{FT} <: AbstractGravity
    g :: FT # ms⁻²
    a :: FT # m
end

@Base.kwdef struct ThinShellGravity{FT} <: AbstractGravity
    g :: FT # ms⁻²
end

@Base.kwdef struct Buoyancy{FT} <: AbstractGravity
    α :: FT # K⁻¹
    g :: FT # ms⁻²
end

@inline calc_force!(source, ::Nothing, state, _...) = nothing

@inline function calc_force!(source, gravity::DeepShellGravity, state, aux, orientation, _...)
    g = gravity.g
    a = gravity.a
    ρ = state.ρ
    r⃗ = @SVector [aux.x, aux.y, aux.z]
    k̂ = vertical_unit_vector(orientation, aux)

    r = norm(r⃗)
   
    # TODO!: Need numerical gradient of geopotential
    source.ρu -= g * (a / r)^2 * k̂ * ρ

    return nothing
end

@inline function calc_force!(source, gravity::ThinShellGravity, state, aux, orientation,_...)
    g = gravity.g
    ρ = state.ρ
    k̂ = vertical_unit_vector(orientation, aux)
    
    source.ρu -= g * k̂ * ρ

    return nothing
end

@inline function calc_force!(source, gravity::Buoyancy, state, aux, orientation,_...)
    α = gravity.α 
    g = gravity.g
    ρθ = state.ρθ
    k̂ = vertical_unit_vector(orientation, aux)
        
    source.ρu -= -α * g * k̂ * ρθ

    return nothing
end

"""
    Maciek's world
"""
struct Gravity end
function source!(m::DryAtmosModel, ::Gravity, source, state, aux)
    ∇Φ = aux.∇Φ
    if !fluctuation_gravity
        source.ρu -= state.ρ * ∇Φ
    end
    if !total_energy
        source.ρe -= state.ρu' * ∇Φ
    end
end