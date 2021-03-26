abstract type Gravity <: Forcing end

@Base.kwdef struct DeepShellGravity{FT} <: Gravity
    g :: FT # ms⁻²
    a :: FT # m
end

@Base.kwdef struct ThinShellGravity{FT} <: Gravity
    g :: FT # ms⁻²
end

@Base.kwdef struct Buoyancy{FT} <: Gravity
    α :: FT # K⁻¹
    g :: FT # ms⁻²
end

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
