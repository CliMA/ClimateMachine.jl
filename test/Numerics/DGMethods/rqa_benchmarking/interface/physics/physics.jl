abstract type AbstractPhysicsComponent end
abstract type AbstractTerm{𝒯} end

"""
    Physics

    Essentially a NamedTuple, but one that returns `nothing` by default.
    Example:
        physics = Physics(diffusion = NiceDiffusion())
        `physics.diffusion` returns NiceDiffusion()
        `physics.coriolis` returns nothing

    With great power comes great responsibility:
        Great for flexibility, just like NamedTuple, but potential
        errors pass silently (e.g, typos in keywords)
"""
struct Physics{𝒯} <: AbstractPhysicsComponent
    components::𝒯
end

function Physics(; kwargs...)
    components = (; collect(kwargs)...)
    return Physics(components)
end

"""
    Base module extensions
"""
function Base.getproperty(value::Physics, name::Symbol)
    # Physics default for everything is `nothing`
    components = getfield(value, :components, false)
    return get(components, name, nothing)
end

function Base.propertynames(value::Physics)
    components = getfield(value, :components, false)
    return keys(components)
end

function Base.show(io::IO, physics::Physics)
    color = :white
    printstyled(io, "Physics(\n", color = color)
    for name in propertynames(physics)
        property = getproperty(physics, name)
        print("  ")
        printstyled(property, color = Int(rand(UInt8)))
        print("\n")
    end
    printstyled(io, ")", color = color)
end