# module Domains

export Interval, Periodic

import LinearAlgebra: ×
import Base: *, getindex

abstract type AbstractDomain end

struct IntervalDomain{S, B} <: AbstractDomain
    min::S
    max::S
    periodic::B

    function IntervalDomain(a, b, periodic)
        a, b = promote(a, b)

        return new{typeof(a), typeof(periodic)}(a, b, periodic)
    end
end

function Interval(a, b)
    return IntervalDomain(a, b, false)
end

function Periodic(a, b)
    return IntervalDomain(a, b, true)
end

function Base.show(io::IO, Ω::IntervalDomain)
    min = Ω.min
    max = Ω.max
    periodic = Ω.periodic

    printstyled(io, "[", color = 226)
    printstyled(io, "$min, $max", color = 7)
    if periodic
        printstyled(io, ")", color = 226)
    else
        printstyled(io, "]", color = 226)
    end

    return nothing
end

struct ProductDomain{T} <: AbstractDomain
    domains::T
end

function ×(a::IntervalDomain, b::IntervalDomain)
    return ProductDomain((a, b))
end

function ×(a::ProductDomain, b::IntervalDomain)
    return ProductDomain((a.domains..., b))
end

function ×(a::IntervalDomain, b::ProductDomain)
    return ProductDomain((a, b.domains...))
end

function *(a::AbstractDomain, b::AbstractDomain)
    return a × b
end

function Base.show(io::IO, Ω::ProductDomain)
    for (i, domain) in enumerate(Ω.domains)
        print(domain)

        if i != length(Ω.domains)
            printstyled(io, " × ", color = 118)
        end
    end

    return nothing
end


getindex(Ω::ProductDomain, i::Int) = Ω.domains[i]

ndims(Ω::IntervalDomain) = 1
ndims(Ω::ProductDomain) = +(ndims.(Ω.domains)...)

function periodicityof(Ω::ProductDomain)
    periodicity = ones(Bool, ndims(Ω))

    for i in 1:ndims(Ω)
        periodicity[i] = Ω[i].periodic
    end

    return Tuple(periodicity)
end


```
CubedSphere for GCM 
```

struct AtmosDomain{S} <: AbstractDomain
    radius::S
    height::S

    function AtmosDomain(; radius = nothing, height = nothing )
        radius, height = promote(radius, height)

        return new{typeof(radius)}(radius, height)
    end
end

struct OceanDomain{S} <: AbstractDomain
    radius::S
    depth::S

    function OceanDomain(; radius = nothing, depth = nothing )
        radius, height = promote(radius, depth)

        return new{typeof(radius)}(radius, depth)
    end
end


```
Sphere helper functions for GCM 
```
rad(x,y,z) = sqrt(x^2 + y^2 + z^2)
lat(x,y,z) = asin(z/rad(x,y,z)) # ϕ ∈ [-π/2, π/2] 
lon(x,y,z) = atan(y,x) # λ ∈ [-π, π) 

r̂ⁿᵒʳᵐ(x,y,z) = norm([x,y,z]) ≈ 0 ? 1 : norm([x, y, z])^(-1)
ϕ̂ⁿᵒʳᵐ(x,y,z) = norm([x,y,0]) ≈ 0 ? 1 : (norm([x, y, z]) * norm([x, y, 0]))^(-1)
λ̂ⁿᵒʳᵐ(x,y,z) = norm([x,y,0]) ≈ 0 ? 1 : norm([x, y, 0])^(-1)

r̂(x,y,z) = r̂ⁿᵒʳᵐ(x,y,z) * @SVector([x, y, z])
ϕ̂(x,y,z) = ϕ̂ⁿᵒʳᵐ(x,y,z) * @SVector [x*z, y*z, -(x^2 + y^2)]
λ̂(x,y,z) = λ̂ⁿᵒʳᵐ(x,y,z) * @SVector [-y, x, 0] 

# end of module
