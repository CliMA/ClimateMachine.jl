using Printf

import Base: getindex, ndims, length, *
import LinearAlgebra: ×

abstract type AbstractDomain end

"""
    IntervalDomain
"""
struct IntervalDomain{T} <: AbstractDomain
    min::T
    max::T
    periodic::Bool
end

function IntervalDomain(; min, max, periodic = false)
    @assert min < max
    return IntervalDomain(min, max, periodic)
end

"""
    ProductDomain
"""
struct ProductDomain{T} <: AbstractDomain
    domains::T
end

function ProductDomain(; domains)
    if length(domains) < 2
        error("A product domain needs more than one subdomain!")
    else
        return reduce(×, domains)
    end
end

"""
    SphericalShell
"""
struct SphericalShell{T} <: AbstractDomain
    radius::T
    height::T
end

function SphericalShell(; radius, height)
    @assert radius > 0 && height > 0
    return SphericalShell(radius, height)
end

"""
    Nary Operations
"""
×(domain1::AbstractDomain, domain2::AbstractDomain) = ProductDomain((domain1, domain2))
×(product::ProductDomain, domain::AbstractDomain) = ProductDomain((product.domains..., domain))
×(domain::AbstractDomain, product::ProductDomain)  = ProductDomain((domain, product.domains...))
×(product1::ProductDomain, product2::ProductDomain)  = ProductDomain((product1.domains..., product2.domains...))

"""
    Extensions
"""
Base.ndims(domain::IntervalDomain) = 1
Base.ndims(domain::ProductDomain)  = +(ndims.(domain.domains)...)
Base.ndims(domain::SphericalShell) = 3

Base.length(domain::IntervalDomain) = domain.max - domain.min
Base.length(domain::ProductDomain)  = length.(domain.domains)
Base.length(domain::SphericalShell) = (domain.radius, domain.radius, domain.height)

Base.getindex(domain::ProductDomain, i::Int) = domain.domains[i]

function Base.getproperty(value::ProductDomain, name::Symbol)
    if name == :periodicity
        periodicity = ones(Bool, ndims(value))
        for i in 1:ndims(value)
            periodicity[i] = value[i].periodic
        end
        return Tuple(periodicity)
    else
        # default
        return getfield(value, name)
    end
end

function Base.propertynames(::ProductDomain)
    return (:domains, :periodicity)
end

function Base.show(io::IO, domain::IntervalDomain)
    min = domain.min
    max = domain.max
    printstyled(io, "[", color = 226)
    astring = @sprintf("%0.2f", min)
    bstring = @sprintf("%0.2f", max)
    printstyled(astring, ", ", bstring, color = 7)
    domain.periodic ? printstyled(io, ")", color = 226) :
    printstyled(io, "]", color = 226)
end

function Base.show(io::IO, product::ProductDomain)
    for (i, domain) in enumerate(product.domains)
        print(domain)
        if i < ndims(product)
            printstyled(io, "×", color = 118)
        end
    end
end

function info(domain::ProductDomain)
    println("This is a ", ndims(domain), "-dimensional tensor product domain.")
    print("The domain is ")
    println(domain, ".")
    for (i, domain) in enumerate(domain.domains)
        domain_string = domain.periodic ? "periodic" : "wall-bounded"
        length = @sprintf("%.2f ", domain.max - domain.min)
        println(
            "The dimension $i domain is ",
            domain_string,
            " with length ≈ ",
            length,
        )
    end
end