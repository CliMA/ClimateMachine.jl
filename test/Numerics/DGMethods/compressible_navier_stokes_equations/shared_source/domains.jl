import Base: getindex, *, ndims, length, ^
import LinearAlgebra: ×

abstract type AbstractDomain end
abstract type AbstractBoundary end

struct DomainBoundary <: AbstractBoundary
    closure::Any
end

struct PointDomain{S} <: AbstractDomain
    point::S
end

struct IntervalDomain{AT, BT, PT} <: AbstractDomain
    min::AT
    max::BT
    periodic::PT
end

function IntervalDomain(min, max; periodic = false)
    @assert min < max
    return IntervalDomain(min, max, periodic)
end

function Periodic(min, max)
    @assert min < max
    return IntervalDomain(min, max, periodic = true)
end

S¹ = Periodic

function Interval(min, max)
    @assert min < max
    return IntervalDomain(min, max)
end

function Periodic(min, max)
    @assert min < max
    return IntervalDomain(min, max, periodic = true)
end

function Base.show(io::IO, Ω::IntervalDomain)
    min = Ω.min
    max = Ω.max
    printstyled(io, "[", color = 226)
    astring = @sprintf("%0.2f", min)
    bstring = @sprintf("%0.2f", max)
    printstyled(astring, ", ", bstring, color = 7)
    # printstyled("$min, $max", color = 7)
    Ω.periodic ? printstyled(io, ")", color = 226) :
    printstyled(io, "]", color = 226)
end

function Base.show(io::IO, o::PointDomain)
    printstyled("{", o.point, "}", color = 201)
end

# Product Domains
struct ProductDomain{DT} <: AbstractDomain
    domains::DT
end

function Base.show(io::IO, Ω::ProductDomain)
    for (i, domain) in enumerate(Ω.domains)
        print(domain)
        if i != length(Ω.domains)
            printstyled(io, "×", color = 118)
        end
    end
end

ndims(p::PointDomain) = 0
ndims(Ω::IntervalDomain) = 1
ndims(Ω::ProductDomain) = +(ndims.(Ω.domains)...)

length(Ω::IntervalDomain) = Ω.max - Ω.min
length(Ω::ProductDomain) = length.(Ω.domains)

×(arg1::AbstractDomain, arg2::AbstractDomain) = ProductDomain((arg1, arg2))
×(args::ProductDomain, arg2::AbstractDomain) =
    ProductDomain((args.domains..., arg2))
×(arg1::AbstractDomain, args::ProductDomain) =
    ProductDomain((arg1, args.domains...))
×(arg1::ProductDomain, args::ProductDomain) =
    ProductDomain((arg1.domains..., args.domains...))
×(args::AbstractDomain) = ProductDomain(args...)
*(arg1::AbstractDomain, arg2::AbstractDomain) = arg1 × arg2

function ^(Ω::IntervalDomain, T::Int)
    Ωᵀ = Ω
    for i in 1:(T - 1)
        Ωᵀ *= Ω
    end
    return Ωᵀ
end

function info(Ω::ProductDomain)
    println("This is a ", ndims(Ω), "-dimensional tensor product domain.")
    print("The domain is ")
    println(Ω, ".")
    for (i, domain) in enumerate(Ω.domains)
        domain_string = domain.periodic ? "periodic" : "wall-bounded"
        length = @sprintf("%.2f ", domain.max - domain.min)
        println(
            "The dimension $i domain is ",
            domain_string,
            " with length ≈ ",
            length,
        )
    end
    return nothing
end

function isperiodic(Ω::ProductDomain)
    max = [Ω.domains[i].periodic for i in eachindex(Ω.domains)]
    return prod(max)
end

function periodicityof(Ω::ProductDomain)
    periodicity = ones(Bool, ndims(Ω))
    for i in 1:ndims(Ω)
        periodicity[i] = Ω[i].periodic
    end
    return Tuple(periodicity)
end

getindex(Ω::ProductDomain, i::Int) = Ω.domains[i]

# Boundaries
struct Boundaries{S}
    boundaries::S
end

getindex(∂Ω::Boundaries, i) = ∂Ω.boundaries[i]

function Base.show(io::IO, ∂Ω::Boundaries)
    for (i, boundary) in enumerate(∂Ω.boundaries)
        printstyled("boundary ", i, ": ", color = 13)
        println(boundary)
    end
end

function ∂(Ω::IntervalDomain)
    if Ω.periodic
        return (nothing)
    else
        return Boundaries((PointDomain(Ω.min), PointDomain(Ω.max)))
    end
    return nothing
end

function ∂(Ω::ProductDomain)
    ∂Ω = []
    for domain in Ω.domains
        push!(∂Ω, ∂(domain))
    end
    splitb = []
    for (i, boundary) in enumerate(∂Ω)
        tmp = Any[]
        push!(tmp, Ω.domains...)
        if boundary != nothing
            tmp1 = copy(tmp)
            tmp2 = copy(tmp)
            tmp1[i] = boundary[1]
            push!(splitb, ProductDomain(Tuple(tmp1)))
            tmp2[i] = boundary[2]
            push!(splitb, ProductDomain(Tuple(tmp2)))
        end
    end
    return Boundaries(Tuple(splitb))
end

Base.@kwdef struct SphericalShellDomain{ℛ, ℋ, 𝒟} <: AbstractDomain
    radius::ℛ = 6378e3
    height::ℋ = 30e3
    depth::𝒟 = 3e3
end


length(Ω::SphericalShellDomain) = (Ω.radius, Ω.radius, Ω.height - Ω.depth)
ndims(Ω::SphericalShellDomain) = 3

function AtmosDomain(; radius = 6378e3, height = 30e3)
    depth = 0
    return SphericalShellDomain(; radius, height, depth)
end

function OceanDomain(; radius = 6378e3, depth = 3e3)
    height = 0
    return SphericalShellDomain(; radius, height, depth)
end
