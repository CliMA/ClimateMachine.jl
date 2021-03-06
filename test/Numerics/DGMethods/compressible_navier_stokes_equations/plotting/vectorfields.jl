import Base: getindex

abstract type AbstractRepresentation end

struct Cartesian <: AbstractRepresentation end
struct Spherical <: AbstractRepresentation end
struct Covariant <: AbstractRepresentation end
struct Contravariant <: AbstractRepresentation end

Base.@kwdef struct VectorField{S, T, C} <: AbstractField
    data::S
    grid::T
    representation::C = Cartesian()
end

function VectorField(ϕ::VectorField; representation = Cartesian())
    return VectorField(data = ϕ.data, grid = ϕ.grid, representation = representation)
end

function (ϕ::VectorField)(representation::AbstractRepresentation)
    return VectorField(ϕ, representation = representation)
end

function components(data, ::Cartesian)
    one = @sprintf("%0.2e", data[1])
    two = @sprintf("%0.2e", data[2])
    three = @sprintf("%0.2e", data[3])
    println(one, "x̂ +" ,two,"ŷ +",three, "ẑ ")
    return data
end

getindex(ϕ::VectorField, ijk, e; verbose = true) = components(getindex.(ϕ.data, ijk, e), ϕ.grid, ijk, e, ϕ.representation, verbose = verbose)

## Component Grabber

function convenienceprint(grid, ijk, e)
    print("location: ")
    x,y,z = getposition(grid, ijk, e)
    println("x=", @sprintf("%0.2e", x), ",y=", @sprintf("%0.2e", y) , ",z=", @sprintf("%0.2e", z), " " )
    print("field:     ")
end

function components(v⃗, grid, ijk, e, ::Cartesian; verbose = true)
    if verbose
        one = @sprintf("%0.2e", v⃗[1])
        two = @sprintf("%0.2e", v⃗[2])
        three = @sprintf("%0.2e", v⃗[3])
        convenienceprint(grid, ijk, e)
        println(one, "x̂ +" ,two,"ŷ +",three, "ẑ ")
    end
    return v⃗
end

function components(v⃗, grid, ijk, e, ::Covariant; verbose = true)
    v⃗¹, v⃗², v⃗³ = getcontravariant(grid, ijk, e)
    v₁ = dot(v⃗¹,v⃗)
    v₂ = dot(v⃗²,v⃗)  
    v₃ = dot(v⃗³,v⃗)
    if verbose 
        one = @sprintf("%0.2e", v₁)
        two = @sprintf("%0.2e", v₂)
        three = @sprintf("%0.2e", v₃)
        convenienceprint(grid, ijk, e)
        println(one, "v⃗¹ +" ,two,"v⃗² +",three, "v⃗³ ")
    end
    return (; v₁, v₂, v₃)
end

function components(v⃗, grid, ijk, e, ::Contravariant; verbose = true)
    v⃗₁, v⃗₂, v⃗₃ = getcovariant(grid, ijk, e)
    v¹ = dot(v⃗₁,v⃗)
    v² = dot(v⃗₂,v⃗)  
    v³ = dot(v⃗₃,v⃗) 
    if verbose
        one = @sprintf("%0.2e", v¹)
        two = @sprintf("%0.2e", v²)
        three = @sprintf("%0.2e", v³)
        convenienceprint(grid, ijk, e)
        println(one, "v⃗₁ +" ,two,"v⃗₂ +",three, "v⃗₃ ")
    end
    return (; v¹, v², v³)
end

function components(v⃗, grid, ijk, e, ::Spherical; verbose = true)
    r̂, θ̂, φ̂ = getspherical(grid, ijk, e)
    vʳ = dot(r̂,v⃗)
    vᶿ = dot(θ̂,v⃗)  
    vᵠ = dot(φ̂,v⃗) 
    if verbose
        one = @sprintf("%0.2e", vʳ)
        two = @sprintf("%0.2e", vᶿ)
        three = @sprintf("%0.2e", vᵠ)
        convenienceprint(grid, ijk, e)
        println(one, "r̂ +" ,two,"θ̂ +",three, "φ̂ ")
    end
    return (; vʳ, vᶿ, vᵠ)
end

## Helper functions
function getjacobian(grid, ijk, e)
    ξ1x1 = grid.vgeo[ijk, grid.ξ1x1id, e]
    ξ1x2 = grid.vgeo[ijk, grid.ξ1x2id, e]
    ξ1x3 = grid.vgeo[ijk, grid.ξ1x3id, e]

    ξ2x1 = grid.vgeo[ijk, grid.ξ2x1id, e]
    ξ2x2 = grid.vgeo[ijk, grid.ξ2x2id, e]
    ξ2x3 = grid.vgeo[ijk, grid.ξ2x3id, e]

    ξ3x1 = grid.vgeo[ijk, grid.ξ3x1id, e]
    ξ3x2 = grid.vgeo[ijk, grid.ξ3x2id, e]
    ξ3x3 = grid.vgeo[ijk, grid.ξ3x3id, e]

    J = [ξ1x1 ξ1x2 ξ1x3;
         ξ2x1 ξ2x2 ξ2x3;
         ξ3x1 ξ3x2 ξ3x3;  
    ]
    # rows are the contravariant vectors
    # columns of the inverse are the covariant vectors
    return J
end

function getcontravariant(grid, ijk, e)
    J = getjacobian(grid, ijk, e)
    a⃗¹ = J[1,:] 
    a⃗² = J[2,:] 
    a⃗³ = J[3,:] 
    return (; a⃗¹, a⃗², a⃗³)
end

function getcovariant(grid, ijk, e)
    J = inv(getjacobian(grid, ijk, e))
    a⃗₁ = J[:, 1] 
    a⃗₂ = J[:, 2] 
    a⃗₃ = J[:, 3] 
    return (; a⃗₁, a⃗₂, a⃗₃)
end

function getspherical(grid, ijk, e)
    x,y,z = getposition(grid, ijk, e)
    r̂ = [x, y, z] ./ norm([x, y, z])
    θ̂ = [x*z, y*z, -(x^2 + y^2)] ./ ( norm([x, y, z]) * norm([x, y, 0]))
    φ̂ = [-y, x, 0] ./ norm([x, y, 0])
    return (; r̂, θ̂, φ̂)
end

function getposition(grid, ijk, e)
    x1 = grid.vgeo[ijk, grid.x1id, e]
    x2 = grid.vgeo[ijk, grid.x2id, e]
    x3 = grid.vgeo[ijk, grid.x3id, e]
    r = [x1 x2 x3]
    return r
end

function constructdeterminant(grid)
    M = grid.vgeo[:, grid.Mid, :]

    ωx = reshape(grid.ω[1], (length(grid.ω[1]), 1, 1, 1))
    ωy = reshape(grid.ω[2], (1, length(grid.ω[2]), 1, 1))
    ωz = reshape(grid.ω[3], (1, 1, length(grid.ω[3]), 1))
    ω = reshape(ωx .* ωy .* ωz, (size(M)[1],1) )
    J = M ./ ω
    return J
end

function getdetjacobian(grid, ijk, e)
    M = grid.vgeo[ijk, grid.Mid, e]
end
