export SimpleBoxProblem, HomogeneousBox, OceanGyre

############################
# Basic box problem        #
# Set up dimensions of box #
############################
abstract type AbstractSimpleBoxProblem <: AbstractHydrostaticBoussinesqProblem end

"""
    SimpleBoxProblem <: AbstractSimpleBoxProblem

Stub structure with the dimensions of the box.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
"""
struct SimpleBox{T, BC} <: AbstractSimpleBoxProblem
    Lˣ::T
    Lʸ::T
    H::T
    boundary_condition::BC
    function SimpleBox{FT}(
        Lˣ,
        Lʸ,
        H;
        BC = (
            CoastlineNoSlip(),
            OceanFloorNoSlip(),
            OceanSurfaceNoStressNoForcing(),
        ),
    ) where {FT <: AbstractFloat}
        return new{FT, typeof(BC)}(Lˣ, Lʸ, H, BC)
    end
end

"""
    ocean_init_aux!(::HBModel, ::AbstractSimpleBoxProblem)

save y coordinate for computing coriolis, wind stress, and sea surface temperature

# Arguments
- `m`: model object to dispatch on and get viscosities and diffusivities
- `p`: problem object to dispatch on and get additional parameters
- `A`: auxiliary state vector
- `geom`: geometry stuff
"""
function ocean_init_aux!(m::HBModel, p::AbstractSimpleBoxProblem, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]

    return nothing
end

##########################
# Homogenous wind stress #
# Constant temperature   #
##########################

"""
    HomogeneousBox <: AbstractSimpleBoxProblem

Container structure for a simple box problem with wind-stress and coriolis force.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
τₒ = maximum value of wind-stress (amplitude)
fₒ = first coriolis parameter (constant term)
β  = second coriolis parameter (linear term)
"""
struct HomogeneousBox{T, BC} <: AbstractSimpleBoxProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    boundary_condition::BC
    function HomogeneousBox{FT}(
        Lˣ,
        Lʸ,
        H;
        τₒ = FT(1e-1),  # (m/s)^2
        BC = (
            CoastlineNoSlip(),
            OceanFloorNoSlip(),
            OceanSurfaceStressNoForcing(),
        ),
    ) where {FT <: AbstractFloat}
        return new{FT, typeof(BC)}(Lˣ, Lʸ, H, τₒ, BC)
    end
end

"""
    ocean_init_state!(::HomogeneousBox)

initialize u,v with random values, η with 0, and θ with a constant (20)

# Arguments
- `p`: HomogeneousBox problem object, used to dispatch on
- `Q`: state vector
- `A`: auxiliary state vector, not used
- `coords`: the coordidinates, not used
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(p::HomogeneousBox, Q, A, coords, t)
    Q.u = @SVector [rand(), rand()]
    Q.η = 0
    Q.θ = 20

    return nothing
end

"""
    kinematic_stress(::HomogeneousBox)

jet stream like windstress

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
"""
@inline kinematic_stress(p::HomogeneousBox, y, ρ) =
    -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

##########################
# Homogenous wind stress #
# Temperature forcing    #
##########################

"""
    OceanGyre <: AbstractSimpleBoxProblem

Container structure for a simple box problem with wind-stress, coriolis force, and temperature forcing.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
τₒ = maximum value of wind-stress (amplitude)
λʳ = temperature relaxation penetration constant (meters / second)
θᴱ = maximum surface temperature
"""
struct OceanGyre{T, BC} <: AbstractSimpleBoxProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    λʳ::T
    θᴱ::T
    boundary_condition::BC
    function OceanGyre{FT}(
        Lˣ,
        Lʸ,
        H;
        τₒ = FT(1e-1),       # (m/s)^2
        λʳ = FT(4 // 86400), # m / s
        θᴱ = FT(10),         # K
        BC = (
            CoastlineNoSlip(),
            OceanFloorNoSlip(),
            OceanSurfaceStressForcing(),
        ),
    ) where {FT <: AbstractFloat}
        return new{FT, typeof(BC)}(Lˣ, Lʸ, H, τₒ, λʳ, θᴱ, BC)
    end
end

"""
    ocean_init_state!(::OceanGyre)

initialize u,v,η with 0 and θ linearly distributed between 9 at z=0 and 1 at z=H

# Arguments
- `p`: OceanGyre problem object, used to dispatch on and obtain ocean height H
- `Q`: state vector
- `A`: auxiliary state vector, not used
- `coords`: the coordidinates
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(p::OceanGyre, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [0, 0]
    Q.η = 0
    Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    return nothing
end

"""
    kinematic_stress(::OceanGyre)

jet stream like windstress

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
"""
@inline kinematic_stress(p::OceanGyre, y, ρ) = -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

"""
    temperature_flux(::OceanGyre)

cool-warm north-south linear temperature gradient

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
- `θ`: temperature within element on boundary
"""
@inline function temperature_flux(p::OceanGyre, y, θ)
    θʳ = p.θᴱ * (1 - y / p.Lʸ)
    return p.λʳ * (θʳ - θ)
end
