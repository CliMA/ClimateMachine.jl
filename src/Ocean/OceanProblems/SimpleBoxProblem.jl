module OceanProblems

export SimpleBox, HomogeneousBox, OceanGyre

using StaticArrays
using CLIMAParameters.Planet: grav

using ...Problems

using ..HydrostaticBoussinesq
using ..ShallowWater

import ..Ocean:
    ocean_init_state!,
    ocean_init_aux!,
    kinematic_stress,
    surface_flux,
    coriolis_parameter

HBModel = HydrostaticBoussinesqModel
SWModel = ShallowWaterModel

abstract type AbstractOceanProblem <: AbstractProblem end

############################
# Basic box problem        #
# Set up dimensions of box #
############################
abstract type AbstractSimpleBoxProblem <: AbstractOceanProblem end

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

    # needed for proper CFL condition calculation
    A.w = 0
    A.pkin = 0
    A.wz0 = 0

    A.uᵈ = @SVector [-0, -0]
    A.ΔGᵘ = @SVector [-0, -0]

    return nothing

    return nothing
end

function ocean_init_aux!(m::SWModel, p::AbstractSimpleBoxProblem, A, geom)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.Δu = @SVector [-0, -0]

    return nothing
end

@inline coriolis_parameter(m::SWModel, p::AbstractSimpleBoxProblem, y) =
    m.fₒ + m.β * y

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
        Lˣ, # m
        Lʸ, # m
        H;  # m
        BC = (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), Insulating()),
        ),
    ) where {FT <: AbstractFloat}
        return new{FT, typeof(BC)}(Lˣ, Lʸ, H, BC)
    end
end

function barotropic_state!(x, t, νʰ, kˣ, gH)
    M = @SMatrix [-νʰ * kˣ^2 gH * kˣ; -kˣ 0]
    A = exp(M * t) * @SVector [1, 1]

    U = A[1] * sin(kˣ * x)
    η = A[2] * cos(kˣ * x)

    return (U = U, η = η)
end

function ocean_init_state!(m::HBModel, p::SimpleBox, Q, A, coords, t)
    @inbounds x = coords[1]
    @inbounds y = coords[2]
    @inbounds z = coords[3]

    kˣ = 2π / p.Lˣ
    kʸ = 2π / p.Lʸ
    kᶻ = 2π / p.H

    gH = grav(m.param_set) * p.H
    U, η = barotropic_state!(x, t, m.νʰ, kˣ, gH)

    λ = m.νʰ * kˣ^2 + m.νᶻ * kᶻ^2
    u° = exp(-λ * t) * cos(kᶻ * z) * sin(kˣ * x)
    u = u° + U / p.H

    Q.u = @SVector [u, -0]
    Q.η = η
    Q.θ = -0

    return nothing
end

function ocean_init_state!(m::SWModel, p::SimpleBox, Q, A, coords, t)
    @inbounds x = coords[1]
    kˣ = 2π / p.Lˣ
    νʰ = m.turbulence.ν
    gH = grav(m.param_set) * p.H

    U, η = barotropic_state!(x, t, νʰ, kˣ, gH)

    Q.U = @SVector [U, -0]
    Q.η = η

    return nothing
end

@inline kinematic_stress(p::SimpleBox, y) = @SVector [-0, -0]

##########################
# Homogenous wind stress #
# Constant temperature   #
##########################

"""
    HomogeneousBox <: AbstractSimpleBoxProblem

Container structure for a simple box problem with wind-stress.
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
        Lˣ,             # m
        Lʸ,             # m
        H;              # m
        τₒ = FT(1e-1),  # N/m²
        BC = (
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Penetrable(KinematicStress()), Insulating()),
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
function ocean_init_state!(m::HBModel, p::HomogeneousBox, Q, A, coords, t)
    Q.u = @SVector [0, 0]
    Q.η = 0
    Q.θ = 20

    return nothing
end

include("ShallowWaterInitialStates.jl")

function ocean_init_state!(m::SWModel, p::HomogeneousBox, Q, A, coords, t)
    if t == 0
        null_init_state!(p, m.turbulence, Q, A, coords, 0)
    else
        gyre_init_state!(m, p, m.turbulence, Q, A, coords, t)
    end
end

@inline coriolis_parameter(m::SWModel, p::HomogeneousBox, y) =
    m.fₒ + m.β * (y - p.Lʸ / 2)

"""
    kinematic_stress(::HomogeneousBox)

jet stream like windstress

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
"""
@inline kinematic_stress(p::HomogeneousBox, y, ρ) =
    @SVector [(p.τₒ / ρ) * cos(y * π / p.Lʸ), -0]

@inline kinematic_stress(
    p::HomogeneousBox,
    y,
) = @SVector [-p.τₒ * cos(π * y / p.Lʸ), -0]

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
        Lˣ,                  # m
        Lʸ,                  # m
        H;                   # m
        τₒ = FT(1e-1),       # N/m²
        λʳ = FT(4 // 86400), # m/s
        θᴱ = FT(10),         # K
        BC = (
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Impenetrable(NoSlip()), Insulating()),
            OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
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
function ocean_init_state!(m::HBModel, p::OceanGyre, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [0, 0]
    Q.η = 0
    Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    return nothing
end

function ocean_init_state!(m::SWModel, p::OceanGyre, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [0, 0]
    Q.η = 0

    return nothing
end

"""
    kinematic_stress(::OceanGyre)

jet stream like windstress

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
"""
@inline kinematic_stress(p::OceanGyre, y, ρ) =
    @SVector [(p.τₒ / ρ) * cos(y * π / p.Lʸ), -0]

@inline kinematic_stress(
    p::OceanGyre,
    y,
) = @SVector [-p.τₒ * cos(π * y / p.Lʸ), -0]

"""
    surface_flux(::OceanGyre)

cool-warm north-south linear temperature gradient

# Arguments
- `p`: problem object to dispatch on and get additional parameters
- `y`: y-coordinate in the box
- `θ`: temperature within element on boundary
"""
@inline function surface_flux(p::OceanGyre, y, θ)
    Lʸ = p.Lʸ
    θᴱ = p.θᴱ
    λʳ = p.λʳ

    θʳ = θᴱ * (1 - y / Lʸ)
    return λʳ * (θ - θʳ)
end

end
