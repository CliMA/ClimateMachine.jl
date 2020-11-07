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
    boundary_conditions::BC
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
- `coords`: local spatial coordiantes
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(
    ::Union{HBModel, OceanModel},
    p::OceanGyre,
    Q,
    A,
    coords,
    t,
)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [-0, -0]
    Q.η = -0
    Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    return nothing
end

function ocean_init_state!(
    ::Union{SWModel, BarotropicModel},
    ::OceanGyre,
    Q,
    A,
    coords,
    t,
)
    Q.U = @SVector [-0, -0]
    Q.η = -0

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
