module OceanProblems

export SimpleBox, Fixed, Rotating, HomogeneousBox, OceanGyre

using StaticArrays
using CLIMAParameters.Planet: grav

using ...Problems

using ..Ocean
using ..HydrostaticBoussinesq
using ..ShallowWater
using ..SplitExplicit01

import ..Ocean:
    ocean_init_state!,
    ocean_init_aux!,
    kinematic_stress,
    surface_flux,
    coriolis_parameter

HBModel = HydrostaticBoussinesqModel
SWModel = ShallowWaterModel

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
function ocean_init_aux!(::HBModel, ::AbstractSimpleBoxProblem, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]

    # needed for proper CFL condition calculation
    A.w = -0
    A.pkin = -0
    A.wz0 = -0

    A.uᵈ = @SVector [-0, -0]
    A.ΔGᵘ = @SVector [-0, -0]

    return nothing
end

function ocean_init_aux!(::OceanModel, ::AbstractSimpleBoxProblem, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]

    # needed for proper CFL condition calculation
    A.w = -0
    A.pkin = -0
    A.wz0 = -0

    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

function ocean_init_aux!(::SWModel, ::AbstractSimpleBoxProblem, A, geom)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.Δu = @SVector [-0, -0]

    return nothing
end

function ocean_init_aux!(::BarotropicModel, ::AbstractSimpleBoxProblem, A, geom)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.U_c = @SVector [-0, -0]
    A.η_c = -0
    A.U_s = @SVector [-0, -0]
    A.η_s = -0
    A.Δu = @SVector [-0, -0]
    A.η_diag = -0
    A.Δη = -0

    return nothing
end

"""
    coriolis_parameter

northern hemisphere coriolis

# Arguments
- `m`: model object to dispatch on and get coriolis parameters
- `y`: y-coordinate in the box
"""
@inline coriolis_parameter(
    m::Union{HBModel, OceanModel},
    ::AbstractSimpleBoxProblem,
    y,
) = m.fₒ + m.β * y
@inline coriolis_parameter(
    m::Union{SWModel, BarotropicModel},
    ::AbstractSimpleBoxProblem,
    y,
) = m.fₒ + m.β * y

############################
# Basic box problem        #
# Set up dimensions of box #
############################

abstract type AbstractRotation end
struct Rotating <: AbstractRotation end
struct Fixed <: AbstractRotation end

"""
    SimpleBoxProblem <: AbstractSimpleBoxProblem

Stub structure with the dimensions of the box.
Lˣ = zonal (east-west) length
Lʸ = meridional (north-south) length
H  = height of the ocean
"""
struct SimpleBox{R, T, BC} <: AbstractSimpleBoxProblem
    rotation::R
    Lˣ::T
    Lʸ::T
    H::T
    boundary_conditions::BC
    function SimpleBox{FT}(
        Lˣ, # m
        Lʸ, # m
        H;  # m
        rotation = Fixed(),
        BC = (
            OceanBC(Impenetrable(FreeSlip()), Insulating()),
            OceanBC(Penetrable(FreeSlip()), Insulating()),
        ),
    ) where {FT <: AbstractFloat}
        return new{typeof(rotation), FT, typeof(BC)}(rotation, Lˣ, Lʸ, H, BC)
    end
end

@inline coriolis_parameter(
    m::Union{HBModel, OceanModel},
    ::SimpleBox{R},
    y,
) where {R <: Fixed} = -0
@inline coriolis_parameter(
    m::Union{SWModel, BarotropicModel},
    ::SimpleBox{R},
    y,
) where {R <: Fixed} = -0

@inline coriolis_parameter(
    m::Union{HBModel, OceanModel},
    ::SimpleBox{R},
    y,
) where {R <: Rotating} = m.fₒ
@inline coriolis_parameter(
    m::Union{SWModel, BarotropicModel},
    ::SimpleBox{R},
    y,
) where {R <: Rotating} = m.fₒ

function ocean_init_state!(
    m::Union{SWModel, BarotropicModel},
    p::SimpleBox,
    Q,
    A,
    localgeo,
    t,
)
    coords = localgeo.coord
    k = (2π / p.Lˣ, 2π / p.Lʸ, 2π / p.H)
    ν = viscosity(m)

    gH = gravity_speed(m)
    @inbounds f = coriolis_parameter(m, p, coords[2])

    U, V, η = barotropic_state!(p.rotation, (coords..., t), ν, k, (gH, f))

    Q.U = @SVector [U, V]
    Q.η = η

    return nothing
end

viscosity(m::SWModel) = (m.turbulence.ν, m.turbulence.ν, -0)
viscosity(m::BarotropicModel) = (m.baroclinic.νʰ, m.baroclinic.νʰ, -0)

gravity_speed(m::SWModel) = grav(m.param_set) * m.problem.H
gravity_speed(m::BarotropicModel) =
    grav(m.baroclinic.param_set) * m.baroclinic.problem.H

function ocean_init_state!(
    m::Union{HBModel, OceanModel},
    p::SimpleBox,
    Q,
    A,
    localgeo,
    t,
)
    coords = localgeo.coord
    k = (2π / p.Lˣ, 2π / p.Lʸ, 2π / p.H)
    ν = (m.νʰ, m.νʰ, m.νᶻ)

    gH = grav(m.param_set) * p.H
    @inbounds f = coriolis_parameter(m, p, coords[2])

    U, V, η = barotropic_state!(p.rotation, (coords..., t), ν, k, (gH, f))
    u°, v° = baroclinic_deviation(p.rotation, (coords..., t), ν, k, f)

    u = u° + U / p.H
    v = v° + V / p.H

    Q.u = @SVector [u, v]
    Q.η = η
    Q.θ = -0

    return nothing
end

function barotropic_state!(
    ::Fixed,
    (x, y, z, t),
    (νˣ, νʸ, νᶻ),
    (kˣ, kʸ, kᶻ),
    params,
)
    gH, _ = params

    M = @SMatrix [-νˣ * kˣ^2 gH * kˣ; -kˣ 0]
    A = exp(M * t) * @SVector [1, 1]

    U = A[1] * sin(kˣ * x)
    V = -0
    η = A[2] * cos(kˣ * x)

    return (U = U, V = V, η = η)
end

function baroclinic_deviation(
    ::Fixed,
    (x, y, z, t),
    (νˣ, νʸ, νᶻ),
    (kˣ, kʸ, kᶻ),
    f,
)
    λ = νˣ * kˣ^2 + νᶻ * kᶻ^2

    u° = exp(-λ * t) * cos(kᶻ * z) * sin(kˣ * x)
    v° = -0

    return (u° = u°, v° = v°)
end

function barotropic_state!(
    ::Rotating,
    (x, y, z, t),
    (νˣ, νʸ, νᶻ),
    (kˣ, kʸ, kᶻ),
    params,
)
    gH, f = params

    M = @SMatrix [-νˣ * kˣ^2 f gH * kˣ; -f -νˣ * kˣ^2 0; -kˣ 0 0]
    A = exp(M * t) * @SVector [1, 1, 1]

    U = A[1] * sin(kˣ * x)
    V = A[2] * sin(kˣ * x)
    η = A[3] * cos(kˣ * x)

    return (U = U, V = V, η = η)
end

function baroclinic_deviation(
    ::Rotating,
    (x, y, z, t),
    (νˣ, νʸ, νᶻ),
    (kˣ, kʸ, kᶻ),
    f,
)
    λ = νˣ * kˣ^2 + νᶻ * kᶻ^2

    M = @SMatrix[-λ f; -f -λ]
    A = exp(M * t) * @SVector[1, 1]

    u° = A[1] * cos(kᶻ * z) * sin(kˣ * x)
    v° = A[2] * cos(kᶻ * z) * sin(kˣ * x)

    return (u° = u°, v° = v°)
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
"""
struct HomogeneousBox{T, BC} <: AbstractSimpleBoxProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T
    boundary_conditions::BC
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
- `localgeo`: the local geometry, not used
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(m::HBModel, p::HomogeneousBox, Q, A, localgeo, t)
    Q.u = @SVector [0, 0]
    Q.η = 0
    Q.θ = 20

    return nothing
end

include("ShallowWaterInitialStates.jl")

function ocean_init_state!(m::SWModel, p::HomogeneousBox, Q, A, localgeo, t)
    coords = localgeo.coord
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
- `localgeo`: the local geometry information
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(
    ::Union{HBModel, OceanModel},
    p::OceanGyre,
    Q,
    A,
    localgeo,
    t,
)
    coords = localgeo.coord
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
    localgeo,
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

end
