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
- `coords`: the local spatial coordinates, not used
- `t`: time to evaluate at, not used
"""
function ocean_init_state!(m::HBModel, p::HomogeneousBox, Q, A, coords, t)
    Q.u = @SVector [0, 0]
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
    @SVector [(p.τₒ / ρ) * cos(y * π / p.Lʸ), -0]

@inline kinematic_stress(
    p::HomogeneousBox,
    y,
) = @SVector [-p.τₒ * cos(π * y / p.Lʸ), -0]
