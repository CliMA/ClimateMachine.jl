"""
    Orientations

Orientation functions:

 - `vertical_unit_vector`
 - `altitude`
 - `latitude`
 - `longitude`
 - `gravitational_potential`
 - `projection_normal`
 - `projection_tangential`

for orientations:

 - [`NoOrientation`](@ref)
 - [`FlatOrientation`](@ref)
 - [`SphericalOrientation`](@ref)
"""
module Orientations

using CLIMAParameters: AbstractParameterSet
const APS = AbstractParameterSet
using CLIMAParameters.Planet: grav, planet_radius
using StaticArrays
using LinearAlgebra

using ..VariableTemplates
import ..BalanceLaws: BalanceLaw, vars_state_auxiliary

export Orientation, NoOrientation, FlatOrientation, SphericalOrientation

export init_aux!,
    vertical_unit_vector,
    altitude,
    latitude,
    longitude,
    projection_normal,
    gravitational_potential,
    ∇gravitational_potential,
    projection_tangential


abstract type Orientation <: BalanceLaw end

#####
##### Fallbacks
#####

function vars_state_auxiliary(m::Orientation, FT)
    @vars begin
        Φ::FT # gravitational potential
        ∇Φ::SVector{3, FT}
    end
end

gravitational_potential(::Orientation, aux::Vars) = aux.orientation.Φ

∇gravitational_potential(::Orientation, aux::Vars) = aux.orientation.∇Φ

function altitude(orientation::Orientation, param_set::APS, aux::Vars)
    FT = eltype(aux)
    return gravitational_potential(orientation, aux) / FT(grav(param_set))
end

function vertical_unit_vector(
    orientation::Orientation,
    param_set::APS,
    aux::Vars,
)
    FT = eltype(aux)
    return ∇gravitational_potential(orientation, aux) / FT(grav(param_set))
end

function projection_normal(
    orientation::Orientation,
    param_set::APS,
    aux::Vars,
    u⃗::AbstractVector,
)
    n̂ = vertical_unit_vector(orientation, param_set, aux)
    return n̂ * (n̂' * u⃗)
end

function projection_tangential(
    orientation::Orientation,
    param_set::APS,
    aux::Vars,
    u⃗::AbstractVector,
)
    return u⃗ .- projection_normal(orientation, param_set, aux, u⃗)
end

#####
##### NoOrientation
#####

"""
    NoOrientation

No gravitational force or potential.
"""
struct NoOrientation <: Orientation end

init_aux!(::NoOrientation, param_set::APS, aux::Vars) = nothing
vars_state_auxiliary(m::NoOrientation, FT) = @vars()

gravitational_potential(::NoOrientation, aux::Vars) = -zero(eltype(aux))
∇gravitational_potential(::NoOrientation, aux::Vars) =
    SVector{3, eltype(aux)}(0, 0, 0)
altitude(orientation::NoOrientation, param_set::APS, aux::Vars) =
    -zero(eltype(aux))

#####
##### SphericalOrientation
#####

"""
    SphericalOrientation <: Orientation

Gravity acts towards the origin `(0,0,0)`, and the gravitational potential is relative
to the surface of the planet.
"""
struct SphericalOrientation <: Orientation end

function init_aux!(::SphericalOrientation, param_set::APS, aux::Vars)
    FT = eltype(aux)
    _grav::FT = grav(param_set)
    _planet_radius::FT = planet_radius(param_set)
    normcoord = norm(aux.coord)
    aux.orientation.Φ = _grav * (normcoord - _planet_radius)
    aux.orientation.∇Φ = _grav / normcoord .* aux.coord
end

# TODO: should we define these for non-spherical orientations?
latitude(orientation::SphericalOrientation, aux::Vars) =
    @inbounds asin(aux.coord[3] / norm(aux.coord, 2))

longitude(orientation::SphericalOrientation, aux::Vars) =
    @inbounds atan(aux.coord[2], aux.coord[1])


#####
##### FlatOrientation
#####

"""
    FlatOrientation <: Orientation

Gravity acts in the third coordinate, and the gravitational potential is relative to
`coord[3] == 0`.
"""
struct FlatOrientation <: Orientation
    # for Coriolis we could add latitude?
end
function init_aux!(::FlatOrientation, param_set::APS, aux::Vars)
    FT = eltype(aux)
    _grav::FT = grav(param_set)
    @inbounds aux.orientation.Φ = _grav * aux.coord[3]
    aux.orientation.∇Φ = SVector{3, FT}(0, 0, _grav)
end

end
