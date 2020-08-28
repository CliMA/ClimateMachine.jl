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
using ..BalanceLaws
using ..MPIStateArrays: MPIStateArray
import ..BalanceLaws: vars_state
using ..DGMethods:
    init_state_auxiliary!, continuous_field_gradient!, LocalGeometry

export Orientation, NoOrientation, FlatOrientation, SphericalOrientation

export init_aux!,
    orientation_nodal_init_aux!,
    vertical_unit_vector,
    altitude,
    latitude,
    longitude,
    projection_normal,
    gravitational_potential,
    ∇gravitational_potential,
    projection_tangential,
    sphr_to_cart_vec,
    cart_to_sphr_vec


abstract type Orientation <: BalanceLaw end

#####
##### Fallbacks
#####

function vars_state(m::Orientation, ::Auxiliary, FT)
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

function init_aux!(
    m,
    ::Orientation,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_state_auxiliary!(
        m,
        (m, aux, tmp, geom) ->
            orientation_nodal_init_aux!(m.orientation, m.param_set, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )

    continuous_field_gradient!(
        m,
        state_auxiliary,
        ("orientation.∇Φ",),
        state_auxiliary,
        ("orientation.Φ",),
        grid,
        direction,
    )
end

#####
##### NoOrientation
#####

"""
    NoOrientation

No gravitational force or potential.
"""
struct NoOrientation <: Orientation end

init_aux!(m, ::NoOrientation, state_auxiliary::MPIStateArray, grid, direction) =
    nothing

vars_state(m::NoOrientation, ::Auxiliary, FT) = @vars()

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

function orientation_nodal_init_aux!(
    ::SphericalOrientation,
    param_set::APS,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    _grav::FT = grav(param_set)
    _planet_radius::FT = planet_radius(param_set)
    normcoord = norm(geom.coord)
    aux.orientation.Φ = _grav * (normcoord - _planet_radius)
end


# TODO: should we define these for non-spherical orientations?
latitude(orientation::SphericalOrientation, aux::Vars) =
    @inbounds asin(aux.coord[3] / norm(aux.coord, 2))

longitude(orientation::SphericalOrientation, aux::Vars) =
    @inbounds atan(aux.coord[2], aux.coord[1])

"""
    sphr_to_cart_vec(orientation::SphericalOrientation, state::Vars, aux::Vars)

Projects a vector defined based on unit vectors in spherical coordinates to cartesian unit vectors.
"""
function sphr_to_cart_vec(
    orientation::SphericalOrientation,
    vec::AbstractVector,
    aux::Vars,
)
    FT = eltype(aux)
    lat = latitude(orientation, aux)
    lon = longitude(orientation, aux)

    slat, clat = sin(lat), cos(lat)
    slon, clon = sin(lon), cos(lon)

    u = MVector{3, FT}(
        -slon * vec[1] - slat * clon * vec[2] + clat * clon * vec[3],
        clon * vec[1] - slat * slon * vec[2] + clat * slon * vec[3],
        clat * vec[2] + slat * vec[3],
    )

    return u
end

"""
    cart_to_sphr_vec(orientation::SphericalOrientation, state::Vars, aux::Vars)

Projects a vector defined based on unit vectors in cartesian coordinates to a spherical unit vectors.
"""
function cart_to_sphr_vec(
    orientation::SphericalOrientation,
    vec::AbstractVector,
    aux::Vars,
)
    FT = eltype(aux)
    lat = latitude(orientation, aux)
    lon = longitude(orientation, aux)

    slat, clat = sin(lat), cos(lat)
    slon, clon = sin(lon), cos(lon)

    u = MVector{3, FT}(
        -slon * vec[1] + clon * vec[2],
        -slat * clon * vec[1] - slat * slon * vec[2] + clat * vec[3],
        clat * clon * vec[1] + clat * slon * vec[2] + slat * vec[3],
    )

    return u
end

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
function orientation_nodal_init_aux!(
    ::FlatOrientation,
    param_set::APS,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    _grav::FT = grav(param_set)
    @inbounds aux.orientation.Φ = _grav * geom.coord[3]
end

end
