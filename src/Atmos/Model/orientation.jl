# TODO: add Coriolis vectors
using CLIMAParameters.Planet: grav, planet_radius
export Orientation, NoOrientation, FlatOrientation, SphericalOrientation
export vertical_unit_vector,
    altitude,
    latitude,
    longitude,
    gravitational_potential,
    projection_normal,
    projection_tangential

abstract type Orientation end


function vars_state_auxiliary(m::Orientation, T)
    @vars begin
        Φ::T # gravitational potential
        ∇Φ::SVector{3, T}
    end
end

altitude(atmos::AtmosModel, aux::Vars) =
    altitude(atmos.orientation, atmos.param_set, aux)
latitude(atmos::AtmosModel, aux::Vars) = latitude(atmos.orientation, aux)
longitude(atmos::AtmosModel, aux::Vars) = longitude(atmos.orientation, aux)
vertical_unit_vector(atmos::AtmosModel, aux::Vars) =
    vertical_unit_vector(atmos.orientation, atmos.param_set, aux)
projection_normal(atmos::AtmosModel, aux::Vars, u⃗::AbstractVector) =
    projection_normal(atmos.orientation, atmos.param_set, aux, u⃗)
projection_tangential(atmos::AtmosModel, aux::Vars, u⃗::AbstractVector) =
    projection_tangential(atmos.orientation, atmos.param_set, aux, u⃗)


gravitational_potential(::Orientation, aux::Vars) = aux.orientation.Φ
∇gravitational_potential(::Orientation, aux::Vars) = aux.orientation.∇Φ
function altitude(orientation::Orientation, param_set, aux::Vars)
    FT = eltype(aux)
    return gravitational_potential(orientation, aux) / FT(grav(param_set))
end
function vertical_unit_vector(orientation::Orientation, param_set, aux::Vars)
    FT = eltype(aux)
    ∇gravitational_potential(orientation, aux) / FT(grav(param_set))
end

function projection_normal(
    orientation::Orientation,
    param_set,
    aux::Vars,
    u⃗::AbstractVector,
)
    n̂ = vertical_unit_vector(orientation, param_set, aux)
    return n̂ * (n̂' * u⃗)
end

function projection_tangential(
    orientation::Orientation,
    param_set,
    aux::Vars,
    u⃗::AbstractVector,
)
    return u⃗ .- projection_normal(orientation, param_set, aux, u⃗)
end


"""
    NoOrientation

No gravitional force or potential.
"""
struct NoOrientation <: Orientation end
function vars_state_auxiliary(m::NoOrientation, T)
    @vars()
end
atmos_init_aux!(::NoOrientation, ::AtmosModel, aux::Vars, geom::LocalGeometry) =
    nothing
gravitational_potential(::NoOrientation, aux::Vars) = -zero(eltype(aux))
∇gravitational_potential(::NoOrientation, aux::Vars) =
    SVector{3, eltype(aux)}(0, 0, 0)
altitude(orientation::NoOrientation, param_set, aux::Vars) = -zero(eltype(aux))

"""
    SphericalOrientation <: Orientation

Gravity acts towards the origin `(0,0,0)`, and the gravitational potential is relative
to the surface of the planet.
"""
struct SphericalOrientation <: Orientation end
function atmos_init_aux!(
    ::SphericalOrientation,
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    param_set = atmos.param_set
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


"""
    FlatOrientation <: Orientation

Gravity acts in the third coordinate, and the gravitational potential is relative to
`coord[3] == 0`.
"""
struct FlatOrientation <: Orientation
    # for Coriolis we could add latitude?
end
function atmos_init_aux!(
    ::FlatOrientation,
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
)
    FT = eltype(aux)
    param_set = atmos.param_set
    _grav::FT = grav(param_set)
    aux.orientation.Φ = _grav * aux.coord[3]
    aux.orientation.∇Φ = SVector{3, eltype(aux)}(0, 0, _grav)
end
