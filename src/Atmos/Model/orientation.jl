# TODO: add Coriolis vectors
import ..PlanetParameters: grav, planet_radius
export NoOrientation, FlatOrientation, SphericalOrientation

abstract type Orientation
end

function vars_aux(m::Orientation, T)
  @vars begin
    Φ::T # gravitational potential
    ∇Φ::SVector{3,T}
  end
end

gravitational_potential(::Orientation, aux::Vars) = aux.orientation.Φ
altitude(orientation::Orientation, aux::Vars) = gravitational_potential(orientation, aux) / grav


"""
    NoOrientation

No gravitional force or potential.
"""
struct NoOrientation <: Orientation
end
function vars_aux(m::NoOrientation, T)
  @vars()
end
atmos_init_aux!(::NoOrientation, ::AtmosModel, aux::Vars, geom::LocalGeometry) = nothing
gravitational_potential(::NoOrientation, aux::Vars) = -zero(eltype(aux))
altitude(orientation::Orientation, aux::Vars) = -zero(eltype(aux))

"""
    SphericalOrientation <: Orientation

Gravity acts towards the origin `(0,0,0)`, and the gravitational potential is relative
to the surface of the planet.
"""
struct SphericalOrientation <: Orientation
end
function atmos_init_aux!(::SphericalOrientation, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  normcoord = norm(aux.coord)
  aux.orientation.Φ = grav * (normcoord - planet_radius)
  aux.orientation.∇Φ = grav / normcoord .* aux.coord
end

"""
    FlatOrientation <: Orientation

Gravity acts in the third coordinate, and the gravitational potential is relative to
`coord[3] == 0`.
"""
struct FlatOrientation <: Orientation
  # for Coriolis we could add latitude?
end
function atmos_init_aux!(::FlatOrientation, ::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.orientation.Φ = grav * aux.coord[3]
  aux.orientation.∇Φ = SVector{3,eltype(aux)}(0,0,grav)
end
