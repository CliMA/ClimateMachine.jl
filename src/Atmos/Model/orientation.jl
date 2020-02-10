# TODO: add Coriolis vectors
import ..PlanetParameters: grav, planet_radius
using ..UnitAnnotations
export Orientation, NoOrientation, FlatOrientation, SphericalOrientation
export vertical_unit_vector

using Unitful: numtype

abstract type Orientation
end


function vars_aux(m::Orientation, T)
  @uvars m begin
    Φ::U(T,:gravpot) # gravitational potential
    ∇Φ::SVector{3, U(T,:accel)}
  end
end

gravitational_potential(::Orientation, aux::Vars) = aux.orientation.Φ
∇gravitational_potential(::Orientation, aux::Vars) = aux.orientation.∇Φ
altitude(orientation::Orientation, aux::Vars) =
  (gravitational_potential(orientation, aux) / eltype(aux)(grav, aux))
vertical_unit_vector(orientation::Orientation, aux::Vars) = ∇gravitational_potential(orientation, aux) / eltype(aux)(grav, aux)


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
gravitational_potential(::NoOrientation, aux::Vars) = -zero(eltype(aux)) * get_unit(aux, :gravpot)
∇gravitational_potential(::NoOrientation, aux::Vars) = SVector{3,eltype(aux)}(0,0,0) * get_unit(aux, :accel)
altitude(orientation::NoOrientation, aux::Vars) = -zero(eltype(aux)) * get_unit(aux, :space)

"""
    SphericalOrientation <: Orientation

Gravity acts towards the origin `(0,0,0)`, and the gravitational potential is relative
to the surface of the planet.
"""
struct SphericalOrientation <: Orientation
end
function atmos_init_aux!(::SphericalOrientation, m::AtmosModel, aux::Vars, geom::LocalGeometry)
  FT = eltype(aux)
  normcoord = norm(aux.coord)
  aux.orientation.Φ = FT(grav,m) * (normcoord - FT(planet_radius,m))
  aux.orientation.∇Φ = FT(grav,m) / normcoord .* aux.coord
end

"""
    FlatOrientation <: Orientation

Gravity acts in the third coordinate, and the gravitational potential is relative to
`coord[3] == 0`.
"""
struct FlatOrientation <: Orientation
  # for Coriolis we could add latitude?
end
function atmos_init_aux!(::FlatOrientation, m::AtmosModel, aux::Vars, geom::LocalGeometry)
  FT = eltype(aux)
  aux.orientation.Φ = grav * aux.coord[3]
  u_grad_pot = get_unit(m,:accel)
  v = (0*u_grad_pot, 0*u_grad_pot, FT(grav,m))
  aux.orientation.∇Φ = SVector{3, U(FT, u_grad_pot)}(v)
end
