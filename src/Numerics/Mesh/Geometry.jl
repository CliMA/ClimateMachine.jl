module Geometry

using StaticArrays, LinearAlgebra, DocStringExtensions
using ..Grids:
    _ξ1x1,
    _ξ2x1,
    _ξ3x1,
    _ξ1x2,
    _ξ2x2,
    _ξ3x2,
    _ξ1x3,
    _ξ2x3,
    _ξ3x3,
    _M,
    _MI,
    _x1,
    _x2,
    _x3,
    _JcV
export LocalGeometry, lengthscale, resolutionmetric

"""
    LocalGeometry

The local geometry at a nodal point.

# Constructors

    LocalGeometry(polynomial::Val, vgeo::AbstractArray{T}, n::Integer, e::Integer)

Extracts a `LocalGeometry` object from the `vgeo` array at node `n` in element `e`.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct LocalGeometry{T, P}
    "Polynomial interpolant: currently this is assumed to be `Val{polyorder}`, but this may change in future."
    polynomial::P
    "Cartesian coordinates"
    coord::SVector{3, T}
    "Jacobian from Cartesian to element coordinates: `invJ[i,j]` is ``∂ξ_i/∂x_j``"
    invJ::SMatrix{3, 3, T, 9}
end

function LocalGeometry(
    polynomial::Val,
    vgeo::AbstractArray{T},
    n::Integer,
    e::Integer,
) where {T}
    coord = @SVector T[vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e]]
    invJ = @SMatrix T[
        vgeo[n, _ξ1x1, e] vgeo[n, _ξ1x2, e] vgeo[n, _ξ1x3, e]
        vgeo[n, _ξ2x1, e] vgeo[n, _ξ2x2, e] vgeo[n, _ξ2x3, e]
        vgeo[n, _ξ3x1, e] vgeo[n, _ξ3x2, e] vgeo[n, _ξ3x3, e]
    ]

    LocalGeometry(polynomial, coord, invJ)
end

"""
    resolutionmetric(g::LocalGeometry)

The metric tensor of the discretisation resolution. Given a unit vector `u` in Cartesian
coordinates and `M = resolutionmetric(g)`, `sqrt(u'*M*u)` is the degree-of-freedom density
in the direction of `u`.
"""
function resolutionmetric(
    g::LocalGeometry{T, Val{polyorder}},
) where {T, polyorder}
    S = polyorder * g.invJ / 2
    S' * S # TODO: return an eigendecomposition / symmetric object?
end

"""
    lengthscale(g::LocalGeometry)

The effective grid resolution at the point.
"""
function lengthscale(g::LocalGeometry{T, Val{polyorder}}) where {T, polyorder}
    2 / (cbrt(det(g.invJ)) * polyorder)
end

end # module
