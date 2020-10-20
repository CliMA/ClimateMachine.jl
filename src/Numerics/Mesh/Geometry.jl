module Geometry

using StaticArrays, LinearAlgebra, DocStringExtensions
using KernelAbstractions.Extras: @unroll
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

    LocalGeometry{Np, N}(vgeo::AbstractArray{T}, n::Integer, e::Integer)

Extracts a `LocalGeometry` object from the `vgeo` array at node `n` in element
`e` with `Np` being the number of points in the element and `N` being the
polynomial order

# Fields

- `polyorder`

   polynomial order of the element

- `coord`

   local degree of freedom Cartesian coordinate 

- `invJ`

   Jacobian from Cartesian to element coordinates: `invJ[i,j]` is ``∂ξ_i / ∂x_j``

$(DocStringExtensions.FIELDS)
"""
struct LocalGeometry{Np, N, AT, IT}
    "Global volume geometry array"
    vgeo::AT
    "element local linear node index"
    n::IT
    "process local element index"
    e::IT

    LocalGeometry{Np, N}(vgeo::AT, n::IT, e::IT) where {Np, N, AT, IT} =
        new{Np, N, AT, IT}(vgeo, n, e)
end

@inline function Base.getproperty(
    geo::LocalGeometry{Np, N},
    sym::Symbol,
) where {Np, N}
    if sym === :polyorder
        return N
    elseif sym === :coord
        vgeo, n, e = getfield(geo, :vgeo), getfield(geo, :n), getfield(geo, :e)
        FT = eltype(vgeo)
        return @SVector FT[vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e]]
    elseif sym === :invJ
        vgeo, n, e = getfield(geo, :vgeo), getfield(geo, :n), getfield(geo, :e)
        FT = eltype(vgeo)
        return @SMatrix FT[
            vgeo[n, _ξ1x1, e] vgeo[n, _ξ1x2, e] vgeo[n, _ξ1x3, e]
            vgeo[n, _ξ2x1, e] vgeo[n, _ξ2x2, e] vgeo[n, _ξ2x3, e]
            vgeo[n, _ξ3x1, e] vgeo[n, _ξ3x2, e] vgeo[n, _ξ3x3, e]
        ]
    elseif sym === :center_coord
        vgeo, n, e = getfield(geo, :vgeo), getfield(geo, :n), getfield(geo, :e)
        FT = eltype(vgeo)
        coords = SVector(vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e])
        V = FT(0)
        xc = FT(0)
        yc = FT(0)
        zc = FT(0)
        @unroll for i in 1:Np
            M = vgeo[i, _M, e]
            V += M
            xc += M * vgeo[i, _x1, e]
            yc += M * vgeo[i, _x2, e]
            zc += M * vgeo[i, _x3, e]
        end
        return SVector(xc / V, yc / V, zc / V)
    else
        return getfield(geo, sym)
    end
end

"""
    resolutionmetric(g::LocalGeometry)

The metric tensor of the discretisation resolution. Given a unit vector `u` in
Cartesian coordinates and `M = resolutionmetric(g)`, `sqrt(u'*M*u)` is the
degree-of-freedom density in the direction of `u`.
"""
function resolutionmetric(g::LocalGeometry)
    S = g.polyorder * g.invJ / 2
    S' * S # TODO: return an eigendecomposition / symmetric object?
end

"""
    lengthscale(g::LocalGeometry)

The effective grid resolution at the point.
"""
lengthscale(g::LocalGeometry) = 2 / (cbrt(det(g.invJ)) * g.polyorder)

end # module
