module Metrics

using ..GeomData

"""
    creategrid!(vgeo, elemtocoord, ξ1)

Create a 1-D grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D
`(-1, 1)` reference coordinates `ξ1`. The element grids are filled using linear
interpolation of the element coordinates.

If `Nq = length(ξ1)` and `nelem = size(elemtocoord, 3)` then the preallocated
array `vgeo.x1` should be `Nq * nelem == length(x1)`.
"""
function creategrid!(vgeo::VolumeGeometry{NTuple{1,Int}, <:AbstractArray}, e2c, ξ1)
    (d, nvert, nelem) = size(e2c)
    @assert d == 1
    Nq = length(ξ1)
    vgeo.x1 = reshape(vgeo.x1, (Nq, nelem))

    # linear blend
    @inbounds for e in 1:nelem
        for i in 1:Nq
            vgeo.x1[i, e] =
                ((1 - ξ1[i]) * e2c[1, 1, e] + (1 + ξ1[i]) * e2c[1, 2, e]) / 2
        end
    end
    nothing
end

"""
    creategrid!(vgeo, elemtocoord, ξ1, ξ2)

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `ξ1` and `ξ2`. The element grids
are filled using bilinear interpolation of the element coordinates.

If `Nq = (length(ξ1), length(ξ2))` and `nelem = size(elemtocoord, 3)` then the
preallocated arrays `vgeo.x1` and `vgeo.x2` should be
`prod(Nq) * nelem == size(vgeo.x1) == size(vgeo.x2)`.
"""
function creategrid!(vgeo::VolumeGeometry{NTuple{2,Int}, <:AbstractArray}, e2c, ξ1, ξ2)
    (d, nvert, nelem) = size(e2c)
    @assert d == 2
    Nq = (length(ξ1), length(ξ2))
    vgeo.x1 = reshape(vgeo.x1, (Nq..., nelem))
    vgeo.x2 = reshape(vgeo.x2, (Nq..., nelem))

    # # bilinear blend of corners
    @inbounds for (f, n) in zip((vgeo.x1, vgeo.x2), 1:d)
        for e in 1:nelem, j in 1:Nq[2], i in 1:Nq[1]
            f[i, j, e] =
                (
                    (1 - ξ1[i]) * (1 - ξ2[j]) * e2c[n, 1, e] +
                    (1 + ξ1[i]) * (1 - ξ2[j]) * e2c[n, 2, e] +
                    (1 - ξ1[i]) * (1 + ξ2[j]) * e2c[n, 3, e] +
                    (1 + ξ1[i]) * (1 + ξ2[j]) * e2c[n, 4, e]
                ) / 4
        end
    end
    nothing
end

"""
    creategrid!(vgeo, elemtocoord, ξ1, ξ2, ξ3)

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `ξ1`. The element grids are filled
using trilinear interpolation of the element coordinates.

If `Nq = (length(ξ1), length(ξ2), length(ξ3))` and
`nelem = size(elemtocoord, 3)` then the preallocated arrays `vgeo.x1`, `vgeo.x2`,
and `vgeo.x3` should be `prod(Nq) * nelem == size(vgeo.x1) == size(vgeo.x2) == size(vgeo.x3)`.
"""
function creategrid!(vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray}, e2c, ξ1, ξ2, ξ3)
    (d, nvert, nelem) = size(e2c)
    @assert d == 3
    Nq = (length(ξ1), length(ξ2), length(ξ3))
    vgeo.x1 = reshape(vgeo.x1, (Nq..., nelem))
    vgeo.x2 = reshape(vgeo.x2, (Nq..., nelem))
    vgeo.x3 = reshape(vgeo.x3, (Nq..., nelem))

    # trilinear blend of corners
    @inbounds for (f, n) in zip((vgeo.x1, vgeo.x2, vgeo.x3), 1:d)
        for e in 1:nelem, k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]
            f[i, j, k, e] =
                (
                    (1 - ξ1[i]) * (1 - ξ2[j]) * (1 - ξ3[k]) * e2c[n, 1, e] +
                    (1 + ξ1[i]) * (1 - ξ2[j]) * (1 - ξ3[k]) * e2c[n, 2, e] +
                    (1 - ξ1[i]) * (1 + ξ2[j]) * (1 - ξ3[k]) * e2c[n, 3, e] +
                    (1 + ξ1[i]) * (1 + ξ2[j]) * (1 - ξ3[k]) * e2c[n, 4, e] +
                    (1 - ξ1[i]) * (1 - ξ2[j]) * (1 + ξ3[k]) * e2c[n, 5, e] +
                    (1 + ξ1[i]) * (1 - ξ2[j]) * (1 + ξ3[k]) * e2c[n, 6, e] +
                    (1 - ξ1[i]) * (1 + ξ2[j]) * (1 + ξ3[k]) * e2c[n, 7, e] +
                    (1 + ξ1[i]) * (1 + ξ2[j]) * (1 + ξ3[k]) * e2c[n, 8, e]
                ) / 8
        end
    end
    nothing
end

"""
    setup_geom_factors_data!(vgeo, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D::DAT2, 1-D derivative operator on the device in the first dimension

Setup the variable needed for geometric factors data in vgeo.
"""
function setup_geom_factors_data!(
    vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray},
    D
)
    Nq = size(D, 1)
    nelem = div(length(vgeo.ωJ), Nq)

    vgeo.x1 = reshape(vgeo.x1, (Nq, nelem))
    vgeo.ωJ = reshape(vgeo.ωJ, (Nq, nelem))
    vgeo.JcV = reshape(vgeo.JcV, (Nq, nelem))
    vgeo.ξ1x1 = reshape(vgeo.ξ1x1, (Nq, nelem))

end

"""
    setup_geom_factors_data!(vgeo, D1, D2)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D1::DAT2, 1-D derivative operator on the device in the first dimension
- D2::DAT2, 1-D derivative operator on the device in the second dimension

Setup the variable needed for geometric factors data in vgeo.
"""
function setup_geom_factors_data!(
    vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray},
    D1,
    D2
)
    Nq = (size(D1, 1), size(D2, 1))
    nelem = div(length(vgeo.ωJ), prod(Nq))

    vgeo.x1 = reshape(vgeo.x1, (Nq..., nelem))
    vgeo.x2 = reshape(vgeo.x2, (Nq..., nelem))
    vgeo.ωJ = reshape(vgeo.ωJ, (Nq..., nelem))
    vgeo.JcV = reshape(vgeo.JcV, (Nq..., nelem))
    vgeo.ξ1x1 = reshape(vgeo.ξ1x1, (Nq..., nelem))
    vgeo.ξ2x1 = reshape(vgeo.ξ2x1, (Nq..., nelem))
    vgeo.ξ1x2 = reshape(vgeo.ξ1x2, (Nq..., nelem))
    vgeo.ξ2x2 = reshape(vgeo.ξ2x2, (Nq..., nelem))

end

"""
    setup_geom_factors_data!(vgeo, D1, D2, D3)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D1::DAT2, 1-D derivative operator on the device in the first dimension
- D2::DAT2, 1-D derivative operator on the device in the second dimension
- D3::DAT2, 1-D derivative operator on the device in the third dimension

Setup the variable needed for geometric factors data in vgeo.
"""
function setup_geom_factors_data!(
    vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray},
    D1,
    D2,
    D3,
)
    Nq = (size(D1, 1), size(D2, 1), size(D3, 1))
    nelem = div(length(vgeo.ωJ), prod(Nq))

    vgeo.x1 = reshape(vgeo.x1, (Nq..., nelem))
    vgeo.x2 = reshape(vgeo.x2, (Nq..., nelem))
    vgeo.x3 = reshape(vgeo.x3, (Nq..., nelem))
    vgeo.ωJ = reshape(vgeo.ωJ, (Nq..., nelem))
    vgeo.JcV = reshape(vgeo.JcV, (Nq..., nelem))
    vgeo.ξ1x1 = reshape(vgeo.ξ1x1, (Nq..., nelem))
    vgeo.ξ2x1 = reshape(vgeo.ξ2x1, (Nq..., nelem))
    vgeo.ξ3x1 = reshape(vgeo.ξ3x1, (Nq..., nelem))
    vgeo.ξ1x2 = reshape(vgeo.ξ1x2, (Nq..., nelem))
    vgeo.ξ2x2 = reshape(vgeo.ξ2x2, (Nq..., nelem))
    vgeo.ξ3x2 = reshape(vgeo.ξ3x2, (Nq..., nelem))
    vgeo.ξ1x3 = reshape(vgeo.ξ1x3, (Nq..., nelem))
    vgeo.ξ2x3 = reshape(vgeo.ξ2x3, (Nq..., nelem))
    vgeo.ξ3x3 = reshape(vgeo.ξ3x3, (Nq..., nelem))

end

"""
    compute_dxdxi_jacobian!(vgeo, D1, D2)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D1::DAT2, 1-D derivative operator on the device in the first dimension
- D2::DAT2, 1-D derivative operator on the device in the second dimension

Compute the Jacobian matrix of the mapping from physical coordinates,
`vgeo.x1`, `vgeo.x2` with respect to reference coordinates `ξ1`, `ξ2`,
for each quadrature point in element e.
"""
function compute_dxdxi_jacobian!(
    vgeo::VolumeGeometry{NTuple{2,Int}, <:AbstractArray},
    D1,
    D2
)
    T = eltype(vgeo.x1)
    Nq = (size(D1, 1), size(D2, 1))

    vgeo.x1ξ1 .= vgeo.x1ξ2 .= zero(T)
    vgeo.x2ξ1 .= vgeo.x2ξ2 .= zero(T)

    for e in 1:nelem
        for j in 1:Nq[2], i in 1:Nq[1]
            for n in 1:Nq[1]
                vgeo.x1ξ1[i, j, e] += D1[i, n] * vgeo.x1[n, j, e]
                vgeo.x2ξ1[i, j, e] += D1[i, n] * vgeo.x2[n, j, e]
            end
            for n in 1:Nq[2]
                vgeo.x1ξ2[i, j, e] += D2[j, n] * vgeo.x1[i, n, e]
                vgeo.x2ξ2[i, j, e] += D2[j, n] * vgeo.x2[i, n, e]
            end
        end
    end

    return vgeo
end

"""
    compute_dxdxi_jacobian!(vgeo, D1, D2, D3)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D1::DAT2, 1-D derivative operator on the device in the first dimension
- D2::DAT2, 1-D derivative operator on the device in the second dimension
- D3::DAT2, 1-D derivative operator on the device in the third dimension

Compute the Jacobian matrix of the mapping from physical coordinates,
`vgeo.x1`, `vgeo.x2`, `vgeo.x3`, with respect to reference coordinates `ξ1`,
`ξ2`, `ξ3`, for each quadrature point in element e.
"""
function compute_dxdxi_jacobian!(
    vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray},
    D1,
    D2,
    D3
)

    T = eltype(vgeo.x1)
    Nq = (size(D1, 1), size(D2, 1), size(D3, 1))

    vgeo.x1ξ1 .= vgeo.x1ξ2 .= vgeo.x1ξ3 .= zero(T)
    vgeo.x2ξ1 .= vgeo.x2ξ2 .= vgeo.x2ξ3 .= zero(T)
    vgeo.x3ξ1 .= vgeo.x3ξ2 .= vgeo.x3ξ3 .= zero(T)

    @inbounds for e in 1:nelem
        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]

            for n in 1:Nq[1]
                vgeo.x1ξ1[i, j, k, e] += D1[i, n] * vgeo.x1[n, j, k, e]
                vgeo.x2ξ1[i, j, k, e] += D1[i, n] * vgeo.x2[n, j, k, e]
                vgeo.x3ξ1[i, j, k, e] += D1[i, n] * vgeo.x3[n, j, k, e]
            end
            for n in 1:Nq[2]
                vgeo.x1ξ2[i, j, k, e] += D2[j, n] * vgeo.x1[i, n, k, e]
                vgeo.x2ξ2[i, j, k, e] += D2[j, n] * vgeo.x2[i, n, k, e]
                vgeo.x3ξ2[i, j, k, e] += D2[j, n] * vgeo.x3[i, n, k, e]
            end
            for n in 1:Nq[3]
                vgeo.x1ξ3[i, j, k, e] += D3[k, n] * vgeo.x1[i, j, n, e]
                vgeo.x2ξ3[i, j, k, e] += D3[k, n] * vgeo.x2[i, j, n, e]
                vgeo.x3ξ3[i, j, k, e] += D3[k, n] * vgeo.x3[i, j, n, e]
            end

        end
    end

    return vgeo
end

"""
    computemetric!(vgeo, sgeo, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- sgeo::SurfaceGeometry, a struct containing the surface geometric factors
- D::DAT2, 1-D derivative operator on the device

Compute the 1-D metric terms from the element grid arrays `vgeo.x1`. All the arrays
are preallocated by the user and the (square) derivative matrix `D` should be
consistent with the reference grid `ξ1` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x1), Nq)` then the volume arrays
`x1`, `J`, and `ξ1x1` should all have length `Nq * nelem`.  Similarly, the face
arrays `sJ` and `n1` should be of length `nface * nelem` with `nface = 2`.
"""
function computemetric!(
    vgeo::VolumeGeometry{NTuple{1,Int}, <:AbstractArray},
    sgeo::SurfaceGeometry{NTuple{1,Int}, <:AbstractArray},
    D
)

    Nq = size(D, 1)
    nelem = div(length(vgeo.ωJ), Nq)

    nface = 2
    sgeo.n1 = reshape(sgeo.n1, (1, nface, nelem))
    sgeo.sωJ = reshape(sgeo.sωJ, (1, nface, nelem))

    @inbounds for e in 1:nelem
        vgeo.JcV[:, e] = vgeo.ωJ[:, e] = D * vgeo.x1[:, e]
    end
    vgeo.ξ1x1 .= 1 ./ vgeo.ωJ

    sgeo.n1[1, 1, :] .= -sign.(vgeo.ωJ[1, :])
    sgeo.n1[1, 2, :] .= sign.(vgeo.ωJ[Nq, :])
    sgeo.sωJ .= 1
    nothing
end

"""
    computemetric!(vgeo, sgeo, D1, D2)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- sgeo::SurfaceGeometry, a struct containing the surface geometric factors
- D1::DAT2, 1-D derivative operator on the device in the first dimension
- D2::DAT2, 1-D derivative operator on the device in the second dimension

Compute the 2-D metric terms from the element grid arrays `vgeo.x1` and `vgeo.x2`. All the
arrays are preallocated by the user and the (square) derivative matrice `D1` and
`D2` should be consistent with the reference grid `ξ1` and `ξ2` used in
[`creategrid!`](@ref).

If `Nq = (size(D1, 1), size(D2, 1))` and `nelem = div(length(vgeo.x1), prod(Nq))`
then the volume arrays `vgeo.x1`, `vgeo.x2`, `vgeo.ωJ`, `vgeo.ξ1x1`, `vgeo.ξ2x1`, `vgeo.ξ1x2`, and `vgeo.ξ2x2`
should all be of size `(Nq..., nelem)`.  Similarly, the face arrays `sgeo.sωJ`, `sgeo.n1`,
and `sgeo.n2` should be of size `(maximum(Nq), nface, nelem)` with `nface = 4`

"""
function computemetric!(
    vgeo::VolumeGeometry{NTuple{2,Int}, <:AbstractArray},
    sgeo::SurfaceGeometry{NTuple{2,Int}, <:AbstractArray},
    D1,
    D2,
)
    T = eltype(vgeo.x1)
    Nq = (size(D1, 1), size(D2, 1))
    nelem = div(length(vgeo.ωJ), prod(Nq))
    nface = 4
    Nfp = div.(prod(Nq), Nq)
    sgeo.n1 = reshape(sgeo.n1, (maximum(Nfp), nface, nelem))
    sgeo.n2 = reshape(sgeo.n2, (maximum(Nfp), nface, nelem))
    sgeo.sωJ = reshape(sgeo.sωJ, (maximum(Nfp), nface, nelem))

    for e in 1:nelem
        for j in 1:Nq[2], i in 1:Nq[1]

            # Compute vertical Jacobian determinant per quadrature point
            vgeo.JcV[i, j, e] = hypot(vgeo.x1ξ2, vgeo.x2ξ2)
            # Compute Jacobian determinant, det(∂x/∂ξ), per quadrature point
            vgeo.ωJ[i, j, e] = vgeo.x1ξ1 * vgeo.x2ξ2 - vgeo.x2ξ1 * vgeo.x1ξ2

            vgeo.ξ1x1[i, j, e] =  vgeo.x2ξ2 / vgeo.ωJ[i, j, e]
            vgeo.ξ2x1[i, j, e] = -vgeo.x2ξ1 / vgeo.ωJ[i, j, e]
            vgeo.ξ1x2[i, j, e] = -vgeo.x1ξ2 / vgeo.ωJ[i, j, e]
            vgeo.ξ2x2[i, j, e] =  vgeo.x1ξ1 / vgeo.ωJ[i, j, e]
        end

        for i in 1:maximum(Nfp)
            if i <= Nfp[1]
                sgeo.n1[i, 1, e] = -vgeo.ωJ[1, i, e] * vgeo.ξ1x1[1, i, e]
                sgeo.n2[i, 1, e] = -vgeo.ωJ[1, i, e] * vgeo.ξ1x2[1, i, e]
                sgeo.n1[i, 2, e] =  vgeo.ωJ[Nq[1], i, e] * vgeo.ξ1x1[Nq[1], i, e]
                sgeo.n2[i, 2, e] =  vgeo.ωJ[Nq[1], i, e] * vgeo.ξ1x2[Nq[1], i, e]
            else
                sgeo.n1[i, 1:2, e] .= NaN
                sgeo.n2[i, 1:2, e] .= NaN
            end
            if i <= Nfp[2]
                sgeo.n1[i, 3, e] = -vgeo.ωJ[i, 1, e] * vgeo.ξ2x1[i, 1, e]
                sgeo.n2[i, 3, e] = -vgeo.ωJ[i, 1, e] * vgeo.ξ2x2[i, 1, e]
                sgeo.n1[i, 4, e] =  vgeo.ωJ[i, Nq[2], e] * vgeo.ξ2x1[i, Nq[2], e]
                sgeo.n2[i, 4, e] =  vgeo.ωJ[i, Nq[2], e] * vgeo.ξ2x2[i, Nq[2], e]
            else
                sgeo.n1[i, 3:4, e] .= NaN
                sgeo.n2[i, 3:4, e] .= NaN
            end

            for n in 1:nface
                sgeo.sωJ[i, n, e] = hypot(sgeo.n1[i, n, e], sgeo.n2[i, n, e])
                sgeo.n1[i, n, e] /= sgeo.sωJ[i, n, e]
                sgeo.n2[i, n, e] /= sgeo.sωJ[i, n, e]
            end
        end
    end

    nothing
end

"""
    computemetric!(vgeo, sgeo, D1, D2, D3)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- sgeo::SurfaceGeometry, a struct containing the surface geometric factors
- D1::DAT2, 1-D derivative operator on the device in the first dimension
- D2::DAT2, 1-D derivative operator on the device in the second dimension
- D3::DAT2, 1-D derivative operator on the device in the third dimension

Compute the 3-D metric terms from the element grid arrays `vgeo.x1`, `vgeo.x2`, and `vgeo.x3`.
All the arrays are preallocated by the user and the (square) derivative matrice `D1`,
`D2`, and `D3` should be consistent with the reference grid `ξ1`, `ξ2`, and `ξ3` used in
[`creategrid!`](@ref).

If `Nq = size(D1, 1)` and `nelem = div(length(vgeo.x1), Nq^3)` then the volume arrays
`vgeo.x1`, `vgeo.x2`, `vgeo.x3`, `vgeo.ωJ`, `vgeo.ξ1x1`, `vgeo.ξ2x1`, `vgeo.ξ3x1`,
`vgeo.ξ1x2`, `vgeo.ξ2x2`, `vgeo.ξ3x2`, `vgeo.ξ1x3`,`vgeo.ξ2x3`, and `vgeo.ξ3x3`
should all be of length `Nq^3 * nelem`.  Similarly, the face
arrays `sgeo.sωJ`, `sgeo.n1`, `sgeo.n2`, and `sgeo.n3` should be of size `Nq^2 * nface * nelem` with
`nface = 6`.

The curl invariant formulation of Kopriva (2006), equation 37, is used.

Reference:
 - [Kopriva2006](@cite)
"""
function computemetric!(
    vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray},
    sgeo::SurfaceGeometry{NTuple{3,Int}, <:AbstractArray},
    D1,
    D2,
    D3
)
    T = eltype(vgeo.x1)

    Nq = (size(D1, 1), size(D2, 1), size(D3, 1))
    Np = prod(Nq)
    Nfp = div.(Np, Nq)
    nelem = div(length(vgeo.ωJ), Np)

    nface = 6
    sgeo.n1 = reshape(sgeo.n1, maximum(Nfp), nface, nelem)
    sgeo.n2 = reshape(sgeo.n2, maximum(Nfp), nface, nelem)
    sgeo.n3 = reshape(sgeo.n3, maximum(Nfp), nface, nelem)
    sgeo.sωJ = reshape(sgeo.sωJ, maximum(Nfp), nface, nelem)

    JI2 = similar(vgeo.ωJ, Nq...)
    (yzr, yzs, yzt) = (similar(JI2), similar(JI2), similar(JI2))
    (zxr, zxs, zxt) = (similar(JI2), similar(JI2), similar(JI2))
    (xyr, xys, xyt) = (similar(JI2), similar(JI2), similar(JI2))

    vgeo.ξ1x1 .= vgeo.ξ2x1 .= vgeo.ξ3x1 .= zero(T)
    vgeo.ξ1x2 .= vgeo.ξ2x2 .= vgeo.ξ3x2 .= zero(T)
    vgeo.ξ1x3 .= vgeo.ξ2x3 .= vgeo.ξ3x3 .= zero(T)

    fill!(sgeo.n1, NaN)
    fill!(sgeo.n2, NaN)
    fill!(sgeo.n3, NaN)
    fill!(sgeo.sωJ, NaN)

    @inbounds for e in 1:nelem
        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]

            # Compute vertical Jacobian determinant per quadrature point
            vgeo.JcV[i, j, k, e] = hypot(vgeo.x1ξ3, vgeo.x2ξ3, vgeo.x3ξ3)
            # Compute Jacobian determinant, det(∂x/∂ξ), per quadrature point
            J[i, j, k, e] = (
                vgeo.x1ξ1 * (vgeo.x2ξ2 * vgeo.x3ξ3 - vgeo.x3ξ2 * vgeo.x2ξ3) +
                vgeo.x2ξ1 * (vgeo.x3ξ2 * vgeo.x1ξ3 - vgeo.x1ξ2 * vgeo.x3ξ3) +
                vgeo.x3ξ1 * (vgeo.x1ξ2 * vgeo.x2ξ3 - vgeo.x2ξ2 * vgeo.x1ξ3)
            )

            JI2[i, j, k] = 1 / (2 * vgeo.ωJ[i, j, k, e])

            yzr[i, j, k] = vgeo.x2[i, j, k, e] * vgeo.x3ξ1 - vgeo.x3[i, j, k, e] * vgeo.x2ξ1
            yzs[i, j, k] = vgeo.x2[i, j, k, e] * vgeo.x3ξ2 - vgeo.x3[i, j, k, e] * vgeo.x2ξ2
            yzt[i, j, k] = vgeo.x2[i, j, k, e] * vgeo.x3ξ3 - vgeo.x3[i, j, k, e] * vgeo.x2ξ3
            zxr[i, j, k] = vgeo.x3[i, j, k, e] * vgeo.x1ξ1 - vgeo.x1[i, j, k, e] * vgeo.x3ξ1
            zxs[i, j, k] = vgeo.x3[i, j, k, e] * vgeo.x1ξ2 - vgeo.x1[i, j, k, e] * vgeo.x3ξ2
            zxt[i, j, k] = vgeo.x3[i, j, k, e] * vgeo.x1ξ3 - vgeo.x1[i, j, k, e] * vgeo.x3ξ3
            xyr[i, j, k] = vgeo.x1[i, j, k, e] * vgeo.x2ξ1 - vgeo.x2[i, j, k, e] * vgeo.x1ξ1
            xys[i, j, k] = vgeo.x1[i, j, k, e] * vgeo.x2ξ2 - vgeo.x2[i, j, k, e] * vgeo.x1ξ2
            xyt[i, j, k] = vgeo.x1[i, j, k, e] * vgeo.x2ξ3 - vgeo.x2[i, j, k, e] * vgeo.x1ξ3
        end

        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]
            for n in 1:Nq[1]
                vgeo.ξ2x1[i, j, k, e] -= D1[i, n] * yzt[n, j, k]
                vgeo.ξ3x1[i, j, k, e] += D1[i, n] * yzs[n, j, k]
                vgeo.ξ2x2[i, j, k, e] -= D1[i, n] * zxt[n, j, k]
                vgeo.ξ3x2[i, j, k, e] += D1[i, n] * zxs[n, j, k]
                vgeo.ξ2x3[i, j, k, e] -= D1[i, n] * xyt[n, j, k]
                vgeo.ξ3x3[i, j, k, e] += D1[i, n] * xys[n, j, k]
            end
            for n in 1:Nq[2]
                vgeo.ξ1x1[i, j, k, e] += D2[j, n] * yzt[i, n, k]
                vgeo.ξ3x1[i, j, k, e] -= D2[j, n] * yzr[i, n, k]
                vgeo.ξ1x2[i, j, k, e] += D2[j, n] * zxt[i, n, k]
                vgeo.ξ3x2[i, j, k, e] -= D2[j, n] * zxr[i, n, k]
                vgeo.ξ1x3[i, j, k, e] += D2[j, n] * xyt[i, n, k]
                vgeo.ξ3x3[i, j, k, e] -= D2[j, n] * xyr[i, n, k]
            end
            for n in 1:Nq[3]
                vgeo.ξ1x1[i, j, k, e] -= D3[k, n] * yzs[i, j, n]
                vgeo.ξ2x1[i, j, k, e] += D3[k, n] * yzr[i, j, n]
                vgeo.ξ1x2[i, j, k, e] -= D3[k, n] * zxs[i, j, n]
                vgeo.ξ2x2[i, j, k, e] += D3[k, n] * zxr[i, j, n]
                vgeo.ξ1x3[i, j, k, e] -= D3[k, n] * xys[i, j, n]
                vgeo.ξ2x3[i, j, k, e] += D3[k, n] * xyr[i, j, n]
            end
            vgeo.ξ1x1[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ2x1[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ3x1[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ1x2[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ2x2[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ3x2[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ1x3[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ2x3[i, j, k, e] *= JI2[i, j, k]
            vgeo.ξ3x3[i, j, k, e] *= JI2[i, j, k]
        end

        # faces 1 & 2
        for k in 1:Nq[3], j in 1:Nq[2]
            n = j + (k - 1) * Nq[2]
            sgeo.n1[n, 1, e] = -vgeo.ωJ[1, j, k, e] * vgeo.ξ1x1[1, j, k, e]
            sgeo.n2[n, 1, e] = -vgeo.ωJ[1, j, k, e] * vgeo.ξ1x2[1, j, k, e]
            sgeo.n3[n, 1, e] = -vgeo.ωJ[1, j, k, e] * vgeo.ξ1x3[1, j, k, e]
            sgeo.n1[n, 2, e] =  vgeo.ωJ[Nq[1], j, k, e] * vgeo.ξ1x1[Nq[1], j, k, e]
            sgeo.n2[n, 2, e] =  vgeo.ωJ[Nq[1], j, k, e] * vgeo.ξ1x2[Nq[1], j, k, e]
            sgeo.n3[n, 2, e] =  vgeo.ωJ[Nq[1], j, k, e] * vgeo.ξ1x3[Nq[1], j, k, e]
            for f in 1:2
                sgeo.sωJ[n, f, e] = hypot(sgeo.n1[n, f, e], sgeo.n2[n, f, e], sgeo.n3[n, f, e])
                sgeo.n1[n, f, e] /= sgeo.sωJ[n, f, e]
                sgeo.n2[n, f, e] /= sgeo.sωJ[n, f, e]
                sgeo.n3[n, f, e] /= sgeo.sωJ[n, f, e]
            end
        end
        # faces 3 & 4
        for k in 1:Nq[3], i in 1:Nq[1]
            n = i + (k - 1) * Nq[1]
            sgeo.n1[n, 3, e] = -vgeo.ωJ[i, 1, k, e] * vgeo.ξ2x1[i, 1, k, e]
            sgeo.n2[n, 3, e] = -vgeo.ωJ[i, 1, k, e] * vgeo.ξ2x2[i, 1, k, e]
            sgeo.n3[n, 3, e] = -vgeo.ωJ[i, 1, k, e] * vgeo.ξ2x3[i, 1, k, e]
            sgeo.n1[n, 4, e] =  vgeo.ωJ[i, Nq[2], k, e] * vgeo.ξ2x1[i, Nq[2], k, e]
            sgeo.n2[n, 4, e] =  vgeo.ωJ[i, Nq[2], k, e] * vgeo.ξ2x2[i, Nq[2], k, e]
            sgeo.n3[n, 4, e] =  vgeo.ωJ[i, Nq[2], k, e] * vgeo.ξ2x3[i, Nq[2], k, e]
            for f in 3:4
                sgeo.sωJ[n, f, e] = hypot(sgeo.n1[n, f, e], sgeo.n2[n, f, e], sgeo.n3[n, f, e])
                sgeo.n1[n, f, e] /= sgeo.sωJ[n, f, e]
                sgeo.n2[n, f, e] /= sgeo.sωJ[n, f, e]
                sgeo.n3[n, f, e] /= sgeo.sωJ[n, f, e]
            end
        end
        # faces 5 & 6
        for j in 1:Nq[2], i in 1:Nq[1]
            n = i + (j - 1) * Nq[1]
            sgeo.n1[n, 5, e] = -vgeo.ωJ[i, j, 1, e] * vgeo.ξ3x1[i, j, 1, e]
            sgeo.n2[n, 5, e] = -vgeo.ωJ[i, j, 1, e] * vgeo.ξ3x2[i, j, 1, e]
            sgeo.n3[n, 5, e] = -vgeo.ωJ[i, j, 1, e] * vgeo.ξ3x3[i, j, 1, e]
            sgeo.n1[n, 6, e] =  vgeo.ωJ[i, j, Nq[3], e] * vgeo.ξ3x1[i, j, Nq[3], e]
            sgeo.n2[n, 6, e] =  vgeo.ωJ[i, j, Nq[3], e] * vgeo.ξ3x2[i, j, Nq[3], e]
            sgeo.n3[n, 6, e] =  vgeo.ωJ[i, j, Nq[3], e] * vgeo.ξ3x3[i, j, Nq[3], e]
            for f in 5:6
                sgeo.sωJ[n, f, e] = hypot(sgeo.n1[n, f, e], sgeo.n2[n, f, e], sgeo.n3[n, f, e])
                sgeo.n1[n, f, e] /= sgeo.sωJ[n, f, e]
                sgeo.n2[n, f, e] /= sgeo.sωJ[n, f, e]
                sgeo.n3[n, f, e] /= sgeo.sωJ[n, f, e]
            end
        end
    end

    nothing
end

end # module
