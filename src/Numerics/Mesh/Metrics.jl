module Metrics

"""
    creategrid!(x1, elemtocoord, ξ1)

Create a 1-D grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D
`(-1, 1)` reference coordinates `ξ1`. The element grids are filled using linear
interpolation of the element coordinates.

If `Nq = length(ξ1)` and `nelem = size(elemtocoord, 3)` then the preallocated
array `x1` should be `Nq * nelem == length(x1)`.
"""
function creategrid!(x1, e2c, ξ1)
    (d, nvert, nelem) = size(e2c)
    @assert d == 1
    Nq = length(ξ1)
    x1 = reshape(x1, (Nq, nelem))

    # linear blend
    @inbounds for e in 1:nelem
        for i in 1:Nq
            x1[i, e] =
                ((1 - ξ1[i]) * e2c[1, 1, e] + (1 + ξ1[i]) * e2c[1, 2, e]) / 2
        end
    end
    nothing
end

"""
    creategrid!(x1, x2, elemtocoord, ξ1, ξ2)

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `ξ1` and `ξ2`. The element grids
are filled using bilinear interpolation of the element coordinates.

If `Nq = (length(ξ1), length(ξ2))` and `nelem = size(elemtocoord, 3)` then the
preallocated arrays `x1` and `x2` should be
`prod(Nq) * nelem == size(x1) == size(x2)`.
"""
function creategrid!(x1, x2, e2c, ξ1, ξ2)
    (d, nvert, nelem) = size(e2c)
    @assert d == 2
    Nq = (length(ξ1), length(ξ2))
    x1 = reshape(x1, (Nq..., nelem))
    x2 = reshape(x2, (Nq..., nelem))

    # # bilinear blend of corners
    @inbounds for (f, n) in zip((x1, x2), 1:d)
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
    creategrid!(x1, x2, x3, elemtocoord, ξ1, ξ2, ξ3)

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `ξ1`. The element grids are filled
using trilinear interpolation of the element coordinates.

If `Nq = (length(ξ1), length(ξ2), length(ξ3))` and
`nelem = size(elemtocoord, 3)` then the preallocated arrays `x1`, `x2`, and `x3`
should be `prod(Nq) * nelem == size(x1) == size(x2) == size(x3)`.
"""
function creategrid!(x1, x2, x3, e2c, ξ1, ξ2, ξ3)
    (d, nvert, nelem) = size(e2c)
    @assert d == 3
    Nq = (length(ξ1), length(ξ2), length(ξ3))
    x1 = reshape(x1, (Nq..., nelem))
    x2 = reshape(x2, (Nq..., nelem))
    x3 = reshape(x3, (Nq..., nelem))

    # trilinear blend of corners
    @inbounds for (f, n) in zip((x1, x2, x3), 1:d)
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
    computemetric!(vgeo::VolumeGeometry{NTuple{1,Int}, <:AbstractArray}, sgeo::SurfaceGeometry{NTuple{1,Int}, <:AbstractArray}, D)

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
    D)

    Nq = size(D, 1)
    nelem = div(length(vgeo.MJ), Nq)
    vgeo.x1 = reshape(vgeo.x1, (Nq, nelem))
    vgeo.MJ = reshape(vgeo.MJ, (Nq, nelem))
    vgeo.JcV = reshape(vgeo.JcV, (Nq, nelem))
    vgeo.ξ1x1 = reshape(vgeo.ξ1x1, (Nq, nelem))
    nface = 2
    sgeo.n1 = reshape(sgeo.n1, (1, nface, nelem))
    sgeo.sMJ = reshape(sgeo.sMJ, (1, nface, nelem))

    @inbounds for e in 1:nelem
        vgeo.JcV[:, e] = vgeo.MJ[:, e] = D * vgeo.x1[:, e]
    end
    vgeo.ξ1x1 .= 1 ./ vgeo.MJ

    sgeo.n1[1, 1, :] .= -sign.(vgeo.MJ[1, :])
    sgeo.n1[1, 2, :] .= sign.(vgeo.MJ[Nq, :])
    sgeo.sMJ .= 1
    nothing
end

"""
    computemetric!(vgeo::VolumeGeometry{NTuple{2,Int}, <:AbstractArray}, sgeo::SurfaceGeometry{NTuple{2,Int}, <:AbstractArray}, D1, D2)

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
then the volume arrays `vgeo.x1`, `vgeo.x2`, `vgeo.MJ`, `vgeo.ξ1x1`, `vgeo.ξ2x1`, `vgeo.ξ1x2`, and `vgeo.ξ2x2`
should all be of size `(Nq..., nelem)`.  Similarly, the face arrays `sgeo.sMJ`, `sgeo.n1`,
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
    nelem = div(length(vgeo.MJ), prod(Nq))
    vgeo.x1 = reshape(vgeo.x1, (Nq..., nelem))
    vgeo.x2 = reshape(vgeo.x2, (Nq..., nelem))
    vgeo.MJ = reshape(vgeo.MJ, (Nq..., nelem))
    vgeo.JcV = reshape(vgeo.JcV, (Nq..., nelem))
    vgeo.ξ1x1 = reshape(vgeo.ξ1x1, (Nq..., nelem))
    vgeo.ξ2x1 = reshape(vgeo.ξ2x1, (Nq..., nelem))
    vgeo.ξ1x2 = reshape(vgeo.ξ1x2, (Nq..., nelem))
    vgeo.ξ2x2 = reshape(vgeo.ξ2x2, (Nq..., nelem))
    nface = 4
    Nfp = div.(prod(Nq), Nq)
    sgeo.n1 = reshape(sgeo.n1, (maximum(Nfp), nface, nelem))
    sgeo.n2 = reshape(sgeo.n2, (maximum(Nfp), nface, nelem))
    sgeo.sMJ = reshape(sgeo.sMJ, (maximum(Nfp), nface, nelem))

    for e in 1:nelem
        for j in 1:Nq[2], i in 1:Nq[1]
            vgeo.x1ξ1 = vgeo.x1ξ2 = zero(T)
            vgeo.x2ξ1 = vgeo.x2ξ2 = zero(T)
            for n in 1:Nq[1]
                vgeo.x1ξ1 += D1[i, n] * vgeo.x1[n, j, e]
                vgeo.x2ξ1 += D1[i, n] * vgeo.x2[n, j, e]
            end
            for n in 1:Nq[2]
                vgeo.x1ξ2 += D2[j, n] * vgeo.x1[i, n, e]
                vgeo.x2ξ2 += D2[j, n] * vgeo.x2[i, n, e]
            end
            vgeo.JcV[i, j, e] = hypot(vgeo.x1ξ2, vgeo.x2ξ2)
            vgeo.MJ[i, j, e] = vgeo.x1ξ1 * vgeo.x2ξ2 - vgeo.x2ξ1 * vgeo.x1ξ2
            vgeo.ξ1x1[i, j, e] =  vgeo.x2ξ2 / vgeo.MJ[i, j, e]
            vgeo.ξ2x1[i, j, e] = -vgeo.x2ξ1 / vgeo.MJ[i, j, e]
            vgeo.ξ1x2[i, j, e] = -vgeo.x1ξ2 / vgeo.MJ[i, j, e]
            vgeo.ξ2x2[i, j, e] =  vgeo.x1ξ1 / vgeo.MJ[i, j, e]
        end

        for i in 1:maximum(Nfp)
            if i <= Nfp[1]
                sgeo.n1[i, 1, e] = -vgeo.MJ[1, i, e] * vgeo.ξ1x1[1, i, e]
                sgeo.n2[i, 1, e] = -vgeo.MJ[1, i, e] * vgeo.ξ1x2[1, i, e]
                sgeo.n1[i, 2, e] =  vgeo.MJ[Nq[1], i, e] * vgeo.ξ1x1[Nq[1], i, e]
                sgeo.n2[i, 2, e] =  vgeo.MJ[Nq[1], i, e] * vgeo.ξ1x2[Nq[1], i, e]
            else
                sgeo.n1[i, 1:2, e] .= NaN
                sgeo.n2[i, 1:2, e] .= NaN
            end
            if i <= Nfp[2]
                sgeo.n1[i, 3, e] = -vgeo.MJ[i, 1, e] * vgeo.ξ2x1[i, 1, e]
                sgeo.n2[i, 3, e] = -vgeo.MJ[i, 1, e] * vgeo.ξ2x2[i, 1, e]
                sgeo.n1[i, 4, e] =  vgeo.MJ[i, Nq[2], e] * vgeo.ξ2x1[i, Nq[2], e]
                sgeo.n2[i, 4, e] =  vgeo.MJ[i, Nq[2], e] * vgeo.ξ2x2[i, Nq[2], e]
            else
                sgeo.n1[i, 3:4, e] .= NaN
                sgeo.n2[i, 3:4, e] .= NaN
            end

            for n in 1:nface
                sgeo.sMJ[i, n, e] = hypot(sgeo.n1[i, n, e], sgeo.n2[i, n, e])
                sgeo.n1[i, n, e] /= sgeo.sMJ[i, n, e]
                sgeo.n2[i, n, e] /= sgeo.sMJ[i, n, e]
            end
        end
    end

    nothing
end

"""
    computemetric!(vgeo::VolumeGeometry{NTuple{3,Int}, <:AbstractArray}, sgeo::SurfaceGeometry{NTuple{3,Int}, <:AbstractArray}, D1, D2, D3)

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
`vgeo.x1`, `vgeo.x2`, `vgeo.x3`, `vgeo.MJ`, `vgeo.ξ1x1`, `vgeo.ξ2x1`, `vgeo.ξ3x1`,
`vgeo.ξ1x2`, `vgeo.ξ2x2`, `vgeo.ξ3x2`, `vgeo.ξ1x3`,`vgeo.ξ2x3`, and `vgeo.ξ3x3`
should all be of length `Nq^3 * nelem`.  Similarly, the face
arrays `sgeo.sMJ`, `sgeo.n1`, `sgeo.n2`, and `sgeo.n3` should be of size `Nq^2 * nface * nelem` with
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
    D3,
)
    T = eltype(vgeo.x1)

    Nq = (size(D1, 1), size(D2, 1), size(D3, 1))
    Np = prod(Nq)
    Nfp = div.(Np, Nq)
    nelem = div(length(vgeo.MJ), Np)

    vgeo.x1 = reshape(vgeo.x1, (Nq..., nelem))
    vgeo.x2 = reshape(vgeo.x2, (Nq..., nelem))
    vgeo.x3 = reshape(vgeo.x3, (Nq..., nelem))
    vgeo.MJ = reshape(vgeo.MJ, (Nq..., nelem))
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

    nface = 6
    sgeo.n1 = reshape(sgeo.n1, maximum(Nfp), nface, nelem)
    sgeo.n2 = reshape(sgeo.n2, maximum(Nfp), nface, nelem)
    sgeo.n3 = reshape(sgeo.n3, maximum(Nfp), nface, nelem)
    sgeo.sMJ = reshape(sgeo.sMJ, maximum(Nfp), nface, nelem)

    JI2 = similar(vgeo.MJ, Nq...)
    (yzr, yzs, yzt) = (similar(JI2), similar(JI2), similar(JI2))
    (zxr, zxs, zxt) = (similar(JI2), similar(JI2), similar(JI2))
    (xyr, xys, xyt) = (similar(JI2), similar(JI2), similar(JI2))

    vgeo.ξ1x1 .= zero(T)
    vgeo.ξ2x1 .= zero(T)
    vgeo.ξ3x1 .= zero(T)
    vgeo.ξ1x2 .= zero(T)
    vgeo.ξ2x2 .= zero(T)
    vgeo.ξ3x2 .= zero(T)
    vgeo.ξ1x3 .= zero(T)
    vgeo.ξ2x3 .= zero(T)
    vgeo.ξ3x3 .= zero(T)

    fill!(sgeo.n1, NaN)
    fill!(sgeo.n2, NaN)
    fill!(sgeo.n3, NaN)
    fill!(sgeo.sMJ, NaN)

    @inbounds for e in 1:nelem
        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]
            vgeo.x1ξ1 = vgeo.x1ξ2 = vgeo.x1ξ3 = zero(T)
            vgeo.x2ξ1 = vgeo.x2ξ2 = vgeo.x2ξ3 = zero(T)
            vgeo.x3ξ1 = vgeo.x3ξ2 = vgeo.x3ξ3 = zero(T)
            for n in 1:Nq[1]
                vgeo.x1ξ1 += D1[i, n] * vgeo.x1[n, j, k, e]
                vgeo.x2ξ1 += D1[i, n] * vgeo.x2[n, j, k, e]
                vgeo.x3ξ1 += D1[i, n] * vgeo.x3[n, j, k, e]
            end
            for n in 1:Nq[2]
                vgeo.x1ξ2 += D2[j, n] * vgeo.x1[i, n, k, e]
                vgeo.x2ξ2 += D2[j, n] * vgeo.x2[i, n, k, e]
                vgeo.x3ξ2 += D2[j, n] * vgeo.x3[i, n, k, e]
            end
            for n in 1:Nq[3]
                vgeo.x1ξ3 += D3[k, n] * vgeo.x1[i, j, n, e]
                vgeo.x2ξ3 += D3[k, n] * vgeo.x2[i, j, n, e]
                vgeo.x3ξ3 += D3[k, n] * vgeo.x3[i, j, n, e]
            end
            vgeo.JcV[i, j, k, e] = hypot(vgeo.x1ξ3, vgeo.x2ξ3, vgeo.x3ξ3)
            J[i, j, k, e] = (
                vgeo.x1ξ1 * (vgeo.x2ξ2 * vgeo.x3ξ3 - vgeo.x3ξ2 * vgeo.x2ξ3) +
                vgeo.x2ξ1 * (vgeo.x3ξ2 * vgeo.x1ξ3 - vgeo.x1ξ2 * vgeo.x3ξ3) +
                vgeo.x3ξ1 * (vgeo.x1ξ2 * vgeo.x2ξ3 - vgeo.x2ξ2 * vgeo.x1ξ3)
            )

            JI2[i, j, k] = 1 / (2 * vgeo.MJ[i, j, k, e])

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
            sgeo.n1[n, 1, e] = -vgeo.MJ[1, j, k, e] * vgeo.ξ1x1[1, j, k, e]
            sgeo.n2[n, 1, e] = -vgeo.MJ[1, j, k, e] * vgeo.ξ1x2[1, j, k, e]
            sgeo.n3[n, 1, e] = -vgeo.MJ[1, j, k, e] * vgeo.ξ1x3[1, j, k, e]
            sgeo.n1[n, 2, e] =  vgeo.MJ[Nq[1], j, k, e] * vgeo.ξ1x1[Nq[1], j, k, e]
            sgeo.n2[n, 2, e] =  vgeo.MJ[Nq[1], j, k, e] * vgeo.ξ1x2[Nq[1], j, k, e]
            sgeo.n3[n, 2, e] =  vgeo.MJ[Nq[1], j, k, e] * vgeo.ξ1x3[Nq[1], j, k, e]
            for f in 1:2
                sgeo.sMJ[n, f, e] = hypot(sgeo.n1[n, f, e], sgeo.n2[n, f, e], sgeo.n3[n, f, e])
                sgeo.n1[n, f, e] /= sgeo.sMJ[n, f, e]
                sgeo.n2[n, f, e] /= sgeo.sMJ[n, f, e]
                sgeo.n3[n, f, e] /= sgeo.sMJ[n, f, e]
            end
        end
        # faces 3 & 4
        for k in 1:Nq[3], i in 1:Nq[1]
            n = i + (k - 1) * Nq[1]
            sgeo.n1[n, 3, e] = -vgeo.MJ[i, 1, k, e] * vgeo.ξ2x1[i, 1, k, e]
            sgeo.n2[n, 3, e] = -vgeo.MJ[i, 1, k, e] * vgeo.ξ2x2[i, 1, k, e]
            sgeo.n3[n, 3, e] = -vgeo.MJ[i, 1, k, e] * vgeo.ξ2x3[i, 1, k, e]
            sgeo.n1[n, 4, e] =  vgeo.MJ[i, Nq[2], k, e] * vgeo.ξ2x1[i, Nq[2], k, e]
            sgeo.n2[n, 4, e] =  vgeo.MJ[i, Nq[2], k, e] * vgeo.ξ2x2[i, Nq[2], k, e]
            sgeo.n3[n, 4, e] =  vgeo.MJ[i, Nq[2], k, e] * vgeo.ξ2x3[i, Nq[2], k, e]
            for f in 3:4
                sgeo.sMJ[n, f, e] = hypot(sgeo.n1[n, f, e], sgeo.n2[n, f, e], sgeo.n3[n, f, e])
                sgeo.n1[n, f, e] /= sgeo.sMJ[n, f, e]
                sgeo.n2[n, f, e] /= sgeo.sMJ[n, f, e]
                sgeo.n3[n, f, e] /= sgeo.sMJ[n, f, e]
            end
        end
        # faces 5 & 6
        for j in 1:Nq[2], i in 1:Nq[1]
            n = i + (j - 1) * Nq[1]
            sgeo.n1[n, 5, e] = -vgeo.MJ[i, j, 1, e] * vgeo.ξ3x1[i, j, 1, e]
            sgeo.n2[n, 5, e] = -vgeo.MJ[i, j, 1, e] * vgeo.ξ3x2[i, j, 1, e]
            sgeo.n3[n, 5, e] = -vgeo.MJ[i, j, 1, e] * vgeo.ξ3x3[i, j, 1, e]
            sgeo.n1[n, 6, e] =  vgeo.MJ[i, j, Nq[3], e] * vgeo.ξ3x1[i, j, Nq[3], e]
            sgeo.n2[n, 6, e] =  vgeo.MJ[i, j, Nq[3], e] * vgeo.ξ3x2[i, j, Nq[3], e]
            sgeo.n3[n, 6, e] =  vgeo.MJ[i, j, Nq[3], e] * vgeo.ξ3x3[i, j, Nq[3], e]
            for f in 5:6
                sgeo.sMJ[n, f, e] = hypot(sgeo.n1[n, f, e], sgeo.n2[n, f, e], sgeo.n3[n, f, e])
                sgeo.n1[n, f, e] /= sgeo.sMJ[n, f, e]
                sgeo.n2[n, f, e] /= sgeo.sMJ[n, f, e]
                sgeo.n3[n, f, e] /= sgeo.sMJ[n, f, e]
            end
        end
    end

    nothing
end

"""
  computemetric(x1::AbstractArray{T, 2}, D::AbstractMatrix{T}) where T

Compute the 1-D metric terms from the element grid array `x1` using the
derivative matrix `D`. The derivative matrix `D` should be consistent with the
reference grid `ξ1` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

- `J` the Jacobian determinant
- `ξ1x1` derivative ∂ξ1 / ∂x1'
- `sJ` the surface Jacobian
- 'n1` outward pointing unit normal in \$x1\$-direction
"""
function computemetric(x1::AbstractArray{T, 2}, D::AbstractMatrix{T}) where {T}

    Nq = size(D, 1)
    nelem = size(x1, 2)
    nface = 2

    J = similar(x1)
    ξ1x1 = similar(x1)

    sJ = Array{T, 3}(undef, 1, nface, nelem)
    n1 = Array{T, 3}(undef, 1, nface, nelem)

    computemetric!(x1, J, JcV, ξ1x1, sJ, n1, D)

    (J = J, JcV = JcV, ξ1x1 = ξ1x1, sJ = sJ, n1 = n1)
end


"""
  computemetric(x1::AbstractArray{T, 3}, x2::AbstractArray{T, 3},
                D::AbstractMatrix{T}) where T

Compute the 2-D metric terms from the element grid arrays `x1` and `x2` using
the derivative matrix `D`. The derivative matrix `D` should be consistent with
the reference grid `ξ1` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

- `J` the Jacobian determinant
- `ξ1x1` derivative ∂ξ1 / ∂x1'
- `ξ2x1` derivative ∂ξ2 / ∂x1'
 - `ξ1x2` derivative ∂ξ1 / ∂x2'
 - `ξ2x2` derivative ∂ξ2 / ∂x2'
 - `sJ` the surface Jacobian
 - 'n1` outward pointing unit normal in \$x1\$-direction
 - 'n2` outward pointing unit normal in \$x2\$-direction
"""
function computemetric(
    x1::AbstractArray{T, 3},
    x2::AbstractArray{T, 3},
    D::AbstractMatrix{T},
) where {T}
    @assert size(x1) == size(x2)
    Nq = size(D, 1)
    nelem = size(x1, 3)
    nface = 4

    J = similar(x1)
    JcV = similar(x1)
    ξ1x1 = similar(x1)
    ξ2x1 = similar(x1)
    ξ1x2 = similar(x1)
    ξ2x2 = similar(x1)

    sJ = Array{T, 3}(undef, Nq, nface, nelem)
    n1 = Array{T, 3}(undef, Nq, nface, nelem)
    n2 = Array{T, 3}(undef, Nq, nface, nelem)

    computemetric!(x1, x2, J, JcV, ξ1x1, ξ2x1, ξ1x2, ξ2x2, sJ, n1, n2, D)

    (
        J = J,
        JcV = JcV,
        ξ1x1 = ξ1x1,
        ξ2x1 = ξ2x1,
        ξ1x2 = ξ1x2,
        ξ2x2 = ξ2x2,
        sJ = sJ,
        n1 = n1,
        n2 = n2,
    )
end

"""
    computemetric(x1::AbstractArray{T, 3}, x2::AbstractArray{T, 3},
                  D::AbstractMatrix{T}) where T

Compute the 3-D metric terms from the element grid arrays `x1`, `x2`, and `x3`
using the derivative matrix `D`. The derivative matrix `D` should be consistent
with the reference grid `ξ1` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξ1x1` derivative ∂ξ1 / ∂x1'
 - `ξ2x1` derivative ∂ξ2 / ∂x1'
 - `ξ3x1` derivative ∂ξ3 / ∂x1'
 - `ξ1x2` derivative ∂ξ1 / ∂x2'
 - `ξ2x2` derivative ∂ξ2 / ∂x2'
 - `ξ3x2` derivative ∂ξ3 / ∂x2'
 - `ξ1x3` derivative ∂ξ1 / ∂x3'
 - `ξ2x3` derivative ∂ξ2 / ∂x3'
 - `ξ3x3` derivative ∂ξ3 / ∂x3'
 - `sJ` the surface Jacobian
 - 'n1` outward pointing unit normal in \$x1\$-direction
 - 'n2` outward pointing unit normal in \$x2\$-direction
 - 'n3` outward pointing unit normal in \$x3\$-direction

!!! note

   The storage of the volume terms and surface terms from this function are
   slightly different. The volume terms used Cartesian indexing whereas the
   surface terms use linear indexing.
"""
function computemetric(
    x1::AbstractArray{T, 4},
    x2::AbstractArray{T, 4},
    x3::AbstractArray{T, 4},
    D::AbstractMatrix{T},
) where {T}

    @assert size(x1) == size(x2) == size(x3)
    Nq = size(D, 1)
    nelem = size(x1, 4)
    nface = 6

    J = similar(x1)
    JcV = similar(x1)
    ξ1x1 = similar(x1)
    ξ2x1 = similar(x1)
    ξ3x1 = similar(x1)
    ξ1x2 = similar(x1)
    ξ2x2 = similar(x1)
    ξ3x2 = similar(x1)
    ξ1x3 = similar(x1)
    ξ2x3 = similar(x1)
    ξ3x3 = similar(x1)

    sJ = Array{T, 3}(undef, Nq^2, nface, nelem)
    n1 = Array{T, 3}(undef, Nq^2, nface, nelem)
    n2 = Array{T, 3}(undef, Nq^2, nface, nelem)
    n3 = Array{T, 3}(undef, Nq^2, nface, nelem)

    computemetric!(
        x1,
        x2,
        x3,
        J,
        JcV,
        ξ1x1,
        ξ2x1,
        ξ3x1,
        ξ1x2,
        ξ2x2,
        ξ3x2,
        ξ1x3,
        ξ2x3,
        ξ3x3,
        sJ,
        n1,
        n2,
        n3,
        D,
    )

    (
        J = J,
        JcV = JcV,
        ξ1x1 = ξ1x1,
        ξ2x1 = ξ2x1,
        ξ3x1 = ξ3x1,
        ξ1x2 = ξ1x2,
        ξ2x2 = ξ2x2,
        ξ3x2 = ξ3x2,
        ξ1x3 = ξ1x3,
        ξ2x3 = ξ2x3,
        ξ3x3 = ξ3x3,
        sJ = sJ,
        n1 = n1,
        n2 = n2,
        n3 = n3,
    )
end

end # module
