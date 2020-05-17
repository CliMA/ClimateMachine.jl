module Metrics

"""
    creategrid!(x1, elemtocoord, r)

Create a 1-D grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D
`(-1, 1)` reference coordinates `r`. The element grids are filled using linear
interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
array `x1` should be `Nq * nelem == length(x1)`.
"""
function creategrid!(x1, e2c, r)
    (d, nvert, nelem) = size(e2c)
    @assert d == 1
    Nq = length(r)
    x1 = reshape(x1, (Nq, nelem))

    # linear blend
    @inbounds for e in 1:nelem
        for i in 1:Nq
            x1[i, e] =
                ((1 - r[i]) * e2c[1, 1, e] + (1 + r[i]) * e2c[1, 2, e]) / 2
        end
    end
    nothing
end

"""
    creategrid!(x1, x2, elemtocoord, r)

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x1` and `x2` should be `Nq^2 * nelem == size(x1) == size(x2)`.
"""
function creategrid!(x1, x2, e2c, r)
    (d, nvert, nelem) = size(e2c)
    @assert d == 2
    Nq = length(r)
    x1 = reshape(x1, (Nq, Nq, nelem))
    x2 = reshape(x2, (Nq, Nq, nelem))

    # bilinear blend of corners
    @inbounds for (f, n) in zip((x1, x2), 1:d)
        for e in 1:nelem
            for j in 1:Nq
                for i in 1:Nq
                    f[i, j, e] =
                        (
                            (1 - r[i]) * (1 - r[j]) * e2c[n, 1, e] +
                            (1 + r[i]) * (1 - r[j]) * e2c[n, 2, e] +
                            (1 - r[i]) * (1 + r[j]) * e2c[n, 3, e] +
                            (1 + r[i]) * (1 + r[j]) * e2c[n, 4, e]
                        ) / 4
                end
            end
        end
    end
    nothing
end

"""
    creategrid!(x1, x2, x3, elemtocoord, r)

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using trilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x1`, `x2`, and `x3` should be `Nq^3 * nelem == size(x1) == size(x2) ==
size(x3)`.
"""
function creategrid!(x1, x2, x3, e2c, r)
    (d, nvert, nelem) = size(e2c)
    @assert d == 3
    # TODO: Add asserts?
    Nq = length(r)
    x1 = reshape(x1, (Nq, Nq, Nq, nelem))
    x2 = reshape(x2, (Nq, Nq, Nq, nelem))
    x3 = reshape(x3, (Nq, Nq, Nq, nelem))

    # trilinear blend of corners
    @inbounds for (f, n) in zip((x1, x2, x3), 1:d)
        for e in 1:nelem
            for k in 1:Nq
                for j in 1:Nq
                    for i in 1:Nq
                        f[i, j, k, e] =
                            (
                                (1 - r[i]) *
                                (1 - r[j]) *
                                (1 - r[k]) *
                                e2c[n, 1, e] +
                                (1 + r[i]) *
                                (1 - r[j]) *
                                (1 - r[k]) *
                                e2c[n, 2, e] +
                                (1 - r[i]) *
                                (1 + r[j]) *
                                (1 - r[k]) *
                                e2c[n, 3, e] +
                                (1 + r[i]) *
                                (1 + r[j]) *
                                (1 - r[k]) *
                                e2c[n, 4, e] +
                                (1 - r[i]) *
                                (1 - r[j]) *
                                (1 + r[k]) *
                                e2c[n, 5, e] +
                                (1 + r[i]) *
                                (1 - r[j]) *
                                (1 + r[k]) *
                                e2c[n, 6, e] +
                                (1 - r[i]) *
                                (1 + r[j]) *
                                (1 + r[k]) *
                                e2c[n, 7, e] +
                                (1 + r[i]) *
                                (1 + r[j]) *
                                (1 + r[k]) *
                                e2c[n, 8, e]
                            ) / 8
                    end
                end
            end
        end
    end
    nothing
end

"""
    computemetric!(x1, J, ξ1x1, sJ, n1, D)

Compute the 1-D metric terms from the element grid arrays `x1`. All the arrays
are preallocated by the user and the (square) derivative matrix `D` should be
consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x1), Nq)` then the volume arrays
`x1`, `J`, and `ξ1x1` should all have length `Nq * nelem`.  Similarly, the face
arrays `sJ` and `n1` should be of length `nface * nelem` with `nface = 2`.
"""
function computemetric!(x1, J, ξ1x1, sJ, n1, D)
    Nq = size(D, 1)
    nelem = div(length(J), Nq)
    x1 = reshape(x1, (Nq, nelem))
    J = reshape(J, (Nq, nelem))
    ξ1x1 = reshape(ξ1x1, (Nq, nelem))
    nface = 2
    n1 = reshape(n1, (1, nface, nelem))
    sJ = reshape(sJ, (1, nface, nelem))

    d = 1

    @inbounds for e in 1:nelem
        J[:, e] = D * x1[:, e]
    end
    ξ1x1 .= 1 ./ J

    n1[1, 1, :] .= -sign.(J[1, :])
    n1[1, 2, :] .= sign.(J[Nq, :])
    sJ .= 1
    nothing
end

"""
    computemetric!(x1, x2, J, ξ1x1, ξ2x1, ξ1x2, ξ2x2, sJ, n1, n2, D)

Compute the 2-D metric terms from the element grid arrays `x1` and `x2`. All the
arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x1), Nq^2)` then the volume arrays
`x1`, `x2`, `J`, `ξ1x1`, `ξ2x1`, `ξ1x2`, and `ξ2x2` should all be of size `(Nq, Nq,
nelem)`.  Similarly, the face arrays `sJ`, `n1`, and `n2` should be of size
`(Nq, nface, nelem)` with `nface = 4`.
"""
function computemetric!(x1, x2, J, ξ1x1, ξ2x1, ξ1x2, ξ2x2, sJ, n1, n2, D)
    T = eltype(x1)
    Nq = size(D, 1)
    nelem = div(length(J), Nq^2)
    d = 2
    x1 = reshape(x1, (Nq, Nq, nelem))
    x2 = reshape(x2, (Nq, Nq, nelem))
    J = reshape(J, (Nq, Nq, nelem))
    ξ1x1 = reshape(ξ1x1, (Nq, Nq, nelem))
    ξ2x1 = reshape(ξ2x1, (Nq, Nq, nelem))
    ξ1x2 = reshape(ξ1x2, (Nq, Nq, nelem))
    ξ2x2 = reshape(ξ2x2, (Nq, Nq, nelem))
    nface = 4
    n1 = reshape(n1, (Nq, nface, nelem))
    n2 = reshape(n2, (Nq, nface, nelem))
    sJ = reshape(sJ, (Nq, nface, nelem))

    @inbounds for e in 1:nelem
        for j in 1:Nq, i in 1:Nq
            x1ξ1 = x1ξ2 = zero(T)
            x2ξ1 = x2ξ2 = zero(T)
            for n in 1:Nq
                x1ξ1 += D[i, n] * x1[n, j, e]
                x1ξ2 += D[j, n] * x1[i, n, e]
                x2ξ1 += D[i, n] * x2[n, j, e]
                x2ξ2 += D[j, n] * x2[i, n, e]
            end
            J[i, j, e] = x1ξ1 * x2ξ2 - x2ξ1 * x1ξ2
            ξ1x1[i, j, e] = x2ξ2 / J[i, j, e]
            ξ2x1[i, j, e] = -x2ξ1 / J[i, j, e]
            ξ1x2[i, j, e] = -x1ξ2 / J[i, j, e]
            ξ2x2[i, j, e] = x1ξ1 / J[i, j, e]
        end

        for i in 1:Nq
            n1[i, 1, e] = -J[1, i, e] * ξ1x1[1, i, e]
            n2[i, 1, e] = -J[1, i, e] * ξ1x2[1, i, e]
            n1[i, 2, e] = J[Nq, i, e] * ξ1x1[Nq, i, e]
            n2[i, 2, e] = J[Nq, i, e] * ξ1x2[Nq, i, e]
            n1[i, 3, e] = -J[i, 1, e] * ξ2x1[i, 1, e]
            n2[i, 3, e] = -J[i, 1, e] * ξ2x2[i, 1, e]
            n1[i, 4, e] = J[i, Nq, e] * ξ2x1[i, Nq, e]
            n2[i, 4, e] = J[i, Nq, e] * ξ2x2[i, Nq, e]

            for n in 1:4
                sJ[i, n, e] = hypot(n1[i, n, e], n2[i, n, e])
                n1[i, n, e] /= sJ[i, n, e]
                n2[i, n, e] /= sJ[i, n, e]
            end
        end
    end

    nothing
end

"""
    computemetric!(x1, x2, x3, J, ξ1x1, ξ2x1, ξ3x1, ξ1x2, ξ2x2, ξ3x2, ξ1x3,
                   ξ2x3, ξ3x3, sJ, n1, n2, n3, D)

Compute the 3-D metric terms from the element grid arrays `x1`, `x2`, and `x3`.
All the arrays are preallocated by the user and the (square) derivative matrix
`D` should be consistent with the reference grid `r` used in
[`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x1), Nq^3)` then the volume arrays
`x1`, `x2`, `x3`, `J`, `ξ1x1`, `ξ2x1`, `ξ3x1`, `ξ1x2`, `ξ2x2`, `ξ3x2`, `ξ1x3`,
`ξ2x3`, and `ξ3x3` should all be of length `Nq^3 * nelem`.  Similarly, the face
arrays `sJ`, `n1`, `n2`, and `n3` should be of size `Nq^2 * nface * nelem` with
`nface = 6`.

The curl invariant formulation of Kopriva (2006), equation 37, is used.

Reference:
  Kopriva, David A. "Metric identities and the discontinuous spectral element
  method on curvilinear meshes." Journal of Scientific Computing 26.3 (2006):
  301-327. <https://doi.org/10.1007/s10915-005-9070-8>
"""
function computemetric!(
    x1,
    x2,
    x3,
    J,
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
    T = eltype(x1)

    Nq = size(D, 1)
    nelem = div(length(J), Nq^3)

    x1 = reshape(x1, (Nq, Nq, Nq, nelem))
    x2 = reshape(x2, (Nq, Nq, Nq, nelem))
    x3 = reshape(x3, (Nq, Nq, Nq, nelem))
    J = reshape(J, (Nq, Nq, Nq, nelem))
    ξ1x1 = reshape(ξ1x1, (Nq, Nq, Nq, nelem))
    ξ2x1 = reshape(ξ2x1, (Nq, Nq, Nq, nelem))
    ξ3x1 = reshape(ξ3x1, (Nq, Nq, Nq, nelem))
    ξ1x2 = reshape(ξ1x2, (Nq, Nq, Nq, nelem))
    ξ2x2 = reshape(ξ2x2, (Nq, Nq, Nq, nelem))
    ξ3x2 = reshape(ξ3x2, (Nq, Nq, Nq, nelem))
    ξ1x3 = reshape(ξ1x3, (Nq, Nq, Nq, nelem))
    ξ2x3 = reshape(ξ2x3, (Nq, Nq, Nq, nelem))
    ξ3x3 = reshape(ξ3x3, (Nq, Nq, Nq, nelem))

    nface = 6
    #= This code is broken when views are used
    n1 = reshape(n1, Nq, Nq, nface, nelem)
    n2 = reshape(n2, Nq, Nq, nface, nelem)
    n3 = reshape(n3, Nq, Nq, nface, nelem)
    sJ = reshape(sJ, Nq, Nq, nface, nelem)
    =#

    JI2 = similar(J, (Nq, Nq, Nq))
    (yzr, yzs, yzt) = (similar(JI2), similar(JI2), similar(JI2))
    (zxr, zxs, zxt) = (similar(JI2), similar(JI2), similar(JI2))
    (xyr, xys, xyt) = (similar(JI2), similar(JI2), similar(JI2))

    ξ1x1 .= zero(T)
    ξ2x1 .= zero(T)
    ξ3x1 .= zero(T)
    ξ1x2 .= zero(T)
    ξ2x2 .= zero(T)
    ξ3x2 .= zero(T)
    ξ1x3 .= zero(T)
    ξ2x3 .= zero(T)
    ξ3x3 .= zero(T)

    @inbounds for e in 1:nelem
        for k in 1:Nq, j in 1:Nq, i in 1:Nq
            x1ξ1 = x1ξ2 = x1ξ3 = zero(T)
            x2ξ1 = x2ξ2 = x2ξ3 = zero(T)
            x3ξ1 = x3ξ2 = x3ξ3 = zero(T)
            for n in 1:Nq
                x1ξ1 += D[i, n] * x1[n, j, k, e]
                x1ξ2 += D[j, n] * x1[i, n, k, e]
                x1ξ3 += D[k, n] * x1[i, j, n, e]
                x2ξ1 += D[i, n] * x2[n, j, k, e]
                x2ξ2 += D[j, n] * x2[i, n, k, e]
                x2ξ3 += D[k, n] * x2[i, j, n, e]
                x3ξ1 += D[i, n] * x3[n, j, k, e]
                x3ξ2 += D[j, n] * x3[i, n, k, e]
                x3ξ3 += D[k, n] * x3[i, j, n, e]
            end
            J[i, j, k, e] = (
                x1ξ1 * (x2ξ2 * x3ξ3 - x3ξ2 * x2ξ3) +
                x2ξ1 * (x3ξ2 * x1ξ3 - x1ξ2 * x3ξ3) +
                x3ξ1 * (x1ξ2 * x2ξ3 - x2ξ2 * x1ξ3)
            )

            JI2[i, j, k] = 1 / (2 * J[i, j, k, e])

            yzr[i, j, k] = x2[i, j, k, e] * x3ξ1 - x3[i, j, k, e] * x2ξ1
            yzs[i, j, k] = x2[i, j, k, e] * x3ξ2 - x3[i, j, k, e] * x2ξ2
            yzt[i, j, k] = x2[i, j, k, e] * x3ξ3 - x3[i, j, k, e] * x2ξ3
            zxr[i, j, k] = x3[i, j, k, e] * x1ξ1 - x1[i, j, k, e] * x3ξ1
            zxs[i, j, k] = x3[i, j, k, e] * x1ξ2 - x1[i, j, k, e] * x3ξ2
            zxt[i, j, k] = x3[i, j, k, e] * x1ξ3 - x1[i, j, k, e] * x3ξ3
            xyr[i, j, k] = x1[i, j, k, e] * x2ξ1 - x2[i, j, k, e] * x1ξ1
            xys[i, j, k] = x1[i, j, k, e] * x2ξ2 - x2[i, j, k, e] * x1ξ2
            xyt[i, j, k] = x1[i, j, k, e] * x2ξ3 - x2[i, j, k, e] * x1ξ3
        end

        for k in 1:Nq, j in 1:Nq, i in 1:Nq
            for n in 1:Nq
                ξ1x1[i, j, k, e] += D[j, n] * yzt[i, n, k]
                ξ1x1[i, j, k, e] -= D[k, n] * yzs[i, j, n]
                ξ2x1[i, j, k, e] += D[k, n] * yzr[i, j, n]
                ξ2x1[i, j, k, e] -= D[i, n] * yzt[n, j, k]
                ξ3x1[i, j, k, e] += D[i, n] * yzs[n, j, k]
                ξ3x1[i, j, k, e] -= D[j, n] * yzr[i, n, k]
                ξ1x2[i, j, k, e] += D[j, n] * zxt[i, n, k]
                ξ1x2[i, j, k, e] -= D[k, n] * zxs[i, j, n]
                ξ2x2[i, j, k, e] += D[k, n] * zxr[i, j, n]
                ξ2x2[i, j, k, e] -= D[i, n] * zxt[n, j, k]
                ξ3x2[i, j, k, e] += D[i, n] * zxs[n, j, k]
                ξ3x2[i, j, k, e] -= D[j, n] * zxr[i, n, k]
                ξ1x3[i, j, k, e] += D[j, n] * xyt[i, n, k]
                ξ1x3[i, j, k, e] -= D[k, n] * xys[i, j, n]
                ξ2x3[i, j, k, e] += D[k, n] * xyr[i, j, n]
                ξ2x3[i, j, k, e] -= D[i, n] * xyt[n, j, k]
                ξ3x3[i, j, k, e] += D[i, n] * xys[n, j, k]
                ξ3x3[i, j, k, e] -= D[j, n] * xyr[i, n, k]
            end
            ξ1x1[i, j, k, e] *= JI2[i, j, k]
            ξ2x1[i, j, k, e] *= JI2[i, j, k]
            ξ3x1[i, j, k, e] *= JI2[i, j, k]
            ξ1x2[i, j, k, e] *= JI2[i, j, k]
            ξ2x2[i, j, k, e] *= JI2[i, j, k]
            ξ3x2[i, j, k, e] *= JI2[i, j, k]
            ξ1x3[i, j, k, e] *= JI2[i, j, k]
            ξ2x3[i, j, k, e] *= JI2[i, j, k]
            ξ3x3[i, j, k, e] *= JI2[i, j, k]
        end

        for j in 1:Nq, i in 1:Nq
            #= This code is broken when views are used
            n1[i, j, 1, e] = -J[ 1, i, j, e] * ξ1x1[ 1, i, j, e]
            n1[i, j, 2, e] =  J[Nq, i, j, e] * ξ1x1[Nq, i, j, e]
            n1[i, j, 3, e] = -J[ i, 1, j, e] * ξ2x1[ i, 1, j, e]
            n1[i, j, 4, e] =  J[ i,Nq, j, e] * ξ2x1[ i,Nq, j, e]
            n1[i, j, 5, e] = -J[ i, j, 1, e] * ξ3x1[ i, j, 1, e]
            n1[i, j, 6, e] =  J[ i, j,Nq, e] * ξ3x1[ i, j,Nq, e]
            n2[i, j, 1, e] = -J[ 1, i, j, e] * ξ1x2[ 1, i, j, e]
            n2[i, j, 2, e] =  J[Nq, i, j, e] * ξ1x2[Nq, i, j, e]
            n2[i, j, 3, e] = -J[ i, 1, j, e] * ξ2x2[ i, 1, j, e]
            n2[i, j, 4, e] =  J[ i,Nq, j, e] * ξ2x2[ i,Nq, j, e]
            n2[i, j, 5, e] = -J[ i, j, 1, e] * ξ3x2[ i, j, 1, e]
            n2[i, j, 6, e] =  J[ i, j,Nq, e] * ξ3x2[ i, j,Nq, e]
            n3[i, j, 1, e] = -J[ 1, i, j, e] * ξ1x3[ 1, i, j, e]
            n3[i, j, 2, e] =  J[Nq, i, j, e] * ξ1x3[Nq, i, j, e]
            n3[i, j, 3, e] = -J[ i, 1, j, e] * ξ2x3[ i, 1, j, e]
            n3[i, j, 4, e] =  J[ i,Nq, j, e] * ξ2x3[ i,Nq, j, e]
            n3[i, j, 5, e] = -J[ i, j, 1, e] * ξ3x3[ i, j, 1, e]
            n3[i, j, 6, e] =  J[ i, j,Nq, e] * ξ3x3[ i, j,Nq, e]

            for n = 1:6
              sJ[i, j, n, e] = hypot(n1[i, j, n, e], n2[i, j, n, e], n3[i, j, n, e])
              n1[i, j, n, e] /= sJ[i, j, n, e]
              n2[i, j, n, e] /= sJ[i, j, n, e]
              n3[i, j, n, e] /= sJ[i, j, n, e]
            end
            =#

            ije = i + (j - 1) * Nq + (e - 1) * nface * Nq^2
            n1[ije + (1 - 1) * Nq^2] = -J[1, i, j, e] * ξ1x1[1, i, j, e]
            n1[ije + (2 - 1) * Nq^2] = J[Nq, i, j, e] * ξ1x1[Nq, i, j, e]
            n1[ije + (3 - 1) * Nq^2] = -J[i, 1, j, e] * ξ2x1[i, 1, j, e]
            n1[ije + (4 - 1) * Nq^2] = J[i, Nq, j, e] * ξ2x1[i, Nq, j, e]
            n1[ije + (5 - 1) * Nq^2] = -J[i, j, 1, e] * ξ3x1[i, j, 1, e]
            n1[ije + (6 - 1) * Nq^2] = J[i, j, Nq, e] * ξ3x1[i, j, Nq, e]
            n2[ije + (1 - 1) * Nq^2] = -J[1, i, j, e] * ξ1x2[1, i, j, e]
            n2[ije + (2 - 1) * Nq^2] = J[Nq, i, j, e] * ξ1x2[Nq, i, j, e]
            n2[ije + (3 - 1) * Nq^2] = -J[i, 1, j, e] * ξ2x2[i, 1, j, e]
            n2[ije + (4 - 1) * Nq^2] = J[i, Nq, j, e] * ξ2x2[i, Nq, j, e]
            n2[ije + (5 - 1) * Nq^2] = -J[i, j, 1, e] * ξ3x2[i, j, 1, e]
            n2[ije + (6 - 1) * Nq^2] = J[i, j, Nq, e] * ξ3x2[i, j, Nq, e]
            n3[ije + (1 - 1) * Nq^2] = -J[1, i, j, e] * ξ1x3[1, i, j, e]
            n3[ije + (2 - 1) * Nq^2] = J[Nq, i, j, e] * ξ1x3[Nq, i, j, e]
            n3[ije + (3 - 1) * Nq^2] = -J[i, 1, j, e] * ξ2x3[i, 1, j, e]
            n3[ije + (4 - 1) * Nq^2] = J[i, Nq, j, e] * ξ2x3[i, Nq, j, e]
            n3[ije + (5 - 1) * Nq^2] = -J[i, j, 1, e] * ξ3x3[i, j, 1, e]
            n3[ije + (6 - 1) * Nq^2] = J[i, j, Nq, e] * ξ3x3[i, j, Nq, e]

            for n in 1:6
                sJ[ije + (n - 1) * Nq^2] = hypot(
                    n1[ije + (n - 1) * Nq^2],
                    n2[ije + (n - 1) * Nq^2],
                    n3[ije + (n - 1) * Nq^2],
                )
                n1[ije + (n - 1) * Nq^2] /= sJ[ije + (n - 1) * Nq^2]
                n2[ije + (n - 1) * Nq^2] /= sJ[ije + (n - 1) * Nq^2]
                n3[ije + (n - 1) * Nq^2] /= sJ[ije + (n - 1) * Nq^2]
            end
        end
    end

    nothing
end

creategrid1d(elemtocoord, r) = creategrid(Val(1), elemtocoord, r)
creategrid2d(elemtocoord, r) = creategrid(Val(2), elemtocoord, r)
creategrid3d(elemtocoord, r) = creategrid(Val(3), elemtocoord, r)

"""
    creategrid(::Val{1}, elemtocoord::AbstractArray{S, 3},
               r::AbstractVector{T}) where {S, T}

Create a grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D `(-1,
1)` reference coordinates `r`. The element grids are filled using bilinear
interpolation of the element coordinates.

The grid is returned as a tuple of with `x1` array
"""
function creategrid(
    ::Val{1},
    e2c::AbstractArray{S, 3},
    r::AbstractVector{T},
) where {S, T}
    (d, nvert, nelem) = size(e2c)
    @assert d == 1
    Nq = length(r)
    x1 = Array{T, 2}(undef, Nq, nelem)
    creategrid!(x1, e2c, r)
    (x1 = x1,)
end

"""
  creategrid(::Val{2}, elemtocoord::AbstractArray{S, 3},
             r::AbstractVector{T}) where {S, T}

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

The grid is returned as a tuple of the `x1` and `x2` arrays
"""
function creategrid(
    ::Val{2},
    e2c::AbstractArray{S, 3},
    r::AbstractVector{T},
) where {S, T}
    (d, nvert, nelem) = size(e2c)
    @assert d == 2
    Nq = length(r)
    x1 = Array{T, 3}(undef, Nq, Nq, nelem)
    x2 = Array{T, 3}(undef, Nq, Nq, nelem)
    creategrid!(x1, x2, e2c, r)
    (x1 = x1, x2 = x2)
end

"""
  creategrid(::Val{3}, elemtocoord::AbstractArray{S, 3},
             r::AbstractVector{T}) where {S, T}

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

The grid is returned as a tuple of the `x1`, `x2`, `x3` arrays
"""
function creategrid(
    ::Val{3},
    e2c::AbstractArray{S, 3},
    r::AbstractVector{T},
) where {S, T}
    (d, nvert, nelem) = size(e2c)
    @assert d == 3
    Nq = length(r)
    x1 = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
    x2 = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
    x3 = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
    creategrid!(x1, x2, x3, e2c, r)
    (x1 = x1, x2 = x2, x3 = x3)
end

"""
  computemetric(x1::AbstractArray{T, 2}, D::AbstractMatrix{T}) where T

Compute the 1-D metric terms from the element grid array `x1` using the
derivative matrix `D`. The derivative matrix `D` should be consistent with the
reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

- `J` the Jacobian determinant
- `ξ1x1` derivative ∂r / ∂x1'
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

    computemetric!(x1, J, ξ1x1, sJ, n1, D)

    (J = J, ξ1x1 = ξ1x1, sJ = sJ, n1 = n1)
end


"""
  computemetric(x1::AbstractArray{T, 3}, x2::AbstractArray{T, 3},
                D::AbstractMatrix{T}) where T

Compute the 2-D metric terms from the element grid arrays `x1` and `x2` using
the derivative matrix `D`. The derivative matrix `D` should be consistent with
the reference grid `r` used in [`creategrid!`](@ref).

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
    ξ1x1 = similar(x1)
    ξ2x1 = similar(x1)
    ξ1x2 = similar(x1)
    ξ2x2 = similar(x1)

    sJ = Array{T, 3}(undef, Nq, nface, nelem)
    n1 = Array{T, 3}(undef, Nq, nface, nelem)
    n2 = Array{T, 3}(undef, Nq, nface, nelem)

    computemetric!(x1, x2, J, ξ1x1, ξ2x1, ξ1x2, ξ2x2, sJ, n1, n2, D)

    (
        J = J,
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
with the reference grid `r` used in [`creategrid!`](@ref).

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
