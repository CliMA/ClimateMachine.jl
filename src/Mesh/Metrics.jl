module Metrics

"""
    creategrid!(x, elemtocoord, r)

Create a 1-D grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D
`(-1, 1)` reference coordinates `r`. The element grids are filled using linear
interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
array `x` should be `Nq * nelem == length(x)`.
"""
function creategrid!(x, e2c, r)
  (d, nvert, nelem) = size(e2c)
  @assert d == 1
  Nq = length(r)
  x = reshape(x, (Nq, nelem))

  # linear blend
  @inbounds for e = 1:nelem
    for i = 1:Nq
      x[i, e] = ((1 - r[i]) * e2c[1, 1, e] + (1 + r[i])e2c[1, 2, e]) / 2
    end
  end
  nothing
end

"""
    creategrid!(x, y, elemtocoord, r)

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x` and `y` should be `Nq^2 * nelem == size(x) == size(y)`.
"""
function creategrid!(x, y, e2c, r)
  (d, nvert, nelem) = size(e2c)
  @assert d == 2
  Nq = length(r)
  x = reshape(x, (Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, nelem))

  # bilinear blend of corners
  @inbounds for (f, n) = zip((x, y), 1:d)
    for e = 1:nelem
      for j = 1:Nq
        for i = 1:Nq
          f[i, j, e] = ((1 - r[i]) * (1 - r[j]) * e2c[n, 1, e] +
                        (1 + r[i]) * (1 - r[j]) * e2c[n, 2, e] +
                        (1 - r[i]) * (1 + r[j]) * e2c[n, 3, e] +
                        (1 + r[i]) * (1 + r[j]) * e2c[n, 4, e]) / 4
        end
      end
    end
  end
  nothing
end

"""
    creategrid!(x, y, z, elemtocoord, r)

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using trilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x`, `y`, and `z` should be `Nq^3 * nelem == size(x) == size(y) ==
size(z)`.
"""
function creategrid!(x, y, z, e2c, r)
  (d, nvert, nelem) = size(e2c)
  @assert d == 3
  # TODO: Add asserts?
  Nq = length(r)
  x = reshape(x, (Nq, Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, Nq, nelem))
  z = reshape(z, (Nq, Nq, Nq, nelem))

  # trilinear blend of corners
  @inbounds for (f, n) = zip((x,y,z), 1:d)
    for e = 1:nelem
      for k = 1:Nq
        for j = 1:Nq
          for i = 1:Nq
            f[i, j, k, e] = ((1 - r[i]) * (1 - r[j]) *
                               (1 - r[k]) * e2c[n, 1, e] +
                             (1 + r[i]) * (1 - r[j]) *
                               (1 - r[k]) * e2c[n, 2, e] +
                             (1 - r[i]) * (1 + r[j]) *
                               (1 - r[k]) * e2c[n, 3, e] +
                             (1 + r[i]) * (1 + r[j]) *
                               (1 - r[k]) * e2c[n, 4, e] +
                             (1 - r[i]) * (1 - r[j]) *
                               (1 + r[k]) * e2c[n, 5, e] +
                             (1 + r[i]) * (1 - r[j]) *
                               (1 + r[k]) * e2c[n, 6, e] +
                             (1 - r[i]) * (1 + r[j]) *
                               (1 + r[k]) * e2c[n, 7, e] +
                             (1 + r[i]) * (1 + r[j]) *
                               (1 + r[k]) * e2c[n, 8, e]) / 8
          end
        end
      end
    end
  end
  nothing
end

"""
    computemetric!(x, J, ξx, sJ, nx, D)

Compute the 1-D metric terms from the element grid arrays `x`. All the arrays
are preallocated by the user and the (square) derivative matrix `D` should be
consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x), Nq)` then the volume arrays
`x`, `J`, and `ξx` should all have length `Nq * nelem`.  Similarly, the face
arrays `sJ` and `nx` should be of length `nface * nelem` with `nface = 2`.
"""
function computemetric!(x, J, ξx, sJ, nx, D)
  Nq = size(D, 1)
  nelem = div(length(J), Nq)
  x = reshape(x, (Nq, nelem))
  J = reshape(J, (Nq, nelem))
  ξx = reshape(ξx, (Nq, nelem))
  nface = 2
  nx = reshape(nx, (1, nface, nelem))
  sJ = reshape(sJ, (1, nface, nelem))

  d = 1

  @inbounds for e = 1:nelem
    J[:, e] = D * x[:, e]
  end
  ξx .=  1 ./ J

  nx[1, 1, :] .= -sign.(J[ 1, :])
  nx[1, 2, :] .=  sign.(J[Nq, :])
  sJ .= 1
  nothing
end

"""
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)

Compute the 2-D metric terms from the element grid arrays `x` and `y`. All the
arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x), Nq^2)` then the volume arrays
`x`, `y`, `J`, `ξx`, `ηx`, `ξy`, and `ηy` should all be of size `(Nq, Nq,
nelem)`.  Similarly, the face arrays `sJ`, `nx`, and `ny` should be of size
`(Nq, nface, nelem)` with `nface = 4`.
"""
function computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  T = eltype(x)
  Nq = size(D, 1)
  nelem = div(length(J), Nq^2)
  d = 2
  x = reshape(x, (Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, nelem))
  J = reshape(J, (Nq, Nq, nelem))
  ξx = reshape(ξx, (Nq, Nq, nelem))
  ηx = reshape(ηx, (Nq, Nq, nelem))
  ξy = reshape(ξy, (Nq, Nq, nelem))
  ηy = reshape(ηy, (Nq, Nq, nelem))
  nface = 4
  nx = reshape(nx, (Nq, nface, nelem))
  ny = reshape(ny, (Nq, nface, nelem))
  sJ = reshape(sJ, (Nq, nface, nelem))

  @inbounds for e = 1:nelem
    for j = 1:Nq, i = 1:Nq
      xr = xs = zero(T)
      yr = ys = zero(T)
      for n = 1:Nq
        xr += D[i, n] * x[n, j, e]
        xs += D[j, n] * x[i, n, e]
        yr += D[i, n] * y[n, j, e]
        ys += D[j, n] * y[i, n, e]
      end
      J[i, j, e] = xr * ys - yr * xs
      ξx[i, j, e] =  ys / J[i, j, e]
      ηx[i, j, e] = -yr / J[i, j, e]
      ξy[i, j, e] = -xs / J[i, j, e]
      ηy[i, j, e] =  xr / J[i, j, e]
    end

    for i = 1:Nq
      nx[i, 1, e] = -J[ 1,  i, e] * ξx[ 1,  i, e]
      ny[i, 1, e] = -J[ 1,  i, e] * ξy[ 1,  i, e]
      nx[i, 2, e] =  J[Nq,  i, e] * ξx[Nq,  i, e]
      ny[i, 2, e] =  J[Nq,  i, e] * ξy[Nq,  i, e]
      nx[i, 3, e] = -J[ i,  1, e] * ηx[ i,  1, e]
      ny[i, 3, e] = -J[ i,  1, e] * ηy[ i,  1, e]
      nx[i, 4, e] =  J[ i, Nq, e] * ηx[ i, Nq, e]
      ny[i, 4, e] =  J[ i, Nq, e] * ηy[ i, Nq, e]

      for n = 1:4
        sJ[i, n, e] = hypot(nx[i, n, e], ny[i, n, e])
        nx[i, n, e] /= sJ[i, n, e]
        ny[i, n, e] /= sJ[i, n, e]
      end
    end
  end

  nothing
end

"""
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ, nx,
                   ny, nz, D)

Compute the 3-D metric terms from the element grid arrays `x`, `y`, and `z`. All
the arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x), Nq^3)` then the volume arrays
`x`, `y`, `z`, `J`, `ξx`, `ηx`, `ζx`, `ξy`, `ηy`, `ζy`, `ξz`, `ηz`, and `ζz`
should all be of length `Nq^3 * nelem`.  Similarly, the face arrays `sJ`, `nx`,
`ny`, and `nz` should be of size `Nq^2 * nface * nelem` with `nface = 6`.

The curl invariant formulation of Kopriva (2006), equation 37, is used.

Reference:
  Kopriva, David A. "Metric identities and the discontinuous spectral element
  method on curvilinear meshes." Journal of Scientific Computing 26.3 (2006):
  301-327. <https://doi.org/10.1007/s10915-005-9070-8>
"""
function computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ, nx,
                        ny, nz, D)
  T = eltype(x)

  Nq = size(D, 1)
  nelem = div(length(J), Nq^3)

  x = reshape(x, (Nq, Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, Nq, nelem))
  z = reshape(z, (Nq, Nq, Nq, nelem))
  J = reshape(J, (Nq, Nq, Nq, nelem))
  ξx = reshape(ξx, (Nq, Nq, Nq, nelem))
  ηx = reshape(ηx, (Nq, Nq, Nq, nelem))
  ζx = reshape(ζx, (Nq, Nq, Nq, nelem))
  ξy = reshape(ξy, (Nq, Nq, Nq, nelem))
  ηy = reshape(ηy, (Nq, Nq, Nq, nelem))
  ζy = reshape(ζy, (Nq, Nq, Nq, nelem))
  ξz = reshape(ξz, (Nq, Nq, Nq, nelem))
  ηz = reshape(ηz, (Nq, Nq, Nq, nelem))
  ζz = reshape(ζz, (Nq, Nq, Nq, nelem))

  nface = 6
  #= This code is broken when views are used
  nx = reshape(nx, Nq, Nq, nface, nelem)
  ny = reshape(ny, Nq, Nq, nface, nelem)
  nz = reshape(nz, Nq, Nq, nface, nelem)
  sJ = reshape(sJ, Nq, Nq, nface, nelem)
  =#

  JI2 = similar(J, (Nq, Nq, Nq))
  (yzr, yzs, yzt) = (similar(JI2), similar(JI2), similar(JI2))
  (zxr, zxs, zxt) = (similar(JI2), similar(JI2), similar(JI2))
  (xyr, xys, xyt) = (similar(JI2), similar(JI2), similar(JI2))

  ξx .= zero(T)
  ηx .= zero(T)
  ζx .= zero(T)
  ξy .= zero(T)
  ηy .= zero(T)
  ζy .= zero(T)
  ξz .= zero(T)
  ηz .= zero(T)
  ζz .= zero(T)

  @inbounds for e = 1:nelem
    for k = 1:Nq, j = 1:Nq, i = 1:Nq
      xr = xs = xt = zero(T)
      yr = ys = yt = zero(T)
      zr = zs = zt = zero(T)
      for n = 1:Nq
        xr += D[i, n] * x[n, j, k, e]
        xs += D[j, n] * x[i, n, k, e]
        xt += D[k, n] * x[i, j, n, e]
        yr += D[i, n] * y[n, j, k, e]
        ys += D[j, n] * y[i, n, k, e]
        yt += D[k, n] * y[i, j, n, e]
        zr += D[i, n] * z[n, j, k, e]
        zs += D[j, n] * z[i, n, k, e]
        zt += D[k, n] * z[i, j, n, e]
      end
      J[i, j, k, e] = (xr * (ys * zt - zs * yt) +
                       yr * (zs * xt - xs * zt) +
                       zr * (xs * yt - ys * xt))

      JI2[i,j,k] = 1 / (2 * J[i,j,k,e])

      yzr[i, j, k] = y[i, j, k, e] * zr - z[i, j, k, e] * yr
      yzs[i, j, k] = y[i, j, k, e] * zs - z[i, j, k, e] * ys
      yzt[i, j, k] = y[i, j, k, e] * zt - z[i, j, k, e] * yt
      zxr[i, j, k] = z[i, j, k, e] * xr - x[i, j, k, e] * zr
      zxs[i, j, k] = z[i, j, k, e] * xs - x[i, j, k, e] * zs
      zxt[i, j, k] = z[i, j, k, e] * xt - x[i, j, k, e] * zt
      xyr[i, j, k] = x[i, j, k, e] * yr - y[i, j, k, e] * xr
      xys[i, j, k] = x[i, j, k, e] * ys - y[i, j, k, e] * xs
      xyt[i, j, k] = x[i, j, k, e] * yt - y[i, j, k, e] * xt
    end

    for k = 1:Nq, j = 1:Nq, i = 1:Nq
      for n = 1:Nq
        ξx[i, j, k, e] += D[j, n] * yzt[i, n, k]
        ξx[i, j, k, e] -= D[k, n] * yzs[i, j, n]
        ηx[i, j, k, e] += D[k, n] * yzr[i, j, n]
        ηx[i, j, k, e] -= D[i, n] * yzt[n, j, k]
        ζx[i, j, k, e] += D[i, n] * yzs[n, j, k]
        ζx[i, j, k, e] -= D[j, n] * yzr[i, n, k]
        ξy[i, j, k, e] += D[j, n] * zxt[i, n, k]
        ξy[i, j, k, e] -= D[k, n] * zxs[i, j, n]
        ηy[i, j, k, e] += D[k, n] * zxr[i, j, n]
        ηy[i, j, k, e] -= D[i, n] * zxt[n, j, k]
        ζy[i, j, k, e] += D[i, n] * zxs[n, j, k]
        ζy[i, j, k, e] -= D[j, n] * zxr[i, n, k]
        ξz[i, j, k, e] += D[j, n] * xyt[i, n, k]
        ξz[i, j, k, e] -= D[k, n] * xys[i, j, n]
        ηz[i, j, k, e] += D[k, n] * xyr[i, j, n]
        ηz[i, j, k, e] -= D[i, n] * xyt[n, j, k]
        ζz[i, j, k, e] += D[i, n] * xys[n, j, k]
        ζz[i, j, k, e] -= D[j, n] * xyr[i, n, k]
      end
      ξx[i, j, k, e] *= JI2[i, j, k]
      ηx[i, j, k, e] *= JI2[i, j, k]
      ζx[i, j, k, e] *= JI2[i, j, k]
      ξy[i, j, k, e] *= JI2[i, j, k]
      ηy[i, j, k, e] *= JI2[i, j, k]
      ζy[i, j, k, e] *= JI2[i, j, k]
      ξz[i, j, k, e] *= JI2[i, j, k]
      ηz[i, j, k, e] *= JI2[i, j, k]
      ζz[i, j, k, e] *= JI2[i, j, k]
    end

    for j = 1:Nq, i = 1:Nq
      #= This code is broken when views are used
      nx[i, j, 1, e] = -J[ 1, i, j, e] * ξx[ 1, i, j, e]
      nx[i, j, 2, e] =  J[Nq, i, j, e] * ξx[Nq, i, j, e]
      nx[i, j, 3, e] = -J[ i, 1, j, e] * ηx[ i, 1, j, e]
      nx[i, j, 4, e] =  J[ i,Nq, j, e] * ηx[ i,Nq, j, e]
      nx[i, j, 5, e] = -J[ i, j, 1, e] * ζx[ i, j, 1, e]
      nx[i, j, 6, e] =  J[ i, j,Nq, e] * ζx[ i, j,Nq, e]
      ny[i, j, 1, e] = -J[ 1, i, j, e] * ξy[ 1, i, j, e]
      ny[i, j, 2, e] =  J[Nq, i, j, e] * ξy[Nq, i, j, e]
      ny[i, j, 3, e] = -J[ i, 1, j, e] * ηy[ i, 1, j, e]
      ny[i, j, 4, e] =  J[ i,Nq, j, e] * ηy[ i,Nq, j, e]
      ny[i, j, 5, e] = -J[ i, j, 1, e] * ζy[ i, j, 1, e]
      ny[i, j, 6, e] =  J[ i, j,Nq, e] * ζy[ i, j,Nq, e]
      nz[i, j, 1, e] = -J[ 1, i, j, e] * ξz[ 1, i, j, e]
      nz[i, j, 2, e] =  J[Nq, i, j, e] * ξz[Nq, i, j, e]
      nz[i, j, 3, e] = -J[ i, 1, j, e] * ηz[ i, 1, j, e]
      nz[i, j, 4, e] =  J[ i,Nq, j, e] * ηz[ i,Nq, j, e]
      nz[i, j, 5, e] = -J[ i, j, 1, e] * ζz[ i, j, 1, e]
      nz[i, j, 6, e] =  J[ i, j,Nq, e] * ζz[ i, j,Nq, e]

      for n = 1:6
        sJ[i, j, n, e] = hypot(nx[i, j, n, e], ny[i, j, n, e], nz[i, j, n, e])
        nx[i, j, n, e] /= sJ[i, j, n, e]
        ny[i, j, n, e] /= sJ[i, j, n, e]
        nz[i, j, n, e] /= sJ[i, j, n, e]
      end
      =#

      ije = i + (j-1) * Nq + (e-1) * nface * Nq^2
      nx[ije + (1-1) * Nq^2] = -J[ 1, i, j, e] * ξx[ 1, i, j, e]
      nx[ije + (2-1) * Nq^2] =  J[Nq, i, j, e] * ξx[Nq, i, j, e]
      nx[ije + (3-1) * Nq^2] = -J[ i, 1, j, e] * ηx[ i, 1, j, e]
      nx[ije + (4-1) * Nq^2] =  J[ i,Nq, j, e] * ηx[ i,Nq, j, e]
      nx[ije + (5-1) * Nq^2] = -J[ i, j, 1, e] * ζx[ i, j, 1, e]
      nx[ije + (6-1) * Nq^2] =  J[ i, j,Nq, e] * ζx[ i, j,Nq, e]
      ny[ije + (1-1) * Nq^2] = -J[ 1, i, j, e] * ξy[ 1, i, j, e]
      ny[ije + (2-1) * Nq^2] =  J[Nq, i, j, e] * ξy[Nq, i, j, e]
      ny[ije + (3-1) * Nq^2] = -J[ i, 1, j, e] * ηy[ i, 1, j, e]
      ny[ije + (4-1) * Nq^2] =  J[ i,Nq, j, e] * ηy[ i,Nq, j, e]
      ny[ije + (5-1) * Nq^2] = -J[ i, j, 1, e] * ζy[ i, j, 1, e]
      ny[ije + (6-1) * Nq^2] =  J[ i, j,Nq, e] * ζy[ i, j,Nq, e]
      nz[ije + (1-1) * Nq^2] = -J[ 1, i, j, e] * ξz[ 1, i, j, e]
      nz[ije + (2-1) * Nq^2] =  J[Nq, i, j, e] * ξz[Nq, i, j, e]
      nz[ije + (3-1) * Nq^2] = -J[ i, 1, j, e] * ηz[ i, 1, j, e]
      nz[ije + (4-1) * Nq^2] =  J[ i,Nq, j, e] * ηz[ i,Nq, j, e]
      nz[ije + (5-1) * Nq^2] = -J[ i, j, 1, e] * ζz[ i, j, 1, e]
      nz[ije + (6-1) * Nq^2] =  J[ i, j,Nq, e] * ζz[ i, j,Nq, e]

      for n = 1:6
        sJ[ije + (n-1) * Nq^2] = hypot(nx[ije + (n-1) *  Nq^2],
                                       ny[ije + (n-1) *  Nq^2],
                                       nz[ije + (n-1) *  Nq^2])
        nx[ije + (n-1) * Nq^2] /= sJ[ije + (n-1) *  Nq^2]
        ny[ije + (n-1) * Nq^2] /= sJ[ije + (n-1) *  Nq^2]
        nz[ije + (n-1) * Nq^2] /= sJ[ije + (n-1) *  Nq^2]
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

The grid is returned as a tuple of with `x` array
"""
function creategrid(::Val{1}, e2c::AbstractArray{S, 3},
                    r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 1
  Nq = length(r)
  x = Array{T, 2}(undef, Nq, nelem)
  creategrid!(x, e2c, r)
  (x=x, )
end

"""
    creategrid(::Val{2}, elemtocoord::AbstractArray{S, 3},
               r::AbstractVector{T}) where {S, T}

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

The grid is returned as a tuple of the `x` and `y` arrays
"""
function creategrid(::Val{2}, e2c::AbstractArray{S, 3},
                    r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 2
  Nq = length(r)
  x = Array{T, 3}(undef, Nq, Nq, nelem)
  y = Array{T, 3}(undef, Nq, Nq, nelem)
  creategrid!(x, y, e2c, r)
  (x=x, y=y)
end

"""
    creategrid(::Val{3}, elemtocoord::AbstractArray{S, 3},
               r::AbstractVector{T}) where {S, T}

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

The grid is returned as a tuple of the `x`, `y`, `z` arrays
"""
function creategrid(::Val{3}, e2c::AbstractArray{S, 3},
                    r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 3
  Nq = length(r)
  x = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
  y = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
  z = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
  creategrid!(x, y, z, e2c, r)
  (x=x, y=y, z=z)
end

"""
    computemetric(x::AbstractArray{T, 2}, D::AbstractMatrix{T}) where T

Compute the 1-D metric terms from the element grid array `x` using the
derivative matrix `D`. The derivative matrix `D` should be consistent with the
reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξx` derivative ∂r / ∂x'
 - `sJ` the surface Jacobian
 - 'nx` outward pointing unit normal in \$x\$-direction
"""
function computemetric(x::AbstractArray{T, 2},
                       D::AbstractMatrix{T}) where T

  Nq = size(D,1)
  nelem = size(x, 2)
  nface = 2

  J = similar(x)
  ξx = similar(x)

  sJ = Array{T, 3}(undef, 1, nface, nelem)
  nx = Array{T, 3}(undef, 1, nface, nelem)

  computemetric!(x, J, ξx, sJ, nx, D)

  (J=J, ξx=ξx, sJ=sJ, nx=nx)
end


"""
    computemetric(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                  D::AbstractMatrix{T}) where T

Compute the 2-D metric terms from the element grid arrays `x` and `y` using the
derivative matrix `D`. The derivative matrix `D` should be consistent with the
reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξx` derivative ∂r / ∂x'
 - `ηx` derivative ∂s / ∂x'
 - `ξy` derivative ∂r / ∂y'
 - `ηy` derivative ∂s / ∂y'
 - `sJ` the surface Jacobian
 - 'nx` outward pointing unit normal in \$x\$-direction
 - 'ny` outward pointing unit normal in \$y\$-direction
"""
function computemetric(x::AbstractArray{T, 3},
                       y::AbstractArray{T, 3},
                       D::AbstractMatrix{T}) where T
  @assert size(x) == size(y)
  Nq = size(D,1)
  nelem = size(x, 3)
  nface = 4

  J = similar(x)
  ξx = similar(x)
  ηx = similar(x)
  ξy = similar(x)
  ηy = similar(x)

  sJ = Array{T, 3}(undef, Nq, nface, nelem)
  nx = Array{T, 3}(undef, Nq, nface, nelem)
  ny = Array{T, 3}(undef, Nq, nface, nelem)

  computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)

  (J=J, ξx=ξx, ηx=ηx, ξy=ξy, ηy=ηy, sJ=sJ, nx=nx, ny=ny)
end

"""
    computemetric(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                  D::AbstractMatrix{T}) where T

Compute the 3-D metric terms from the element grid arrays `x`, `y`, and `z`
using the derivative matrix `D`. The derivative matrix `D` should be consistent
with the reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξx` derivative ∂r / ∂x'
 - `ηx` derivative ∂s / ∂x'
 - `ζx` derivative ∂t / ∂x'
 - `ξy` derivative ∂r / ∂y'
 - `ηy` derivative ∂s / ∂y'
 - `ζy` derivative ∂t / ∂y'
 - `ξz` derivative ∂r / ∂z'
 - `ηz` derivative ∂s / ∂z'
 - `ζz` derivative ∂t / ∂z'
 - `sJ` the surface Jacobian
 - 'nx` outward pointing unit normal in \$x\$-direction
 - 'ny` outward pointing unit normal in \$y\$-direction
 - 'nz` outward pointing unit normal in \$z\$-direction

!!! note

   The storage of the volume terms and surface terms from this function are
   slightly different. The volume terms used Cartesian indexing whereas the
   surface terms use linear indexing.
"""
function computemetric(x::AbstractArray{T, 4},
                       y::AbstractArray{T, 4},
                       z::AbstractArray{T, 4},
                       D::AbstractMatrix{T}) where T

  @assert size(x) == size(y) == size(z)
  Nq = size(D,1)
  nelem = size(x, 4)
  nface = 6

  J = similar(x)
  ξx = similar(x)
  ηx = similar(x)
  ζx = similar(x)
  ξy = similar(x)
  ηy = similar(x)
  ζy = similar(x)
  ξz = similar(x)
  ηz = similar(x)
  ζz = similar(x)

  sJ = Array{T, 3}(undef, Nq^2, nface, nelem)
  nx = Array{T, 3}(undef, Nq^2, nface, nelem)
  ny = Array{T, 3}(undef, Nq^2, nface, nelem)
  nz = Array{T, 3}(undef, Nq^2, nface, nelem)

  computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                 nx, ny, nz, D)

  (J=J, ξx=ξx, ηx=ηx, ζx=ζx, ξy=ξy, ηy=ηy, ζy=ζy, ξz=ξz, ηz=ηz, ζz=ζz, sJ=sJ,
   nx=nx, ny=ny, nz=nz)
end

end # module
