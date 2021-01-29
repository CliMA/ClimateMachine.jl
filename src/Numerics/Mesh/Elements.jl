module Elements
import GaussQuadrature

"""
    lglpoints(::Type{T}, N::Integer) where T <: AbstractFloat

returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre-Lobatto quadrature rule of type `T`

"""
function lglpoints(::Type{T}, N::Integer) where {T <: AbstractFloat}
    @assert N ≥ 1
    GaussQuadrature.legendre(T, N + 1, GaussQuadrature.both)
end

"""
    glpoints(::Type{T}, N::Integer) where T <: AbstractFloat

returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre quadrature rule of type `T`
"""
function glpoints(::Type{T}, N::Integer) where {T <: AbstractFloat}
    GaussQuadrature.legendre(T, N + 1, GaussQuadrature.neither)
end

"""
    baryweights(r)

returns the barycentric weights associated with the array of points `r`

Reference:
  [Berrut2004](@cite)
"""
function baryweights(r::AbstractVector{T}) where {T}
    Np = length(r)
    wb = ones(T, Np)

    for j in 1:Np
        for i in 1:Np
            if i != j
                wb[j] = wb[j] * (r[j] - r[i])
            end
        end
        wb[j] = T(1) / wb[j]
    end
    wb
end


"""
    spectralderivative(r::AbstractVector{T},
                       wb=baryweights(r)::AbstractVector{T}) where T

returns the spectral differentiation matrix for a polynomial defined on the
points `r` with associated barycentric weights `wb`

Reference:
 - [Berrut2004](@cite)
"""
function spectralderivative(
    r::AbstractVector{T},
    wb = baryweights(r)::AbstractVector{T},
) where {T}
    Np = length(r)
    @assert Np == length(wb)
    D = zeros(T, Np, Np)

    for k in 1:Np
        for j in 1:Np
            if k == j
                for l in 1:Np
                    if l != k
                        D[j, k] = D[j, k] + T(1) / (r[k] - r[l])
                    end
                end
            else
                D[j, k] = (wb[k] / wb[j]) / (r[j] - r[k])
            end
        end
    end
    D
end

"""
    interpolationmatrix(rsrc::AbstractVector{T}, rdst::AbstractVector{T},
                        wbsrc=baryweights(rsrc)::AbstractVector{T}) where T

returns the polynomial interpolation matrix for interpolating between the points
`rsrc` (with associated barycentric weights `wbsrc`) and `rdst`

Reference:
 - [Berrut2004](@cite)
"""
function interpolationmatrix(
    rsrc::AbstractVector{T},
    rdst::AbstractVector{T},
    wbsrc = baryweights(rsrc)::AbstractVector{T},
) where {T}
    Npdst = length(rdst)
    Npsrc = length(rsrc)
    @assert Npsrc == length(wbsrc)
    I = zeros(T, Npdst, Npsrc)
    for k in 1:Npdst
        for j in 1:Npsrc
            I[k, j] = wbsrc[j] / (rdst[k] - rsrc[j])
            if !isfinite(I[k, j])
                I[k, :] .= T(0)
                I[k, j] = T(1)
                break
            end
        end
        d = sum(I[k, :])
        I[k, :] = I[k, :] / d
    end
    I
end

"""
    function jacobip(α::Int, β::Int, np::Int, x::AbstractVector)

Returns a `(nx, np+1)` array containing the `np+1` Jacobi polynomials, with parameter `(α, β)`, evaluated on 1D grid `x`.
"""
function jacobip(
    α::Int,
    β::Int,
    np::Int,
    x::AbstractArray{FT, 1},
) where {FT <: AbstractFloat}
    nx = length(x)
    a = Vector{FT}(undef, 4)
    V = Array{FT}(undef, nx, np + 1)
    @assert np ≥ 0
    V .= 0.0
    V[:, 1] .= 1.0

    if np > 0
        V[:, 2] .= 0.5 .* (α .- β .+ (α .+ β .+ 2.0) .* x)
        if (np > 1)
            for i in 2:np
                a[1] = (2 * i) * (i + α + β) * (2 * i + α + β - 2)
                a[2] = (2 * i + α + β - 1) * (α * α - β * β)
                a[3] =
                    (2 * i + α + β - 2) * (2 * i + α + β - 1) * (2 * i + α + β)
                a[4] = 2 * (i + α - 1) * (i + β - 1) * (2 * i + α + β)

                V[:, i + 1] .=
                    ((a[2] .+ a[3] .* x) .* V[:, i] .- a[4] .* V[:, i - 1]) ./
                    a[1]
            end
        end
    end
    return V
end

end # module
